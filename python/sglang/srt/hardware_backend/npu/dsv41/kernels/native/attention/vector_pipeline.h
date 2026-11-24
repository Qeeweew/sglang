#pragma once
#include <common/device.h>
#include <kernel_operator.h>
#include <c_api/asc_simd.h>
#include <attention/state.h>

namespace attention {
using native_device::ub_ptr;
class VectorPipeline {
    const Sources &sources_;
    Workspace workspace_;
    RequestCursor cursor_;
    int32_t core_, subblock_, tasks_;
    int32_t compressed_batch_ = -1, compressed_available_ = 0, compressed_residual_ = 0;
    __aicore__ __attribute__((always_inline)) void gather_selected_rows(int32_t task) {
        const int32_t token = core_ + task * kWorkerBudget;
        const auto m = cursor_.locate<false>(sources_, token);
        const bool vector_coordinates = kCoordinate32 && kCompressedColumns <= kPageTableCapacity &&
                                        kCompressedColumns % 8 == 0;
        const bool new_request = compressed_batch_ != m.batch;
        auto page_table = ub_ptr<int32_t>(UbLayout::page_table);
        if (new_request) {
            compressed_batch_ = m.batch;
            compressed_available_ = gm_pointer<int32_t>(sources_.compressed_lengths)[m.batch];
            if constexpr (!kExplicitPositions)
                compressed_residual_ = gm_pointer<int32_t>(sources_.residual)[m.batch];
            if (vector_coordinates)
                asc_copy_gm2ub(page_table,
                               (gm_pointer<int32_t>(sources_.compressed_table) +
                                (int64_t(m.batch) * kCompressedColumns)),
                               uint32_t((kCompressedColumns) * sizeof(int32_t)));
        }
        const int64_t cutoff =
            (kExplicitPositions ? m.position_end
                                : max(int64_t(0), int64_t(compressed_available_) * kRatio +
                                                      compressed_residual_ - m.count + m.offset + 1)) >>
            (kRatio - 1);
        const int32_t limit = min(int64_t(compressed_available_), cutoff),
                      count = min(int32_t(kSelectedRows), limit);
        // For nonnegative IDs, page<columns is equivalent to an element
        // upper bound. Establish it once, avoiding a runtime page division
        // for every selected row on the vector-coordinate path.
        const int32_t valid_limit =
            min(int64_t(limit), int64_t(kCompressedColumns) * kCompressedPageRows);
        const int32_t owned_rows = kSelectedRows / 2;
        auto ids = ub_ptr<int32_t>(UbLayout::indices);
        auto coordinates = ub_ptr<int32_t>(UbLayout::row_indices);
        auto local_ids = ids + (subblock_ * owned_rows);
        auto local_coordinates = coordinates + (subblock_ * owned_rows);
        asc_copy_gm2ub(local_ids,
                       (gm_pointer<int32_t>(sources_.indices) +
                        (int64_t(token) * kSelectedRows + subblock_ * owned_rows)),
                       uint32_t((owned_rows) * sizeof(int32_t)));
        if (vector_coordinates) {
            native_device::pipe_fence<PIPE_MTE2, PIPE_V, kImmediateEvent>();
            auto offsets = ub_ptr<int32_t>(UbLayout::offsets);
            if (new_request) {
                for (int32_t col = 0; col < 8; ++col)
                    offsets[col] = col;
                native_device::pipe_fence<PIPE_S, PIPE_V, kImmediateEvent>();
                for (int32_t columns = 8; columns < kCompressedColumns; columns *= 2) {
                    asc_add_scalar((offsets + (columns)), offsets, columns, columns);
                    pipe_barrier(PIPE_V);
                }
                asc_sub(page_table, page_table, offsets, kCompressedColumns);
                pipe_barrier(PIPE_V);
                asc_shiftleft(reinterpret_cast<__ubuf__ uint32_t *>(page_table),
                              reinterpret_cast<__ubuf__ uint32_t *>(page_table),
                              uint32_t(kCoordinateShift), kCompressedColumns);
                pipe_barrier(PIPE_V);
            }
            // Logical shift bounds every result to INT32 nonnegative range.
            // Invalid IDs select a safe UB entry; the scalar predicate below
            // independently excludes their GM row transfer.
            asc_shiftright(reinterpret_cast<__ubuf__ uint32_t *>(offsets),
                           reinterpret_cast<__ubuf__ uint32_t *>(local_ids),
                           uint32_t(kCoordinateShift), owned_rows);
            pipe_barrier(PIPE_V);
            asc_min_scalar(offsets, offsets, kCompressedColumns - 1, owned_rows);
            pipe_barrier(PIPE_V);
            asc_shiftleft(reinterpret_cast<__ubuf__ uint32_t *>(offsets),
                          reinterpret_cast<__ubuf__ uint32_t *>(offsets), uint32_t(2), owned_rows);
            pipe_barrier(PIPE_V);
            asc_gather(reinterpret_cast<__ubuf__ uint32_t *>(local_coordinates),
                       reinterpret_cast<__ubuf__ uint32_t *>(page_table),
                       reinterpret_cast<__ubuf__ uint32_t *>(offsets), owned_rows);
            pipe_barrier(PIPE_V);
            asc_add(local_coordinates, local_coordinates, local_ids, owned_rows);
            pipe_barrier(PIPE_V);
            if (kCoordinateElements32) {
                asc_mul_scalar(local_coordinates, local_coordinates, int32_t(kWidth), owned_rows);
                pipe_barrier(PIPE_V);
            }
            native_device::pipe_fence<PIPE_V, PIPE_S, kImmediateEvent>();
        } else {
            native_device::pipe_fence<PIPE_MTE2, PIPE_S, kImmediateEvent>();
        }
        const int32_t chunks = owned_rows / kGatherRows;
        for (int32_t step = 0; step <= chunks; ++step) {
            if (step < chunks) {
                const int32_t slot = step % 2, first = subblock_ * owned_rows + step * kGatherRows;
                auto rows =
                    ub_ptr<BF16>(uint32_t(UbLayout::gather + slot * kGatherRows * kWidth * sizeof(BF16)));
                if (step >= 2 || task > 0) {
                    wait_flag(PIPE_MTE3, PIPE_MTE2, ::event_t(slot + 1));
                    wait_flag(PIPE_MTE3, PIPE_S, ::event_t(slot + 1));
                }
                auto masks =
                    ub_ptr<float>(uint32_t(UbLayout::masks + slot * kGatherRows * sizeof(float)));
                for (int32_t row = 0; row < kGatherRows; ++row) {
                    const int32_t selected_slot = first + row, index = ids[selected_slot];
                    const bool valid = selected_slot < count && index >= 0 && index < valid_limit;
                    masks[row] = valid ? 0.0f : -__builtin_inff();
                    if (valid) {
                        int64_t source;
                        if (vector_coordinates) {
                            source = coordinates[selected_slot];
                            if (!kCoordinateElements32)
                                source *= kWidth;
                        } else {
                            const int32_t logical_page = kCoordinateShift >= 0
                                                             ? (index >> kCoordinateShift)
                                                             : index / kCompressedPageRows;
                            const int32_t page = gm_pointer<int32_t>(
                                sources_.compressed_table)[int64_t(m.batch) * kCompressedColumns +
                                                           logical_page];
                            source = (int64_t(page) * kCompressedPageRows + index -
                                      int64_t(logical_page) * kCompressedPageRows) *
                                     kWidth;
                        }
                        asc_copy_gm2ub((rows + (row * kWidth)),
                                       (gm_pointer<BF16>(sources_.compressed) + (source)),
                                       uint32_t((kWidth) * sizeof(BF16)));
                    }
                }
                set_flag(PIPE_MTE2, PIPE_MTE3, ::event_t(slot + 1));
                set_flag(PIPE_S, PIPE_MTE3, ::event_t(slot + 1));
            }
            if (step > 0) {
                const int32_t chunk = step - 1, slot = chunk % 2,
                              first = subblock_ * owned_rows + chunk * kGatherRows;
                auto masks =
                    ub_ptr<float>(uint32_t(UbLayout::masks + slot * kGatherRows * sizeof(float)));
                auto rows =
                    ub_ptr<BF16>(uint32_t(UbLayout::gather + slot * kGatherRows * kWidth * sizeof(BF16)));
                wait_flag(PIPE_S, PIPE_MTE3, ::event_t(slot + 1));
                asc_copy_ub2gm((workspace_.bias(task) + (first)), masks,
                               uint32_t((kGatherRows) * sizeof(float)));
                set_flag(PIPE_MTE3, PIPE_S, ::event_t(slot + 1));
                wait_flag(PIPE_MTE2, PIPE_MTE3, ::event_t(slot + 1));
                asc_copy_ub2gm((workspace_.gathered(task) + (first * kWidth)), rows,
                               uint32_t((kGatherRows * kWidth) * sizeof(BF16)));
                set_flag(PIPE_MTE3, PIPE_MTE2, ::event_t(slot + 1));
            }
        }
        CrossCoreSetFlag<2, PIPE_MTE3>(GatherReady);
    }
    template <bool Sum>
    __aicore__ __attribute__((always_inline)) void
    reduce_rows(__ubuf__ float *dst, __ubuf__ float *source, int32_t rows, int32_t columns) {
        auto partial = ub_ptr<float>(UbLayout::partial),
             merge = ub_ptr<float>(UbLayout::reduction_merge);

        // First reduce each 32B block, then reduce packed block results. This
        // retains the established floating-point sum tree for every row.
        // One counter-mode block reduction, including a long row tile that
        // exceeds the normal-mask 255-repeat limit. The packed block8 tree
        // and subsequent row merge are identical to the normal-mask path.
        asc_set_mask_count();
        asc_set_vector_mask(0, rows * columns);
        if constexpr (Sum)
            asc_datablock_reduce_sum(partial, source, 1, 1, 1, 8);
        else
            asc_datablock_reduce_max(partial, source, 1, 1, 1, 8);
        asc_set_mask_norm();
        pipe_barrier(PIPE_V);
        for (int32_t offset = 0; offset < columns / kBlockElements; offset += kVectorElements) {
            const int32_t count =
                min(int32_t(kVectorElements), columns / int32_t(kBlockElements) - offset);
            auto output = offset ? merge : dst;
            {
                asc_set_vector_mask(0, (count) == 64 ? ~uint64_t(0) : (uint64_t(1) << (count)) - 1);
                if constexpr (Sum)
                    asc_repeat_reduce_sum(output, (partial + (offset)), uint8_t(rows), 1, 1,
                                          columns / kVectorElements);
                else
                    asc_repeat_reduce_max_only_value(output, (partial + (offset)), uint8_t(rows), 1,
                                                     1, columns / kVectorElements);
            };
            if (offset || columns / kBlockElements <= kVectorElements)
                pipe_barrier(PIPE_V);
            if (offset) {
                if constexpr (Sum)
                    asc_add(dst, dst, merge, rows);
                else
                    asc_max(dst, dst, merge, rows);
                pipe_barrier(PIPE_V);
            }
        }
    }
    template <bool Multiply>
    __aicore__ __attribute__((always_inline)) void
    broadcast_rows(__ubuf__ float *matrix, __ubuf__ float *values, int32_t rows, int32_t columns) {
        if (columns > 1024) {
            // Long sequence rows use one scalar-vector instruction each. The
            // dependency retires the statistics producer before Scalar reads;
            // max/sub and reciprocal/multiply retain their FP32 operations.
            native_device::pipe_fence<PIPE_V, PIPE_S, kImmediateEvent>();
            for (int32_t row = 0; row < rows; ++row) {
                const float value = values[row];
                if constexpr (Multiply)
                    asc_mul_scalar((matrix + (row * columns)), (matrix + (row * columns)), value,
                                   columns);
                else
                    asc_add_scalar((matrix + (row * columns)), (matrix + (row * columns)), -value,
                                   columns);
            }
            return;
        }
        auto expanded = ub_ptr<float>(UbLayout::broadcast);
        asc_brcb(reinterpret_cast<__ubuf__ uint32_t *>(expanded),
                 reinterpret_cast<__ubuf__ uint32_t *>(values), 1, 8, uint8_t((rows + 7) / 8));
        pipe_barrier(PIPE_V);
        asc_set_mask_norm();
        asc_set_vector_mask(0, ~uint64_t(0));

        for (int32_t col = 0; col < columns; col += kVectorElements) {
            const int32_t active = min(int32_t(kVectorElements), columns - col);
            if (active < int32_t(kVectorElements))
                asc_set_vector_mask(0, (uint64_t(1) << active) - 1);
            if constexpr (Multiply)
                asc_mul((matrix + (col)), (matrix + (col)), expanded, uint8_t(rows), 1, 1, 0,
                        uint8_t(columns / 8), uint8_t(columns / 8), 1);
            else
                asc_sub((matrix + (col)), (matrix + (col)), expanded, uint8_t(rows), 1, 1, 0,
                        uint8_t(columns / 8), uint8_t(columns / 8), 1);
        }
    }

    __aicore__ __attribute__((always_inline)) void softmax(int32_t task) {
        const int32_t token = core_ + task * kWorkerBudget, local_heads = kHeads / 2;
        const auto m = cursor_.locate(sources_, token);
        auto bias = ub_ptr<float>(UbLayout::bias);
        const int32_t direct_columns = kSparse ? kSparseWindowCapacity : kCapacity;
        // A fully visible direct window has zero bias. Its FP32 addition is
        // an identity, so initialize/apply only the genuinely masked columns.
        // Sparse selected columns always load their separate gather masks.
        const int32_t bias_first = m.visible < direct_columns ? 0 : direct_columns;
        if (bias_first == 0) {
            asc_duplicate(bias, 0.0f, direct_columns);
            pipe_barrier(PIPE_V);
            for (int32_t column = 0; column < direct_columns; column += kVectorElements) {
                uint64_t valid;
                if (kSparse && kWindowRing) {
                    const int32_t first = m.first % kDotRows;
                    valid = valid_interval(first - column, first + m.visible - column) |
                            valid_interval(-column, first + m.visible - kDotRows - column);
                } else
                    valid = valid_interval(-column, m.visible - column);
                uint64_t mask[2]{~valid, 0};
                if (mask[0]) {
                    asc_set_vector_mask(mask[1], mask[0]);
                    asc_duplicate((bias + (column)), -__builtin_inff(), 1, 1, 8);
                };
            }
        }
        CrossCoreWaitFlag<2, PIPE_MTE2>(ScoreReady);
        if constexpr (kSparse)
            asc_copy_gm2ub((bias + (kSparseWindowCapacity)), workspace_.bias(task),
                           uint32_t((kSelectedRows) * sizeof(float)));
        const int32_t head_tile = min(local_heads, kSparse             ? int32_t(kSparseHeadTile)
                                                   : kCapacity <= 512  ? 32
                                                   : kCapacity <= 1024 ? 16
                                                                       : 4);
        auto probabilities = ub_ptr<BF16>(UbLayout::probabilities);
        auto maximum = ub_ptr<float>(UbLayout::maximum), sum = ub_ptr<float>(UbLayout::sum);
        if constexpr (kSparse) {
            for (int32_t local = 0; local < local_heads; local += head_tile) {
                const int32_t rows = min(head_tile, local_heads - local);
                asc_copy_gm2ub(
                    ub_ptr<float>(uint32_t(UbLayout::scores + local * kCapacity * sizeof(float))),
                    (workspace_.scores(task) + ((subblock_ * local_heads + local) * kCapacity)),
                    uint32_t((rows * kCapacity) * sizeof(float)));
                set_flag(PIPE_MTE2, PIPE_V, ::event_t(local / head_tile + 1));
            }
        }
        for (int32_t local = 0; local < local_heads; local += head_tile) {
            const int32_t rows = min(head_tile, local_heads - local),
                          head = subblock_ * local_heads + local;
            auto scores = ub_ptr<float>(uint32_t(UbLayout::scores +
                                            (kSparse ? local * kCapacity * sizeof(float) : 0)));
            auto sink = ub_ptr<float>(uint32_t(UbLayout::sink + (kSparse ? local * sizeof(float) : 0)));
            auto sink_input =
                kSparse ? ub_ptr<float>(uint32_t(UbLayout::sinks + local * sizeof(float))) : sink;
            if constexpr (kSparse)
                wait_flag(PIPE_MTE2, PIPE_V, ::event_t(local / head_tile + 1));
            else {
                asc_copy_gm2ub(scores, (workspace_.scores(task) + (head * kCapacity)),
                               uint32_t((rows * kCapacity) * sizeof(float)));
                if (rows >= 8)
                    asc_copy_gm2ub(sink, (gm_pointer<float>(sources_.sinks) + (head)),
                                   uint32_t((rows) * sizeof(float)));
                else {
                    for (int32_t row = 0; row < rows; ++row)
                        sink[row] = gm_pointer<float>(sources_.sinks)[head + row];
                    native_device::pipe_fence<PIPE_S, PIPE_V, kImmediateEvent>();
                }
                native_device::pipe_fence<PIPE_MTE2, PIPE_V, kImmediateEvent>();
            }
            asc_mul_scalar(scores, scores, kScale, rows * kCapacity);
            pipe_barrier(PIPE_V);
            if (bias_first < kCapacity) {
                if (kCapacity <= 1024) {
                    asc_set_mask_norm();
                    asc_set_vector_mask(0, ~uint64_t(0));

                    for (int32_t col = bias_first; col < kCapacity; col += kVectorElements)
                        asc_add((scores + (col)), (scores + (col)), (bias + (col)), uint8_t(rows),
                                1, 1, 1, uint8_t(kCapacity / 8), uint8_t(kCapacity / 8), 0);
                } else
                    for (int32_t row = 0; row < rows; ++row)
                        asc_add((scores + (row * kCapacity + bias_first)),
                                (scores + (row * kCapacity + bias_first)), (bias + (bias_first)),
                                kCapacity - bias_first);
                pipe_barrier(PIPE_V);
            }
            reduce_rows<false>(maximum, scores, rows, kCapacity);
            asc_max(maximum, maximum, sink_input, rows);
            pipe_barrier(PIPE_V);
            asc_sub(sink, sink_input, maximum, rows);
            pipe_barrier(PIPE_V);
            asc_exp(sink, sink, rows);
            broadcast_rows<false>(scores, maximum, rows, kCapacity);
            pipe_barrier(PIPE_V);
            asc_exp(scores, scores, rows * kCapacity);
            pipe_barrier(PIPE_V);
            if constexpr (kSparse) {
                asc_float2bfloat16_rn(probabilities, scores, rows * kCapacity);
                native_device::pipe_fence<PIPE_V, PIPE_MTE3, kImmediateEvent>();
                asc_copy_ub2gm((workspace_.probabilities(task) + (head * kCapacity)), probabilities,
                               uint32_t((rows * kCapacity) * sizeof(BF16)));
                if (local + rows == local_heads)
                    CrossCoreSetFlag<2, PIPE_MTE3>(ProbabilityReady);
            }
            reduce_rows<true>(sum, scores, rows, kCapacity);
            asc_add(sum, sum, sink, rows);
            pipe_barrier(PIPE_V);
            if constexpr (kSparse)
                asc_copy_ub2ub(
                    ub_ptr<float>(uint32_t(UbLayout::final_denominator +
                                      ((task % 2) * local_heads + local) * sizeof(float))),
                    sum, uint32_t((rows) * sizeof(float)));
            else {
                asc_duplicate(maximum, 1.0f, rows);
                pipe_barrier(PIPE_V);
                asc_div(sink, maximum, sum, rows);
                pipe_barrier(PIPE_V);
                if constexpr (kDraft) {
                    auto refine = ub_ptr<float>(UbLayout::reciprocal_refinement);
                    asc_mul(refine, sink, sum, rows);
                    pipe_barrier(PIPE_V);
                    asc_mul_scalar(refine, refine, -1.0f, rows);
                    pipe_barrier(PIPE_V);
                    asc_add_scalar(refine, refine, 2.0f, rows);
                    pipe_barrier(PIPE_V);
                    asc_mul(sink, sink, refine, rows);
                    pipe_barrier(PIPE_V);
                }
                broadcast_rows<true>(scores, sink, rows, kCapacity);
                pipe_barrier(PIPE_V);
                asc_float2bfloat16_rn(probabilities, scores, rows * kCapacity);
                native_device::pipe_fence<PIPE_V, PIPE_MTE3, kImmediateEvent>();
                asc_copy_ub2gm((workspace_.probabilities(task) + (head * kCapacity)), probabilities,
                               uint32_t((rows * kCapacity) * sizeof(BF16)));
                // Publish on the store pipe before waiting for UB reuse. The
                // final group covers all preceding P stores for this task.
                if (local + rows == local_heads)
                    CrossCoreSetFlag<2, PIPE_MTE3>(ProbabilityReady);
            }
            // Direct attention has no UB reader or writer after its final P
            // export. Intermediate tasks/groups still retire both reuse edges.
            if (kSparse || task + 1 < tasks_ || local + rows < local_heads) {
                native_device::pipe_fence<PIPE_MTE3, PIPE_V, kImmediateEvent>();
                native_device::pipe_fence<PIPE_V, PIPE_MTE2, kImmediateEvent>();
            }
        }
    }
    __aicore__ __attribute__((always_inline)) void finish(int32_t task) {
        CrossCoreWaitFlag<2, PIPE_MTE2>(ProductReady);
        const int32_t token = core_ + task * kWorkerBudget, rows = kHeads / 2,
                      head = subblock_ * rows;
        auto values = ub_ptr<float>(UbLayout::scores);
        auto output = ub_ptr<BF16>(UbLayout::probabilities);
        auto inverse = ub_ptr<float>(UbLayout::sink),
             ones = ub_ptr<float>(UbLayout::maximum);
        auto denominator =
            ub_ptr<float>(uint32_t(UbLayout::final_denominator + (task % 2) * rows * sizeof(float)));
        asc_copy_gm2ub(values, (workspace_.products(task) + (head * kWidth)),
                       uint32_t((rows * kWidth) * sizeof(float)));
        asc_duplicate(ones, 1.0f, rows);
        pipe_barrier(PIPE_V);
        asc_div(inverse, ones, denominator, rows);
        pipe_barrier(PIPE_V);
        native_device::pipe_fence<PIPE_MTE2, PIPE_V, kImmediateEvent>();
        broadcast_rows<true>(values, inverse, rows, kWidth);
        pipe_barrier(PIPE_V);
        asc_float2bfloat16_rn(output, values, rows * kWidth);
        native_device::pipe_fence<PIPE_V, PIPE_MTE3, kImmediateEvent>();
        asc_copy_ub2gm(
            (gm_pointer<BF16>(sources_.out) + ((int64_t(token) * kHeads + head) * kWidth)), output,
            uint32_t((rows * kWidth) * sizeof(BF16)));
        native_device::pipe_fence<PIPE_MTE3, PIPE_V, kImmediateEvent>();
        native_device::pipe_fence<PIPE_V, PIPE_MTE2, kImmediateEvent>();
    }

  public:
    __aicore__
        __attribute__((always_inline)) explicit VectorPipeline(const Sources &args, int32_t actor,
                                                               int32_t subblock)
        : sources_(args), workspace_(args, actor), core_(actor), subblock_(subblock) {
        tasks_ = sources_.tokens <= kWorkerBudget
                     ? int32_t(core_ < sources_.tokens)
                     : (max(0, sources_.tokens - core_) + kWorkerBudget - 1) / kWorkerBudget;
    }
    __aicore__ __attribute__((always_inline)) void run() {
        if constexpr (kSparse) {
            asc_duplicate(ub_ptr<uint16_t>(UbLayout::gather), uint16_t(0),
                          kGatherBytes / sizeof(uint16_t));
            native_device::pipe_fence<PIPE_V, PIPE_MTE2, kImmediateEvent>();
            asc_copy_gm2ub(ub_ptr<float>(UbLayout::sinks),
                           (gm_pointer<float>(sources_.sinks) + (subblock_ * kHeads / 2)),
                           uint32_t((kHeads / 2) * sizeof(float)));
            native_device::pipe_fence<PIPE_MTE2, PIPE_V, kImmediateEvent>();
        }
        if (!kSparse && tasks_ == 1) {
            if constexpr (kSparse)
                gather_selected_rows(0);
            softmax(0);
            if constexpr (kSparse)
                finish(0);
        } else
            for (int32_t step = 0; step < tasks_ + (kSparse ? 2 : 1); ++step) {
                if constexpr (kSparse)
                    if (step < tasks_)
                        gather_selected_rows(step);
                if (step > 0 && step <= tasks_)
                    softmax(step - 1);
                if constexpr (kSparse)
                    if (step > 1)
                        finish(step - 2);
            }
        if constexpr (kSparse)
            for (int32_t slot = 0; slot < 2; ++slot) {
                wait_flag(PIPE_MTE3, PIPE_MTE2, ::event_t(slot + 1));
                wait_flag(PIPE_MTE3, PIPE_S, ::event_t(slot + 1));
            }
        if constexpr (kSparse) {
            native_device::pipe_fence<PIPE_V, PIPE_S, kImmediateEvent>();
            native_device::pipe_fence<PIPE_MTE2, PIPE_S, kImmediateEvent>();
        }
        // For direct attention every input DMA feeds V, and every V operation
        // feeds the final P export. MTE3 therefore transitively retires them.
        native_device::pipe_fence<PIPE_MTE3, PIPE_S, kImmediateEvent>();
    }
};
} // namespace attention

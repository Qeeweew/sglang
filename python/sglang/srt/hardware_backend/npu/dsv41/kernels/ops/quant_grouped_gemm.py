"""Thin tensor interface to the linked C++ quantized grouped GEMM."""

from __future__ import annotations

_FIELDS = (
    "K",
    "N",
    "BM",
    "BN",
    "STEP_K",
    "CK",
    "LOCAL_ELEMENTS",
    "OUTPUT_BYTES",
    "MICRO_ELEMENTS",
)


def geometry(k, n, *, block_m=16):
    """Inspect C++ runtime tiling against the compiled tile resource budgets."""
    from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
        load_library,
    )

    return dict(zip(_FIELDS, load_library().quant_gmm_geometry(k, n, block_m)))


def select_geometry(k, n, *, rows, mean, bound, cores):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
        load_library,
    )

    bm = load_library().quant_gmm_select(k, n, rows, mean, bound, cores)
    return geometry(k, n, block_m=bm)


class NativeGroupedGemm:
    """FP16 A/C, FP32 accumulation; C++ launches the kernel; Python caches per-stream workspace.

    block_m=0 selects tiling from caller-known row bounds. Keep tensors alive
    during Graph replay, and warm the workspace on the capture stream.
    Weights use native NZ C0=16 compressed order, independent of BM/BN.
    MXFP4 scales use startup-packed uint8 natural-order FP16 high bytes,
    with sign-tagged subnormals. Routing uses compact INT32 metadata:
    [active, valid_rows, 0, 1] followed by [expert, begin, rows, 0].
    """

    uses_metadata = True  # Fixed routing contract for local-expert composition.

    def __init__(
        self,
        k,
        n,
        *,
        experts,
        device=None,
        block_m=0,
    ):
        import torch
        from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
            load_library,
            npu_device,
        )

        library = load_library()
        self.device = npu_device(device or f"npu:{torch.npu.current_device()}")
        self.k, self.n, self.experts = k, n, experts
        self.adaptive = block_m == 0
        self.block_m, self._workspaces = block_m, {}
        self.params = geometry(k, n, block_m=block_m or 16)
        self.cores = library.quant_gmm_cores(self.device.index)

    def geometry_for_rows(self, rows, active):
        if not self.adaptive:
            return self.params
        return select_geometry(
            self.k,
            self.n,
            rows=rows,
            mean=rows / active if active else 0,
            bound=rows,
            cores=self.cores,
        )

    def geometry_for_counts(self, counts):
        if not self.adaptive:
            return self.params
        rows, active = sum(counts), sum(v > 0 for v in counts)
        return select_geometry(
            self.k,
            self.n,
            rows=rows,
            mean=rows / active if active else 0,
            bound=max(counts, default=0),
            cores=self.cores,
        )

    def bind(
        self,
        x,
        packed,
        scales,
        routing,
        *,
        out,
        max_expert_rows=None,
        mean_expert_rows=None,
        tiling_rows=None,
        workspace=None,
    ):
        import math

        import torch
        from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
            check_tensor,
            load_library,
            require_disjoint,
        )

        if packed.shape[0] != self.experts:
            raise ValueError("expert count differs from the operator")
        rows = x.shape[0]
        # Allocation capacity and the caller's established tiling budget need
        # not match after eager EP compaction. This hint never changes access
        # bounds, input/output shapes, or the device routing metadata.
        budget = rows if tiling_rows is None else tiling_rows
        bound = rows if max_expert_rows is None else max_expert_rows
        mean = rows / self.experts if mean_expert_rows is None else mean_expert_rows
        if not 0 <= bound <= rows <= budget or not math.isfinite(mean) or mean < 0:
            raise ValueError("valid host-known expert row bounds required")
        params = (
            select_geometry(
                self.k, self.n, rows=budget, mean=mean, bound=bound, cores=self.cores
            )
            if self.adaptive
            else self.params
        )
        bm = params["BM"]
        check_tensor(x, torch.float16, (rows, self.k), self.device, "input")
        check_tensor(
            packed,
            torch.int32,
            (self.experts, self.k * self.n // 8),
            self.device,
            "packed",
        )
        check_tensor(
            scales,
            torch.uint8,
            (self.experts, self.k * self.n // 32),
            self.device,
            "scales",
        )
        check_tensor(out, torch.float16, (rows, self.n), self.device, "output")
        check_tensor(routing, torch.int32, None, self.device, "routing")
        route_capacity = 4 * (self.experts + 1)
        if routing.ndim != 1 or routing.numel() < route_capacity:
            raise ValueError("compact metadata capacity must cover all experts")
        stream = torch.npu.current_stream(self.device)
        count = self.cores * 2 * params["BN"] * params["STEP_K"]
        if workspace is None:
            key = (stream.npu_stream, bm)
            if key not in self._workspaces:
                if torch.npu.is_current_stream_capturing():
                    raise RuntimeError("warm native workspace on the capture stream")
                self._workspaces[key] = x.new_empty(count)
            workspace = self._workspaces[key]
        check_tensor(workspace, torch.float16, None, self.device, "workspace")
        if workspace.numel() < count:
            raise ValueError(f"workspace needs {count} FP16 elements")
        require_disjoint((out, workspace), (x, packed, scales, routing))
        op = load_library().quant_gmm
        with torch.npu.device(self.device):
            load_library().prepare_quant_gmm(
                x, packed, scales, routing, out, workspace, bm
            )

        def invoke():
            op(x, packed, scales, routing, out, workspace, bm)

        invoke.params, invoke.workspace, invoke.block_m = params, workspace, bm
        invoke.kernel_names = (f"quant_grouped_gemm_bm{bm}_bn{params['BN']}",)
        return invoke

    def __call__(self, x, packed, scales, routing, *, out=None, **bounds):
        out = x.new_empty((x.shape[0], self.n)) if out is None else out
        self.bind(x, packed, scales, routing, out=out, **bounds)()
        return out


class MXFP4LocalExperts:
    """Two prepared GMMs and the DSv4.1 clipped SiLU, in sorted local row order.

    The two weight pairs are (P,S2). GMM1 columns are [gate_local,up_local];
    GMM2 returns this rank's down-projection contribution. EP combine and TP
    all-reduce belong to the caller. All expert arithmetic uses FP16 operands,
    FP32 GEMM accumulation, and FP32 activation statistics.
    """

    def __init__(
        self,
        gate_up,
        down,
        gate_up_weight,
        down_weight,
        *,
        limit=10.0,
        activation_op=None,
    ):
        if (
            gate_up.experts != down.experts
            or gate_up.k != down.n
            or gate_up.n != 2 * down.k
            or limit <= 0
        ):
            raise ValueError(
                "inconsistent local gate/up/down geometry or activation limit"
            )
        self.gate_up, self.down = gate_up, down
        self.gate_up_weight, self.down_weight = gate_up_weight, down_weight
        self.limit = float(limit)
        if activation_op is None:
            from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.elementwise import (
                PackedClippedSwiGLU,
            )

            activation_op = PackedClippedSwiGLU(
                dimension=down.k,
                dtype="f16",
                limit=self.limit,
                prefix=True,
            )
        if (
            activation_op.d != down.k
            or activation_op.limit != self.limit
            or not activation_op.prefix
            or activation_op.dtype != "f16"
        ):
            raise ValueError(
                "activation geometry/clamp/FP16 dtype differs from the local experts"
            )
        self.activation_op = activation_op

    def bind(
        self,
        x,
        routing,
        *,
        gate_up,
        activation,
        out,
        max_expert_rows=None,
        mean_expert_rows=None,
        tiling_rows=None,
    ):
        """Bind two GEMMs and one activation to caller-owned stable buffers."""
        hints = {} if tiling_rows is None else {"tiling_rows": tiling_rows}
        gate = self.gate_up.bind(
            x,
            *self.gate_up_weight,
            routing,
            out=gate_up,
            max_expert_rows=max_expert_rows,
            mean_expert_rows=mean_expert_rows,
            **hints,
        )
        rows = routing[1:2]
        activate = self.activation_op.bind(gate_up, out=activation, rows=rows)
        down = self.down.bind(
            activation,
            *self.down_weight,
            routing,
            out=out,
            max_expert_rows=max_expert_rows,
            mean_expert_rows=mean_expert_rows,
            **hints,
        )

        def invoke():
            gate()
            activate()
            down()
            return out

        invoke.projections = (gate, down)
        return invoke

    def __call__(
        self,
        x,
        routing,
        *,
        gate_up=None,
        activation=None,
        out=None,
        max_expert_rows=None,
        mean_expert_rows=None,
        tiling_rows=None,
    ):
        gate_up = (
            x.new_empty((x.shape[0], self.gate_up.n)) if gate_up is None else gate_up
        )
        activation = (
            x.new_empty((x.shape[0], self.down.k)) if activation is None else activation
        )
        out = x.new_empty((x.shape[0], self.down.n)) if out is None else out
        return self.bind(
            x,
            routing,
            gate_up=gate_up,
            activation=activation,
            out=out,
            max_expert_rows=max_expert_rows,
            mean_expert_rows=mean_expert_rows,
            tiling_rows=tiling_rows,
        )()

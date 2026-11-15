"""Request-private BF16 SWA through the existing paged attention ABI.

The paged workspace stages accepted history and the current queries before
attention. Both buffers are allocated before Graph capture.
"""

from types import SimpleNamespace

import torch


class BF16RequestWindow:
    def __init__(
        self,
        pool_factory,
        *,
        num_slots,
        layers,
        page_size,
        capacity,
        workspace_rows=None,
    ):
        from sglang.srt.hardware_backend.npu.dsv41.kernels.ops import CED

        self.page_size = page_size
        self.capacity = ((capacity + page_size - 1) // page_size) * page_size
        self.num_slots = num_slots
        self.rows = num_slots * self.capacity
        self.device = torch.device("npu", torch.npu.current_device())
        self.ring = torch.zeros(
            (layers, self.rows + page_size, 512),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.state = SimpleNamespace(
            kv_buffer=list(self.ring.unbind()),
            dtype=torch.bfloat16,
        )
        self.ring_views = [
            value.view(-1, page_size, 1, 512) for value in self.state.kv_buffer
        ]
        # One alignment page per request, in addition to its 128 history rows.
        size = int(workspace_rows or 0) + (num_slots + 1) * page_size
        size = ((size + page_size - 1) // page_size) * page_size
        self.workspace = SimpleNamespace(
            size=size,
            kv_buffer=[
                torch.zeros(
                    (size // page_size, page_size, 1, 512),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
            ],
        )
        self.movement = CED(device=self.device)
        self.valid_starts = {}
        self.plan = None
        self.prepared = None

    def reset(self, slots):
        self.prepared = None

    def initialize_dummy_history(self):
        # Request zero has no accepted rows. Its capture-time history is masked
        # in prepare(), so no fabricated state needs to enter the real rings.
        self.prepared = None

    def activate(self, plan):
        self.plan = plan
        self.prepared = None

    def prepare(self, meta, *, width=None, counts=None, floors=None):
        """Build a page table for just history + queries, using absolute positions.

        Eager uses host-known variable lengths. Decode/verify use a fixed page
        stride and device positions; all tensor addresses belong to the bucket.
        """
        b, n, p = meta.batch, meta.rows, self.page_size
        if not hasattr(meta, "ring_plan"):

            def empty(size, dtype):
                return torch.empty(size, dtype=dtype, device=self.device)

            meta.ring_plan = SimpleNamespace(
                history_src=empty(b * 128, torch.int32),
                history_dst=empty(b * 128, torch.int32),
                ring_dst=empty(n, torch.int32),
                commit_dst=empty(n, torch.int32),
                group=empty(n, torch.int64),
                positions=empty(n, torch.int64),
            )
        plan = meta.ring_plan
        if counts is not None:
            starts = meta.starts.cpu()
            first = ((starts - 128).clamp_min(0) // p) * p
            spans = [
                max(1, (int(s - f) + c + p - 1) // p)
                for s, f, c in zip(starts, first, counts)
            ]
            bases = torch.tensor([1] + spans, dtype=torch.int64).cumsum(0)[:-1]
            required = (1 + sum(spans)) * p
            group = torch.repeat_interleave(torch.arange(b), torch.tensor(counts))
            offsets = torch.cat([torch.arange(c) for c in counts])
            pos = starts[group] + offsets
            ends = torch.tensor(counts).cumsum(0)
            plan.commit_rows = torch.cat(
                [
                    torch.arange(
                        int(end) - min(c, self.capacity), int(end), dtype=torch.int32
                    )
                    for c, end in zip(counts, ends)
                ]
            ).to(self.device)
            first, bases, group, pos = [
                v.to(self.device) for v in (first, bases, group, pos)
            ]
        else:
            # At most two history pages and two query pages for a six-row verify.
            stride = (128 + (p - 1) + width + p - 1) // p
            required = (1 + b * stride) * p
            first = ((meta.starts - 128).clamp_min(0) // p) * p
            bases = 1 + torch.arange(b, device=self.device) * stride
            group = torch.arange(b, device=self.device).repeat_interleave(width)
            pos = meta.starts[group] + torch.arange(width, device=self.device).repeat(b)
        if required > self.workspace.size:
            raise ValueError(
                f"SWA workspace needs {required} rows, has {self.workspace.size}"
            )
        cols = torch.arange(meta.table.shape[1], device=self.device) * p
        page = cols[None, :] // p - first[:, None] // p + bases[:, None]
        live_page = (cols[None, :] >= first[:, None]) & (
            cols[None, :] < meta.lengths[:, None]
        )
        meta.table.copy_(torch.where(live_page, page, 0).int())
        live = meta.accepted[group] > 0
        meta.swa_slots.copy_(
            torch.where(live, bases[group] * p + pos - first[group], -1)
        )
        history_pos = meta.starts[:, None] - 128 + torch.arange(128, device=self.device)
        floor = (
            torch.zeros(b, dtype=torch.int64, device=self.device)
            if floors is None
            else floors
        )
        valid = (history_pos >= floor[:, None]) & (meta.accepted[:, None] > 0)
        req = meta.requests.long()
        src = req[:, None] * self.capacity + history_pos % self.capacity
        dst = bases[:, None] * p + history_pos - first[:, None]
        plan.history_src.copy_(torch.where(valid, src, -1).flatten().int())
        # Invalid early history can fall before this request's workspace region.
        plan.history_dst.copy_(torch.where(valid, dst, -1).flatten().int())
        plan.ring_dst.copy_(
            torch.where(
                live, req[group] * self.capacity + pos % self.capacity, -1
            ).int()
        )
        plan.group.copy_(group)
        plan.positions.copy_(pos)
        # Never submit duplicate ring destinations from a long prefill.
        last = meta.lengths[group].long() - self.capacity
        plan.commit_dst.copy_(torch.where(pos >= last, plan.ring_dst, -1).int())
        if width and width > 1 and not hasattr(plan, "pending"):
            plan.pending = torch.empty(
                (self.ring.shape[0], n, 512),
                dtype=torch.bfloat16,
                device=self.device,
            )
        self.activate(plan)
        return plan

    def buffer(self, layer):
        if self.plan is None:
            raise RuntimeError("request SWA metadata was not prepared")
        if self.prepared != layer:
            history = self.movement.gather(self.ring[layer], self.plan.history_src)
            self.movement.scatter(
                history,
                self.plan.history_dst,
                out=self.workspace.kv_buffer[0].view(-1, 512),
            )
            self.prepared = layer
        return self.workspace.kv_buffer[0]

    def commit(self, layer, values):
        values = values.view(-1, 512).contiguous()
        dst = self.plan.commit_dst
        rows = getattr(self.plan, "commit_rows", None)
        if rows is not None and rows.numel() < values.shape[0]:
            values = self.movement.gather(values, rows)
            dst = self.movement.gather(dst, rows)
        self.movement.scatter(values, dst, out=self.ring[layer])

    def commit_verify(self, meta):
        plan = meta.ring_plan
        accepted = meta.verify_commit_lens[plan.group]
        keep = plan.positions < meta.starts[plan.group] + accepted
        dst = torch.where(keep & (accepted > 0), plan.ring_dst, -1)
        offsets = (
            torch.arange(self.ring.shape[0], device=self.device) * self.ring.shape[1]
        )
        dst = torch.where(dst[None, :] >= 0, dst[None, :] + offsets[:, None], -1)
        self.movement.scatter(
            plan.pending.view(-1, 512), dst.flatten().int(), out=self.ring.view(-1, 512)
        )


class BF16RequestWindowMemory:
    """910B backing selected by the common request-window memory factory."""

    def __init__(self, spec, *, dtype, device, enable_memory_saver=False):
        if dtype != torch.bfloat16 or torch.device(device).type != "npu":
            raise ValueError("910B request windows require BF16 on NPU")
        self.spec = spec
        self.view = BF16RequestWindow(
            None,
            num_slots=spec.num_slots,
            layers=spec.layers,
            page_size=spec.page_size,
            capacity=spec.capacity,
            workspace_rows=spec.workspace_rows,
        )

    @property
    def nbytes(self):
        return self.view.ring.nbytes + sum(
            t.nbytes for t in self.view.workspace.kv_buffer
        )

    def reset_rows(self, rows):
        for row in rows:
            self.view.valid_starts.pop(int(row), None)
        self.view.prepared = None

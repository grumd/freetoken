"""Elastic VRAM: the KV page pool and the MoE expert slot cache share one CUDA-VMM chunk pool.

Every page-scaled KV tier and every expert bank is an :class:`ElasticTensor` with a fixed
virtual range, so CUDA graphs and copy descriptors captured against them stay valid while
their backing moves. Startup maps the minimum (``min_pages`` KV pages plus each run's dummy
page, ``floor_slots`` expert slots), captures the graphs, then turns everything still free --
minus the prefill workspace and the keep-free margin -- into expert slots (:meth:`finalize`).

At runtime the scheduler grows KV when its free pages run out (:meth:`grow`, which retires the
coldest expert slots) and shrinks it back when idle (:meth:`shrink`). Both are deterministic
functions of the replicated scheduler state, so every TP rank moves the same pages at the
same step; only the slot counts may differ per rank (bank shards can differ in size).

Decode also borrows the prefill workspace: after ``lend_after`` decode steps with no prefill
waiting, :meth:`lend` turns the VRAM it left free into extra expert slots, and :meth:`reclaim`
hands it back before the next prefill (or at idle, between turns). A lend stalls decode ~80 ms
once, mostly torch releasing its cached activations, which blocks decode even from another
thread (kbench/bench_bg_lend.py); the extra slots are worth ~0.5 ms per token at TP=2 and
several ms at TP=1 on RTX 3080s.
"""

from __future__ import annotations

import math
import time
from typing import Callable, List, Tuple

import torch

from freetoken.kernel.vmm import ChunkPool, ElasticTensor, count_chunks
from freetoken.utils import init_logger

logger = init_logger(__name__)

# VRAM left beyond the keep-free margin while lent: decode's eager temporaries (sampling,
# page bookkeeping) still allocate through torch
_LEND_DECODE_MARGIN = 256 << 20


class ElasticVram:
    def __init__(
        self,
        device: torch.device,
        *,
        min_pages: int,
        step_pages: int,
        floor_slots: int,
        cap_slots: int,
        cap_pages: int,
        lend_after: int = 0,
    ) -> None:
        assert 0 < min_pages <= cap_pages and floor_slots <= cap_slots
        self.device = device
        self.pool = ChunkPool(device)
        self.min_pages = min_pages
        self.step_pages = step_pages
        self.floor_slots = floor_slots
        self.cap_slots = cap_slots
        self.cap_pages = cap_pages
        # the page count KV may grow to; settled across ranks in finalize()
        self.limit_pages = cap_pages
        self.pages = min_pages
        self.slots = floor_slots
        self.cache = None  # OffloadMoeCache, attached by the engine
        self.kv_pool = None  # the KV pool, attached by the engine (page moves)
        self._kv: List[Tuple[ElasticTensor, Callable[[int], list]]] = []
        self._banks: List[Tuple[ElasticTensor, int]] = []
        self.transitions = 0
        self.transition_ms = 0.0
        # decode-time lending of the prefill workspace (0 = never)
        self.lend_after = lend_after
        self.lent = 0  # chunks borrowed beyond the persistent pool
        self._quiet_steps = 0
        self._keep_free = 0

    # ------------------------------------------------------------- allocation hooks
    def kv_tier(self, role: str, shape, dtype: torch.dtype, intervals) -> torch.Tensor:
        """``mha_pool.elastic_tier_allocator`` hook: one page-scaled KV tier, zeroed as mapped."""
        need = count_chunks(intervals(self.pages), self.pool.chunk_bytes)
        self.pool.create(max(0, need - self.pool.num_free))
        t = ElasticTensor(self.pool, shape, dtype, intervals(self.pages), zero_on_map=True)
        self._kv.append((t, intervals))
        return t.tensor

    def bank(self, name: str, shape, dtype: torch.dtype) -> torch.Tensor:
        """``OffloadMoeCache.bank_alloc`` hook: one expert bank's slot cache."""
        assert shape[0] == self.cap_slots, (name, shape, self.cap_slots)
        row = math.prod(shape[1:]) * dtype.itemsize
        need = count_chunks([(0, self.slots * row)], self.pool.chunk_bytes)
        self.pool.create(max(0, need - self.pool.num_free))
        t = ElasticTensor(self.pool, shape, dtype, [(0, self.slots * row)])
        self._banks.append((t, row))
        return t.tensor

    # ------------------------------------------------------------- accounting
    def _kv_chunks(self, pages: int) -> int:
        return sum(t.count_required(fn(pages)) for t, fn in self._kv)

    def _bank_chunks(self, slots: int) -> int:
        return sum(t.count_required([(0, slots * row)]) for t, row in self._banks)

    def _max_slots(self, pages: int, handles: int | None = None) -> int:
        """Most expert slots that fit beside ``pages`` KV pages in ``handles`` chunks (default:
        the whole pool, lent chunks included)."""
        budget = (self.pool.num_handles if handles is None else handles) - self._kv_chunks(pages)
        lo, hi = self.floor_slots, self.cap_slots
        if self._bank_chunks(lo) > budget:
            raise RuntimeError(f"{pages} KV pages leave no room for {lo} expert slots")
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._bank_chunks(mid) <= budget:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _max_pages(self, slots: int) -> int:
        budget = self.pool.num_handles - self._bank_chunks(slots)
        lo, hi = self.min_pages, self.cap_pages
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._kv_chunks(mid) <= budget:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def growable_pages(self) -> int:
        return self.limit_pages - self.pages

    def bytes_per(self) -> tuple[int, int]:
        """(KV bytes per page, expert bytes per slot) as mapped, for reports."""
        kv = sum(t.nbytes for t, _ in self._kv) // max(1, self.cap_pages + 1)
        return kv, sum(row for _, row in self._banks)

    # ------------------------------------------------------------- transitions
    def finalize(self, free_bytes: int, reserve_bytes: int, keep_free_bytes: int, tp_group) -> None:
        """After every CUDA graph is captured: turn the remaining free VRAM, minus the prefill
        workspace and the keep-free margin, into chunks and give them all to the experts."""
        spare = free_bytes - reserve_bytes - keep_free_bytes
        if spare < 0:
            raise RuntimeError(
                f"elastic VRAM: {free_bytes / 2**30:.2f} GiB free after startup cannot hold the "
                f"{reserve_bytes / 2**30:.2f} GiB prefill reserve + {keep_free_bytes / 2**20:.0f} MiB "
                "keep-free margin (lower --prefill-reserve-gb / --vram-keep-free-mb or the floors)"
            )
        self.pool.create(spare // self.pool.chunk_bytes)
        limit = torch.tensor([self._max_pages(self.floor_slots)], dtype=torch.int64)
        if tp_group is not None:
            # TP ranks must agree on how far KV can grow; bank shards may differ in size
            torch.distributed.all_reduce(limit, op=torch.distributed.ReduceOp.MIN, group=tp_group)
        self.limit_pages = int(limit.item())
        self._tp_group = tp_group
        self._keep_free = keep_free_bytes
        self._resize(self.pages, self._max_slots(self.pages))
        kv_page, slot = self.bytes_per()
        logger.info_rank0(
            f"Elastic VRAM: {self.pool.num_handles * self.pool.chunk_bytes / 2**30:.2f} GiB pool "
            f"(KV {kv_page / 1024:.0f} KiB/page, experts {slot / 2**20:.2f} MiB/slot): "
            f"KV {self.min_pages}..{self.limit_pages} pages, experts "
            f"{self._max_slots(self.limit_pages)}..{self.slots} slots (floor {self.floor_slots}); "
            f"prefill reserve {reserve_bytes / 2**30:.2f} GiB, keep free {keep_free_bytes / 2**20:.0f} MiB"
        )

    def grow(self, need_pages: int) -> tuple[int, int]:
        """Back at least ``need_pages`` more KV pages (a whole number of steps, capped at the
        limit), retiring expert slots to pay for them. Returns the new page range [old, new)."""
        old = self.pages
        new = min(self.limit_pages, old + -(-max(need_pages, 1) // self.step_pages) * self.step_pages)
        if new > old:
            self._resize(new, self._max_slots(new))
        return old, self.pages

    def shrink(self, pages: int) -> None:
        """Release KV pages ``[pages, self.pages)`` (the caller guarantees they are free) back to
        the experts."""
        assert self.min_pages <= pages <= self.pages
        if pages < self.pages:
            self._resize(pages, self._max_slots(pages))

    def decode_step(self, prefill_pending: bool) -> None:
        """Once per decode batch: lend the prefill workspace after ``lend_after`` steps in a row
        with no prefill waiting."""
        self._quiet_steps += 1
        if (self.lend_after and not self.lent and not prefill_pending
                and self._quiet_steps >= self.lend_after):
            self.lend()

    def lend(self) -> int:
        """Turn the VRAM free beside the pool (the idle prefill workspace) into expert slots."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()  # the last prefill's activations, cached by torch
        free = torch.cuda.mem_get_info(self.device)[0]
        n = (free - self._keep_free - _LEND_DECODE_MARGIN) // self.pool.chunk_bytes
        # no further than the bank reservations reach
        n = min(n, self._kv_chunks(self.pages) + self._bank_chunks(self.cap_slots) - self.pool.num_handles)
        if n <= 0:
            return 0
        self.pool.create(n)
        self.lent = n
        self._resize(self.pages, self._max_slots(self.pages), why="lend")
        return n

    def reclaim(self) -> None:
        """Give lent chunks back to the driver, coldest expert slots first. Before any prefill,
        any KV growth and any idle-time page work: those allocate through torch."""
        self._quiet_steps = 0
        if not self.lent:
            return
        persistent = self.pool.num_handles - self.lent
        self._resize(self.pages, self._max_slots(self.pages, persistent), why="reclaim")
        self.pool.release(self.lent)
        self.lent = 0

    def move_pages(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Copy whole KV pages ``src`` -> ``dst`` (disjoint sets) across every page-scaled tier."""
        src, dst = src.to(self.device, torch.long), dst.to(self.device, torch.long)
        torch.cuda.synchronize(self.device)
        for view in self.kv_pool.page_views():
            for i in range(0, len(src), 256):  # bound the gather temporary
                view[dst[i : i + 256]] = view[src[i : i + 256]]
        torch.cuda.synchronize(self.device)

    def _resize(self, pages: int, slots: int, why: str = "") -> None:
        t0 = time.perf_counter()
        old_pages, old_slots = self.pages, self.slots
        torch.cuda.synchronize(self.device)
        moved = 0
        # release first so the freed handles can back the other side
        if slots < old_slots:
            moved = self.cache.shrink_active(slots) if self.cache is not None else 0
            torch.cuda.synchronize(self.device)  # the compaction copies read rows about to go
            for t, row in self._banks:
                t.set_required([(0, slots * row)])
        if pages != old_pages:
            for t, fn in self._kv:
                t.set_required(fn(pages))
        if slots > old_slots:
            for t, row in self._banks:
                t.set_required([(0, slots * row)])
            if self.cache is not None:
                self.cache.grow_active(slots)
        torch.cuda.synchronize(self.device)
        self.pages, self.slots = pages, slots
        group = getattr(self, "_tp_group", None)
        if group is not None and old_pages != pages:
            seen = torch.tensor([pages, -pages], dtype=torch.int64)
            torch.distributed.all_reduce(seen, op=torch.distributed.ReduceOp.MIN, group=group)
            if int(seen[0]) != pages or -int(seen[1]) != pages:
                raise RuntimeError(f"elastic VRAM: TP ranks disagree on the KV page count ({pages})")
        ms = (time.perf_counter() - t0) * 1e3
        self.transitions += 1
        self.transition_ms += ms
        if old_pages != pages or old_slots != slots:
            logger.info_rank0(
                f"elastic{f' {why}' if why else ''}: kv {old_pages}->{pages} pages, experts "
                f"{old_slots}->{slots} slots{f' ({moved} kept)' if moved else ''}, {ms:.1f} ms"
            )

    def stats(self) -> dict:
        return {
            "kv_pages": self.pages,
            "kv_pages_limit": self.limit_pages,
            "expert_slots": self.slots,
            "expert_slots_floor": self.floor_slots,
            "lent_chunks": self.lent,
            "transitions": self.transitions,
            "transition_ms": round(self.transition_ms, 1),
        }


__all__ = ["ElasticVram"]

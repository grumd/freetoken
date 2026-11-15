"""Elastic VRAM: the KV page pool and the MoE expert slot cache share one CUDA-VMM chunk pool.

Every page-scaled KV tier and every expert bank is an :class:`ElasticTensor` with a fixed
virtual range, so CUDA graphs and copy descriptors captured against them stay valid while
their backing moves. Startup maps the minimum (``min_pages`` KV pages plus each run's dummy
page, ``floor_slots`` expert slots), captures the graphs, then turns everything still free --
minus the standing prefill reserve and the keep-free margin -- into expert slots (:meth:`finalize`).

At runtime the scheduler grows KV when its free pages run out (:meth:`grow`, which retires the
coldest expert slots) and shrinks it back when idle (:meth:`shrink`). Both are deterministic
functions of the replicated scheduler state, so every TP rank moves the same pages at the
same step; only the slot counts may differ per rank (bank shards can differ in size).

Prefill activations come from torch's allocator, outside the pool. The standing reserve covers
a ``prefill_min_tokens`` chunk at any context length. A longer chunk, up to ``prefill_max_tokens``,
first retires the coldest expert slots for exactly the VRAM its activations need beyond the
reserve (:meth:`prefill_tokens` sizes it, :meth:`hold_prefill` takes it); once no prefill is
waiting, :meth:`settle` gives it back as empty slots and lends the idle reserve to the experts too
(``lend_after``), until the next prefill. A give-back stalls the host 10-40 ms after an agent turn
and ~0.13 s after a 16k-token chunk (torch releasing its cached activations, one driver call per
chunk); every slot a prefill keeps costs decode a CPU-computed miss until it is refilled, so the
workspace takes no more than the chunk needs.
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
        prefill_min_tokens: int = 0,
        prefill_max_tokens: int = 0,
        prefill_bytes_per_token: int = 0,
    ) -> None:
        assert 0 < min_pages <= cap_pages and floor_slots <= cap_slots
        assert 0 <= prefill_min_tokens <= prefill_max_tokens
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
        # the pool's own chunks (finalize), and the chunks KV plus the prefill workspace may use
        # beside the expert floor: the smallest over the TP ranks, so sizing is rank-independent
        self.base = 0
        self.room = 0
        # prefill workspace: chunks a long chunk took from the pool beyond the standing reserve
        self.prefill_min_tokens = prefill_min_tokens
        self.prefill_max_tokens = prefill_max_tokens
        self.prefill_bytes_per_token = prefill_bytes_per_token
        self.taken = 0
        # decode-time lending of the idle standing reserve (0 = never): chunks beyond the pool
        self.lend_after = lend_after
        self.lent = 0
        self._quiet_steps = 0  # decode steps since the last prefill
        self._keep_free = 0
        self._tp_group = None

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

    def _pages_within(self, chunks: int) -> int:
        """Most KV pages (at least ``min_pages``) whose chunks fit in ``chunks``."""
        lo, hi = self.min_pages, self.cap_pages
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._kv_chunks(mid) <= chunks:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _page_limit(self) -> int:
        """KV may grow only as far as the prefill workspace held now leaves room for."""
        return min(self.limit_pages, self._pages_within(self.room - self.taken))

    def growable_pages(self) -> int:
        return max(0, self._page_limit() - self.pages)

    def bytes_per(self) -> tuple[int, int]:
        """(KV bytes per page, expert bytes per slot) as mapped, for reports."""
        kv = sum(t.nbytes for t, _ in self._kv) // max(1, self.cap_pages + 1)
        return kv, sum(row for _, row in self._banks)

    # ------------------------------------------------------------- transitions
    def finalize(self, free_bytes: int, reserve_bytes: int, keep_free_bytes: int, tp_group) -> None:
        """After every CUDA graph is captured: turn the remaining free VRAM, minus the standing
        prefill reserve and the keep-free margin, into chunks and give them all to the experts."""
        spare = free_bytes - reserve_bytes - keep_free_bytes
        if spare < 0:
            raise RuntimeError(
                f"elastic VRAM: {free_bytes / 2**30:.2f} GiB free after startup cannot hold the "
                f"{reserve_bytes / 2**30:.2f} GiB prefill reserve + {keep_free_bytes / 2**20:.0f} MiB "
                "keep-free margin (lower --prefill-batch-min / --vram-keep-free-mb or the floors)"
            )
        self.pool.create(spare // self.pool.chunk_bytes)
        self.base = self.pool.num_handles
        kv, cb = self._kv_chunks(self.cap_pages), self.pool.chunk_bytes
        settled = torch.tensor([self.base - self._bank_chunks(self.floor_slots), kv, -kv, cb, -cb], dtype=torch.int64)
        if tp_group is not None:
            # bank shards may differ in size, so the room is the smallest rank's; the chunk sizes and
            # page limits computed from it are the same on every rank only if their KV is
            torch.distributed.all_reduce(settled, op=torch.distributed.ReduceOp.MIN, group=tp_group)
            if int(settled[1]) != -int(settled[2]) or int(settled[3]) != -int(settled[4]):
                raise RuntimeError("elastic VRAM: TP ranks differ in KV chunk count or chunk size")
        self.room = int(settled[0])
        if self._kv_chunks(self.min_pages) > self.room:
            raise RuntimeError(f"{self.min_pages} KV pages leave no room for {self.floor_slots} expert slots")
        self.limit_pages = self._pages_within(self.room)
        self._tp_group = tp_group
        self._keep_free = keep_free_bytes
        self._resize(self.pages, self._max_slots(self.pages))
        kv_page, slot = self.bytes_per()
        logger.info_rank0(
            f"Elastic VRAM: {self.pool.num_handles * cb / 2**30:.2f} GiB pool "
            f"(KV {kv_page / 1024:.0f} KiB/page, experts {slot / 2**20:.2f} MiB/slot): "
            f"KV {self.min_pages}..{self.limit_pages} pages, experts "
            f"{self._max_slots(self.limit_pages)}..{self.slots} slots (floor {self.floor_slots}); "
            f"prefill reserve {reserve_bytes / 2**30:.2f} GiB ({self.prefill_min_tokens}-token chunks, up to "
            f"{self.prefill_max_tokens} from the experts), keep free {keep_free_bytes / 2**20:.0f} MiB"
        )

    def grow(self, need_pages: int) -> tuple[int, int]:
        """Back at least ``need_pages`` more KV pages (a whole number of steps, capped at the
        limit), retiring expert slots to pay for them. Returns the new page range [old, new)."""
        old = self.pages
        new = min(self._page_limit(), old + -(-max(need_pages, 1) // self.step_pages) * self.step_pages)
        if new > old:
            self._resize(new, self._max_slots(new))
        return old, self.pages

    def shrink(self, pages: int) -> None:
        """Release KV pages ``[pages, self.pages)`` (the caller guarantees they are free) back to
        the experts."""
        assert self.min_pages <= pages <= self.pages
        if pages < self.pages:
            self._resize(pages, self._max_slots(pages))

    # ------------------------------------------------------------- prefill workspace
    def _workspace_chunks(self, tokens: int) -> int:
        """Chunks a ``tokens``-token prefill chunk needs beyond the standing reserve."""
        extra = self.prefill_bytes_per_token * max(0, tokens - self.prefill_min_tokens)
        return -(-extra // self.pool.chunk_bytes)

    def prefill_tokens(self, remaining: int) -> int:
        """Size of the next prefill chunk of a prompt with ``remaining`` tokens left: all of them if
        they fit, else the most the expert cache can make room for beside the KV pages backed now,
        between ``prefill_min_tokens`` and ``prefill_max_tokens``. A function of the page count and
        the settled room only, so every TP rank picks the same size."""
        lo, hi = self.prefill_min_tokens, self.prefill_max_tokens
        if remaining <= lo:
            return remaining
        if lo == hi:
            return hi
        spare = max(0, self.room - self._kv_chunks(self.pages)) * self.pool.chunk_bytes
        return max(lo, min(remaining, hi, lo + spare // self.prefill_bytes_per_token))

    def hold_prefill(self, tokens: int) -> None:
        """Before a prefill chunk of ``tokens`` (from :meth:`prefill_tokens`), or with 0 before
        anything else that allocates through torch: end a lend, and retire the coldest expert slots
        until the VRAM beside the pool covers the chunk's activations. Within a prompt the
        workspace only grows; :meth:`settle` gives it back."""
        self._quiet_steps = 0
        want = max(self.taken, self._workspace_chunks(tokens))
        if want == self.taken and not self.lent:
            return
        torch.cuda.synchronize(self.device)  # the batch in flight; GPU time, not the transition's
        t0, old_slots = time.perf_counter(), self.slots
        handles = self.base - want
        self._resize(self.pages, self._max_slots(self.pages, handles), log=False)
        self.pool.release(self.pool.num_handles - handles)
        lent, self.lent, self.taken = self.lent, 0, want
        logger.info_rank0(
            f"prefill workspace: take {tokens} tokens, experts {old_slots}->{self.slots} slots "
            f"({want} chunks beyond the reserve{f', lend of {lent} ended' if lent else ''}), "
            f"{(time.perf_counter() - t0) * 1e3:.1f} ms"
        )

    def settle(self, *, prefill_pending: bool, decoding: bool) -> None:
        """Once per scheduler iteration, before anything runs: with no prefill waiting, give the
        workspace back to the experts, and while a decode runs lend them the idle standing reserve
        too, from its ``lend_after``-th step on (1 = with the first)."""
        if prefill_pending:
            return
        lend = decoding and self.lend_after > 0 and self._quiet_steps + 1 >= self.lend_after
        if self.taken or (lend and not self.lent):
            self.release_prefill(lend=lend)

    def decode_step(self) -> None:
        """Once per decode batch, for :meth:`settle`'s ``lend_after`` count."""
        self._quiet_steps += 1

    def release_prefill(self, lend: bool = False) -> None:
        """Give the workspace back to the experts as empty slots; with ``lend`` also turn the VRAM
        free beside the pool (the idle standing reserve) into expert slots, else end a lend. The
        idle point and :meth:`settle` call it."""
        end_lend, start_lend = bool(self.lent) and not lend, lend and not self.lent
        if not (end_lend or start_lend or self.taken):
            return
        torch.cuda.synchronize(self.device)  # the batch in flight; GPU time, not the transition's
        t0, old_slots, given = time.perf_counter(), self.slots, self.taken
        if end_lend:
            self._resize(self.pages, self._max_slots(self.pages, self.base - self.taken), log=False)
            self.pool.release(self.lent)
            given, self.lent = -self.lent, 0
        else:
            torch.cuda.empty_cache()  # the last chunk's activations, cached by torch
            if self.taken:
                self._take_back()
            if start_lend and not self.taken:
                free = torch.cuda.mem_get_info(self.device)[0]
                n = (free - self._keep_free - _LEND_DECODE_MARGIN) // self.pool.chunk_bytes
                # no further than the bank reservations reach
                n = min(n, self._kv_chunks(self.pages) + self._bank_chunks(self.cap_slots) - self.pool.num_handles)
                if n > 0:
                    self.pool.create(n)
                    self.lent = n
            self._resize(self.pages, self._max_slots(self.pages), log=False)
        if self.slots != old_slots:
            what = [f"return {given} chunks"] if given > 0 else [f"end lend of {-given} chunks"] if given else []
            what += [f"lend {self.lent} chunks"] if self.lent else []
            logger.info_rank0(
                f"prefill workspace: {', '.join(what)}, experts {old_slots}->{self.slots} slots, "
                f"{(time.perf_counter() - t0) * 1e3:.1f} ms"
            )

    def _take_back(self) -> None:
        """Map the workspace's chunks back from the driver. On one GPU a shortfall is retried at
        the next give-back (KV stays capped meanwhile); TP ranks must agree, so there it stops
        the server rather than let their page limits diverge."""
        before, err = self.pool.num_handles, None
        try:
            self.pool.create(self.taken)
        except RuntimeError as e:
            err = e
        made = self.pool.num_handles - before
        if self._tp_group is not None:
            ok = torch.tensor([int(err is None)], dtype=torch.int64)
            torch.distributed.all_reduce(ok, op=torch.distributed.ReduceOp.MIN, group=self._tp_group)
            if not int(ok.item()):
                raise RuntimeError(
                    f"prefill workspace: a TP rank could not take its {self.taken} expert chunks back "
                    f"(this rank: {made}, {err})"
                )
        elif err is not None:
            logger.warning(f"prefill workspace: {made}/{self.taken} chunks back ({err})")
        self.taken -= made

    def move_pages(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Copy whole KV pages ``src`` -> ``dst`` (disjoint sets) across every page-scaled tier."""
        src, dst = src.to(self.device, torch.long), dst.to(self.device, torch.long)
        torch.cuda.synchronize(self.device)
        for view in self.kv_pool.page_views():
            for i in range(0, len(src), 256):  # bound the gather temporary
                view[dst[i : i + 256]] = view[src[i : i + 256]]
        torch.cuda.synchronize(self.device)

    def _resize(self, pages: int, slots: int, log: bool = True) -> None:
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
        if log and (old_pages != pages or old_slots != slots):
            logger.info_rank0(
                f"elastic: kv {old_pages}->{pages} pages, experts "
                f"{old_slots}->{slots} slots{f' ({moved} kept)' if moved else ''}, {ms:.1f} ms"
            )

    def stats(self) -> dict:
        return {
            "kv_pages": self.pages,
            "kv_pages_limit": self.limit_pages,
            "expert_slots": self.slots,
            "expert_slots_floor": self.floor_slots,
            "lent_chunks": self.lent,
            "prefill_chunks": self.taken,
            "transitions": self.transitions,
            "transition_ms": round(self.transition_ms, 1),
        }


__all__ = ["ElasticVram"]

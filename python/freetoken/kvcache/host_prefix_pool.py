"""Pinned host-RAM tier for the hybrid radix prefix cache (--cache-ram-gb).

One host page holds one device KV page of every page-scaled tier (the paged K/V of every
attention layer and, for QSA, the page's compressed index-key rows); one host snapshot slot
holds one LinearStatePool slot (GDN conv + recurrent state of every linear layer). The tree
(HybridRadixCache) decides what lives here; this class only owns the memory and moves bytes.

Everything is preallocated at startup, so an over-committed --cache-ram-gb fails when the
server starts instead of as a cgroup OOM in the middle of a session.
"""

from __future__ import annotations

from typing import List, Sequence

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

# Rows per staging copy: bounds the device-side gather buffer (one row = one page of one tier).
_STAGE_ROWS = 256


def _runs(host_idx: List[int]) -> List[tuple[int, int, int]]:
    """(position, first host page, length) of each run of consecutive host pages."""
    runs: List[tuple[int, int, int]] = []
    for pos, h in enumerate(host_idx):
        if runs and runs[-1][1] + runs[-1][2] == h and runs[-1][0] + runs[-1][2] == pos:
            runs[-1] = (runs[-1][0], runs[-1][1], runs[-1][2] + 1)
        else:
            runs.append((pos, h, 1))
    return runs


class HostPrefixPool:
    def __init__(
        self,
        page_views: Sequence[torch.Tensor],
        state_tensors: Sequence[torch.Tensor],
        budget_bytes: int,
        page_size: int,
    ) -> None:
        """``page_views``: device tensors indexed by page on dim 0 (``KVCache.page_views``).
        ``state_tensors``: LinearStatePool tensors indexed ``[layer, slot, ...]``."""
        from freetoken.kernel.pinned import alloc_pinned_tensor

        self.page_views = list(page_views)
        self.state_tensors = list(state_tensors)
        self.page_size = page_size
        self.device = self.page_views[0].device
        self.page_bytes = sum(v[0].numel() * v.element_size() for v in self.page_views)
        self.snap_bytes = sum(t[:, 0].numel() * t.element_size() for t in self.state_tensors)
        # one snapshot per ~16k tokens of page capacity: a parked conversation needs exactly one
        # (its leaf's), and short ones dominate
        per_snap = self.snap_bytes + (16384 // page_size) * self.page_bytes
        self.num_snaps = max(2, budget_bytes // per_snap) if self.snap_bytes else 0
        self.num_pages = (budget_bytes - self.num_snaps * self.snap_bytes) // self.page_bytes
        if self.num_pages <= 0:
            raise ValueError(
                f"--cache-ram-gb leaves {budget_bytes / 2**20:.0f} MiB per rank, below one GDN "
                f"snapshot ({self.snap_bytes / 2**20:.0f} MiB) plus a page"
            )
        self.host_pages = [
            alloc_pinned_tensor(self.num_pages, *v.shape[1:], dtype=v.dtype) for v in self.page_views
        ]
        self.host_snaps = [
            alloc_pinned_tensor(t.shape[0], self.num_snaps, *t.shape[2:], dtype=t.dtype)
            for t in self.state_tensors
        ]
        # sorted lowest-first so consecutive allocations make long copy runs
        self._free_pages: List[int] = list(range(self.num_pages))
        self._free_snaps: List[int] = list(range(self.num_snaps))
        logger.info_rank0(
            f"Prefix RAM tier: {self.num_pages} pages ({self.num_pages * page_size} tokens, "
            f"{self.num_pages * self.page_bytes / 2**30:.2f} GiB) + {self.num_snaps} GDN snapshots "
            f"({self.num_snaps * self.snap_bytes / 2**30:.2f} GiB) per rank"
        )

    # ------------------------------------------------------------- allocation
    @property
    def free_pages(self) -> int:
        return len(self._free_pages)

    @property
    def free_snaps(self) -> int:
        return len(self._free_snaps)

    def alloc_pages(self, n: int) -> List[int]:
        assert n <= len(self._free_pages), (n, len(self._free_pages))
        out, self._free_pages = self._free_pages[:n], self._free_pages[n:]
        return out

    def release_pages(self, pages: List[int]) -> None:
        self._free_pages = sorted(self._free_pages + list(pages))

    def alloc_snap(self) -> int:
        return self._free_snaps.pop(0)

    def release_snap(self, slot: int) -> None:
        self._free_snaps.append(slot)
        self._free_snaps.sort()

    # ------------------------------------------------------------- copies
    def save_pages(self, device_pages: torch.Tensor, host_pages: List[int]) -> None:
        """Device pages -> host pages (blocking: the caller frees the device pages next)."""
        dev = device_pages.to(self.device, torch.long)
        for pos, h, n in _runs(host_pages):
            for i in range(0, n, _STAGE_ROWS):
                k = min(_STAGE_ROWS, n - i)
                idx = dev[pos + i : pos + i + k]
                for view, host in zip(self.page_views, self.host_pages):
                    host[h + i : h + i + k].copy_(view[idx], non_blocking=True)
        torch.cuda.synchronize(self.device)

    def load_pages(self, host_pages: List[int], device_pages: torch.Tensor) -> None:
        """Host pages -> device pages (blocking: the next forward reads them)."""
        dev = device_pages.to(self.device, torch.long)
        for pos, h, n in _runs(host_pages):
            for i in range(0, n, _STAGE_ROWS):
                k = min(_STAGE_ROWS, n - i)
                idx = dev[pos + i : pos + i + k]
                for view, host in zip(self.page_views, self.host_pages):
                    view[idx] = host[h + i : h + i + k].to(self.device, non_blocking=True)
        torch.cuda.synchronize(self.device)

    def save_snap(self, slot: int, host_slot: int) -> None:
        for t, host in zip(self.state_tensors, self.host_snaps):
            host[:, host_slot].copy_(t[:, slot], non_blocking=True)
        torch.cuda.synchronize(self.device)

    def load_snap(self, host_slot: int, slot: int) -> None:
        for t, host in zip(self.state_tensors, self.host_snaps):
            t[:, slot].copy_(host[:, host_slot], non_blocking=True)
        torch.cuda.synchronize(self.device)


__all__ = ["HostPrefixPool"]

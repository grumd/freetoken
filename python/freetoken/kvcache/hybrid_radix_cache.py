"""Hybrid (full-attn KV + GDN linear-state) radix cache.

A SEPARATE class from ``RadixPrefixCache`` (Option C) that REUSES the shared ``RadixTreeNode``
and walk/split logic, so the production KV radix is untouched (zero risk to non-hybrid models).
It adds a second "currency": an optional GDN state snapshot (``node.mamba_value`` = a
LinearStatePool slot id) attached at chunk/page-aligned boundary nodes, with its own LRU
eviction. Mirrors sglang ``MambaRadixCache`` (donate-not-copy, dual eviction, internal-node
tombstone, ``full_ref >= mamba_ref``) on FreeToken's tree.

Currency seam: the secondary value + its eviction is the slot a future SWA component plugs
into. This class is pool-agnostic -- it stores/returns slot ids and KV page indices; the
caller (CacheManager / scheduler) does the actual LinearStatePool / KV-pool free.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import torch

from freetoken.utils import align_down

from .base import BaseCacheHandle
from .radix_cache import RadixTreeNode, _get_key_fn


@dataclass(frozen=True)
class HybridCacheHandle(BaseCacheHandle):
    """Lock handle for a matched hybrid prefix: the matched node (lock target) + the reusable
    KV page indices. ``cached_len`` is already truncated to the deepest live-snapshot boundary.
    Plugs into PrefillAdder (reads ``.cached_len`` / ``.get_matched_indices()``) like the plain
    RadixCacheHandle; the restore slot rides on ``MatchResult.mamba_value``."""

    node: RadixTreeNode
    kv_indices: torch.Tensor

    def get_matched_indices(self) -> torch.Tensor:
        return self.kv_indices


class HybridMatch(NamedTuple):
    kv_indices: torch.Tensor      # reused KV page indices for [0:cached_len)
    cached_len: int               # truncated to the deepest LIVE-snapshot boundary
    mamba_value: Optional[int]    # GDN snapshot slot to restore from (None = cold start)
    node: RadixTreeNode           # the matched node (lock target)


class EvictResult(NamedTuple):
    kv_indices: torch.Tensor      # KV page indices to free
    mamba_slots: List[int]        # GDN state slots to free


class HybridRadixCache:
    """With a ``host`` (HostPrefixPool, --cache-ram-gb) evicted prefixes spill to pinned RAM
    instead of being dropped: a node may then live on the device (``value`` set), in RAM only
    (``value`` None, ``host_pages`` set) or both. Device nodes are closed under ancestors, so
    device eviction works on the device frontier (device nodes with no device children) and a
    match whose snapshot lives in RAM reloads a contiguous host-only tail of its path."""

    def __init__(self, device: torch.device, page_size: int, host=None) -> None:
        from freetoken.kernel.fla.chunk import CHUNK_SIZE
        # Snapshots land on ×CHUNK_SIZE boundaries; require them to be page-aligned so the KV
        # node boundary and the GDN-state boundary coincide (page_size in {1,2,4,8,16,32,64}).
        assert CHUNK_SIZE % page_size == 0, (
            f"hybrid_radix needs CHUNK_SIZE({CHUNK_SIZE}) % page_size({page_size}) == 0"
        )
        self.device = device
        self.page_size = page_size
        self.key_fn = _get_key_fn(page_size)
        self.empty = torch.empty(0, dtype=torch.int32, device=device)
        self.root = RadixTreeNode(self.key_fn)
        self.root.set_key_value(self.empty, self.empty)
        self.root.ref_count = 1  # root is always protected
        self.full_evictable = 0
        self.full_protected = 0
        self.mamba_evictable = 0     # number of live, unlocked snapshots
        self.mamba_protected = 0
        self.req_seq = 0             # admissions so far; stamps node.last_seq (see note_admission)
        self.host = host
        self.spilled_tokens = 0  # KV tokens moved to the RAM tier so far (logs)

    # ---------------------------------------------------------------- match / insert
    def match_prefix(self, input_ids: torch.Tensor) -> HybridMatch:
        """Match the token prefix, then truncate the reusable length to the deepest node on
        the path that still owns a LIVE snapshot (a continuation can only resume the GDN
        recurrence from a checkpointed boundary)."""
        node, _ = self._walk(input_ids)
        # walk up to the deepest node whose END boundary has a live snapshot
        cur, end_len = node, self._path_len(node)
        while not cur.is_root():
            if cur.mamba_value is not None:
                return HybridMatch(self._collect_kv(cur), end_len, cur.mamba_value, cur)
            end_len -= cur.length
            cur = cur.parent
        return HybridMatch(self.empty, 0, None, self.root)

    def insert(self, input_ids: torch.Tensor, kv_indices: torch.Tensor,
               mamba_value: int) -> Tuple[int, bool]:
        """Insert the committed KV prefix and DONATE ``mamba_value`` at the (page-aligned) end
        boundary node. Returns (matched_prefix_len, mamba_exist). If the boundary node already
        owns a live snapshot, returns mamba_exist=True and does not attach (caller frees the
        donated slot -- dedup)."""
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, kv_indices = input_ids[:insert_len], kv_indices[:insert_len]
        node, prefix_len = self._walk(input_ids)
        adopted = self._adopt_host_path(node, kv_indices) if self.host is not None else None
        if prefix_len != insert_len:
            new_node = RadixTreeNode(self.key_fn)
            new_node.last_seq = self.req_seq
            new_node.set_key_value(input_ids[prefix_len:], kv_indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.full_evictable += new_node.length
            node = new_node
        if adopted is not None:
            # the caller's pages now back the adopted nodes: only [.., adopted) were duplicates
            prefix_len = min(prefix_len, adopted)
        if node.is_root():
            return prefix_len, True   # root can't hold a snapshot; report exist so caller frees it
        if node.mamba_value is not None:
            return prefix_len, True                 # dedup: caller frees its donated slot
        node.mamba_value = mamba_value              # fills a fresh node or a tombstone
        if node.mamba_ref_count == 0:
            self.mamba_evictable += 1
        return prefix_len, False

    def _adopt_host_path(self, node: RadixTreeNode, kv_indices: torch.Tensor) -> Optional[int]:
        """Back the RAM-only nodes on root..node with the caller's freshly computed pages (the
        same tokens); returns where that span starts, or None if the path was all on device."""
        path, cur = [], node
        while not cur.is_root():
            path.append(cur)
            cur = cur.parent
        start, off = None, 0
        for n in reversed(path):
            if n.value is None:
                assert n.ref_count == 0, "a RAM-only node cannot be locked"
                start = off if start is None else start
                n.set_key_value(n._key, kv_indices[off : off + n.length].clone())
                self.full_evictable += n.length
            off += n.length
        return start

    # ---------------------------------------------------------------- RAM tier
    def match_tiered(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, bool]:
        """The deepest node on the matched path with a snapshot on either tier, and whether its
        KV or snapshot must first come back from RAM (see host_path / load_*)."""
        node, _ = self._walk(input_ids)
        cur = node
        while not cur.is_root():
            if cur.mamba_value is not None:
                return cur, False
            if cur.host_mamba is not None:
                return cur, True
            cur = cur.parent
        return self.root, False

    def host_path(self, node: RadixTreeNode) -> List[RadixTreeNode]:
        """The RAM-only nodes on root..node, top-down (a contiguous tail of the path)."""
        out, cur = [], node
        while not cur.is_root() and cur.value is None:
            out.append(cur)
            cur = cur.parent
        return out[::-1]

    def load_pages(self, nodes: List[RadixTreeNode], page_starts: torch.Tensor) -> None:
        """Copy ``nodes``' KV from RAM into the device pages starting at ``page_starts`` (in node
        order) and make them device nodes again; their RAM copy stays as a free re-spill."""
        ps, off = self.page_size, 0
        offsets = torch.arange(ps, dtype=torch.int32, device=self.device)
        for n in nodes:
            k = n.length // ps
            pages = page_starts[off : off + k]
            self.host.load_pages(n.host_pages, pages // ps)
            n.set_key_value(n._key, (pages.unsqueeze(1) + offsets).flatten())
            self.full_evictable += n.length
            off += k

    def load_snapshot(self, node: RadixTreeNode, slot: int) -> None:
        self.host.load_snap(node.host_mamba, slot)
        node.mamba_value = slot
        if node.mamba_ref_count == 0:
            self.mamba_evictable += 1

    def _spill(self, node: RadixTreeNode, kv: list, mamba: list) -> bool:
        """Move an unlocked device frontier node's KV (and, for a leaf, its snapshot) to RAM;
        False if RAM cannot take it (the caller drops it instead)."""
        host, ps = self.host, self.page_size
        needs_snap = node.is_leaf() and node.host_mamba is None
        if needs_snap and node.mamba_value is None:
            return False  # a leaf without a snapshot can never be resumed
        pages = node.length // ps if node.host_pages is None else 0
        if not self._host_make_room(pages, int(needs_snap), keep=node):
            return False
        if pages:
            node.host_pages = host.alloc_pages(pages)
            host.save_pages(node.value[::ps] // ps, node.host_pages)
        if needs_snap:
            node.host_mamba = host.alloc_snap()
            host.save_snap(node.mamba_value, node.host_mamba)
        # an internal node's snapshot is not kept: its children carry their own
        self._free_node_mamba(node, mamba)
        kv.append(node.value)
        self.full_evictable -= node.length
        self.spilled_tokens += node.length
        node.set_key_value(node._key, None)
        return True

    def _host_make_room(self, pages: int, snaps: int, keep: RadixTreeNode | None = None) -> bool:
        """Free RAM-tier pages/snapshots: RAM copies of device nodes go first (they are only
        backups), then whole RAM-only leaves, least recently used first."""
        host = self.host
        if host.free_pages >= pages and host.free_snaps >= snaps:
            return True
        if pages > host.num_pages or snaps > host.num_snaps:
            return False
        nodes = [n for n in self._all_nodes() if n is not keep and not n.host_pinned
                 and (n.host_pages is not None or n.host_mamba is not None)]
        backups = sorted((n for n in nodes if n.value is not None), key=lambda n: n.timestamp)
        for n in backups:
            if host.free_pages >= pages and host.free_snaps >= snaps:
                return True
            self._release_host(n)
        gone: set = set()
        while host.free_pages < pages or host.free_snaps < snaps:
            leaves = [n for n in self._all_nodes() if n.value is None and n.is_leaf()
                      and n is not keep and not n.host_pinned]
            if not leaves:
                return False
            victim = min(leaves, key=lambda n: n.timestamp)
            parent = self._unlink(victim)
            self._release_host(victim)
            while (parent.value is None and parent.is_leaf() and parent.host_mamba is None
                   and not parent.is_root() and not parent.host_pinned):
                self._release_host(parent)
                parent = self._unlink(parent)
        return True

    def _release_host(self, node: RadixTreeNode) -> None:
        if node.host_pages is not None:
            self.host.release_pages(node.host_pages)
            node.host_pages = None
        if node.host_mamba is not None:
            self.host.release_snap(node.host_mamba)
            node.host_mamba = None

    def _evict_node(self, node: RadixTreeNode, kv: list, mamba: list) -> RadixTreeNode:
        """Take an unlocked device frontier node off the GPU -- spilled to RAM when there is a
        RAM tier with room, else dropped with everything below it. Returns the ancestor whose
        eviction should be considered next."""
        if self.host is not None and self._spill(node, kv, mamba):
            return node.parent
        stack = [node]
        while stack:
            n = stack.pop()
            stack.extend(n.children.values())
            if n.value is not None:
                kv.append(n.value)
                self.full_evictable -= n.length
            self._free_node_mamba(n, mamba)
            if self.host is not None:
                self._release_host(n)
        parent, _ = self._cascade_tombstone_leaves(self._unlink(node), kv)
        return parent

    def _has_device_child(self, node: RadixTreeNode) -> bool:
        return any(c.value is not None for c in node.children.values())

    def _evictable_frontier(self, node: RadixTreeNode) -> bool:
        return (not node.is_root() and node.ref_count == 0 and node.value is not None
                and not self._has_device_child(node))

    # ---------------------------------------------------------------- locking (dual)
    def inc_lock(self, node: RadixTreeNode) -> None:
        """Protect a matched node's snapshot (mamba ref on the node) and its KV path
        (full ref node..root). Enforces full_ref >= mamba_ref: using a snapshot at N pins the
        whole root..N KV chain."""
        if node.mamba_value is not None:
            if node.mamba_ref_count == 0:
                self.mamba_evictable -= 1
                self.mamba_protected += 1
            node.mamba_ref_count += 1
        cur = node
        while not cur.is_root():
            if cur.ref_count == 0:
                self.full_evictable -= cur.length
                self.full_protected += cur.length
            cur.ref_count += 1
            cur = cur.parent

    def dec_lock(self, node: RadixTreeNode) -> None:
        if node.mamba_value is not None and node.mamba_ref_count > 0:
            node.mamba_ref_count -= 1
            if node.mamba_ref_count == 0:
                self.mamba_evictable += 1
                self.mamba_protected -= 1
        cur = node
        while not cur.is_root():
            cur.ref_count -= 1
            assert cur.ref_count >= 0
            if cur.ref_count == 0:
                self.full_evictable += cur.length
                self.full_protected -= cur.length
            cur = cur.parent

    def note_admission(self, node: RadixTreeNode) -> None:
        """Count one admitted request and stamp its matched path (node..root) with the new
        sequence number."""
        self.req_seq += 1
        cur = node
        while not cur.is_root():
            cur.last_seq = self.req_seq
            cur = cur.parent

    def device_pages(self) -> torch.Tensor:
        """Every KV page the tree holds (node values are whole, page-aligned pages)."""
        vals, stack = [], list(self.root.children.values())
        while stack:
            n = stack.pop()
            if n.value is not None:
                vals.append(n.value[:: self.page_size] // self.page_size)
            stack.extend(n.children.values())
        return torch.cat(vals) if vals else self.empty

    def remap_pages(self, new_page: torch.Tensor) -> None:
        """Re-point every node at moved KV pages: ``new_page[p]`` is page ``p``'s new index."""
        ps = self.page_size
        stack = list(self.root.children.values())
        while stack:
            n = stack.pop()
            v = n.value
            if v is not None:
                n.set_key_value(n._key, (new_page[(v // ps).long()] * ps + v % ps).to(v.dtype))
            stack.extend(n.children.values())

    # ---------------------------------------------------------------- eviction (dual)
    def evict_cold(self, cutoff_seq: int) -> EvictResult:
        """Evict (spill to RAM when there is a RAM tier) every unlocked device frontier node last
        touched by an admission before ``cutoff_seq``, continuing up through parents that become
        cold frontier nodes. Frees their device snapshots too."""
        cands = [n for n in self._device_frontier() if n.ref_count == 0 and n.last_seq < cutoff_seq]
        kv, mamba = [], []
        while cands:
            node = cands.pop()
            if not self._evictable_frontier(node):
                continue
            parent = self._evict_node(node, kv, mamba)
            if self._evictable_frontier(parent) and parent.last_seq < cutoff_seq:
                cands.append(parent)
        return EvictResult(torch.cat(kv) if kv else self.empty, mamba)

    def evict_full(self, num_tokens: int) -> EvictResult:
        """Evict KV tokens by LRU over UNLOCKED LEAF nodes (an internal node's KV is a prefix
        dependency for all descendants). Frees each evicted node's snapshot too."""
        leaves = [n for n in self._device_frontier() if n.ref_count == 0]
        heapq.heapify(leaves)
        kv, mamba = [], []
        start = self.full_evictable
        while start - self.full_evictable < num_tokens and leaves:
            node = heapq.heappop(leaves)
            if not self._evictable_frontier(node):
                continue
            parent = self._evict_node(node, kv, mamba)
            if self._evictable_frontier(parent):
                heapq.heappush(leaves, parent)
        return EvictResult(torch.cat(kv) if kv else self.empty, mamba)

    def evict_mamba(self, num: int) -> EvictResult:
        """Evict GDN snapshots by LRU over UNLOCKED snapshot-bearing nodes -- internal nodes
        too. Internal node -> TOMBSTONE (free the slot, keep KV + children). Leaf node -> free
        both KV and slot and unlink, then cascade-delete any KV-only tombstone leaves it exposes
        upward (so a leaf always carries a live snapshot -- mirrors sglang)."""
        cands = [n for n in self._snapshot_nodes() if n.mamba_ref_count == 0]
        heapq.heapify(cands)
        kv, mamba, freed = [], [], 0
        while freed < num and cands:
            node = heapq.heappop(cands)
            if node.mamba_value is None or node.mamba_ref_count != 0 or node.is_root():
                continue
            if node.is_leaf() and node.ref_count == 0:
                self._evict_node(node, kv, mamba)  # with a RAM tier the snapshot spills there
                freed += 1
            else:
                self._free_node_mamba(node, mamba)  # tombstone internal (or locked-KV) node
                freed += 1
        return EvictResult(torch.cat(kv) if kv else self.empty, mamba)

    @property
    def full_evictable_size(self) -> int:
        return self.full_evictable

    @property
    def mamba_evictable_size(self) -> int:
        return self.mamba_evictable

    @property
    def size_info(self):
        """KV-page currency, for code that reads a BasePrefixCache size_info (metrics/usage).
        The GDN-snapshot currency is reported via mamba_evictable_size."""
        from .base import SizeInfo
        return SizeInfo(evictable_size=self.full_evictable, protected_size=self.full_protected)

    def check_integrity(self) -> None:
        # Structural: every snapshot-bearing node holds a real slot id; ref counts non-negative.
        # (KV/page conservation is checked by CacheManager.check_integrity.)
        for n in self._snapshot_nodes():
            assert n.mamba_value is not None and n.mamba_ref_count >= 0 and n.ref_count >= 0
        if self.host is not None:
            used_pages = used_snaps = 0
            for n in self._all_nodes():
                if n.value is not None:
                    assert n.parent.is_root() or n.parent.value is not None, "device node under a RAM-only one"
                else:
                    assert n.host_pages is not None and n.mamba_value is None and n.ref_count == 0
                if n.host_pages is not None:
                    assert len(n.host_pages) * self.page_size == n.length
                    used_pages += len(n.host_pages)
                used_snaps += n.host_mamba is not None
            assert used_pages + self.host.free_pages == self.host.num_pages, "RAM-tier page leak"
            assert used_snaps + self.host.free_snaps == self.host.num_snaps, "RAM-tier snapshot leak"

    # ---------------------------------------------------------------- helpers
    def _free_node_mamba(self, node: RadixTreeNode, out: List[int]) -> None:
        if node.mamba_value is not None:
            out.append(node.mamba_value)
            node.mamba_value = None
            if node.mamba_ref_count == 0:
                self.mamba_evictable -= 1

    def _unlink(self, node: RadixTreeNode) -> RadixTreeNode:
        parent = node.parent
        del parent.children[self.key_fn(node._key)]
        return parent

    def _cascade_tombstone_leaves(self, parent: RadixTreeNode, kv_out: List[torch.Tensor]):
        """After a leaf is unlinked, eagerly reclaim the KV-only tombstone leaves it exposes
        upward (mamba_value None, no children, unlocked): free their KV and unlink, walking up.
        Keeps the 'a leaf always carries a live snapshot' invariant (sglang
        _iteratively_delete_tombstone_leaf). Returns (highest surviving ancestor, freed_tokens)."""
        freed = 0
        while (parent.mamba_value is None and parent.host_mamba is None and parent.is_leaf()
               and parent.ref_count == 0 and not parent.is_root() and not parent.host_pinned):
            if parent.value is not None:
                kv_out.append(parent.value)
                self.full_evictable -= parent.length
                freed += parent.length
            if self.host is not None:
                self._release_host(parent)
            parent = self._unlink(parent)
        return parent, freed

    def _path_len(self, node: RadixTreeNode) -> int:
        n, total = node, 0
        while not n.is_root():
            total += n.length
            n = n.parent
        return total

    def _collect_kv(self, node: RadixTreeNode) -> torch.Tensor:
        vals: List[torch.Tensor] = []
        n = node
        while not n.is_root():
            vals.append(n.value)
            n = n.parent
        vals.reverse()
        return torch.cat(vals) if vals else self.empty

    def _all_nodes(self) -> List[RadixTreeNode]:
        out, stack = [], list(self.root.children.values())
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(n.children.values())
        return out

    def _device_frontier(self) -> List[RadixTreeNode]:
        """Device nodes with no device children (all leaves when there is no RAM tier)."""
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            dev = [c for c in n.children.values() if c.value is not None]
            if not dev and not n.is_root() and n.value is not None:
                out.append(n)
            stack.extend(dev)
        return out

    def _leaves(self) -> List[RadixTreeNode]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.is_leaf():
                if not n.is_root():
                    out.append(n)
            else:
                stack.extend(n.children.values())
        return out

    def _snapshot_nodes(self) -> List[RadixTreeNode]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.mamba_value is not None and not n.is_root():
                out.append(n)
            stack.extend(n.children.values())
        return out

    def _walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        prefix_len, total = 0, len(input_ids)
        node = self.root
        tic = time.monotonic_ns()
        while prefix_len < total:
            child = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child is None:
                return node, prefix_len
            node = child
            match_len = align_down(node.get_match_len(input_ids[prefix_len:]), self.page_size)
            prefix_len += match_len
            if match_len != node.length:
                node = node.split_at(match_len)
                node.timestamp = tic
                node.last_seq = self.req_seq
                return node, prefix_len
            node.timestamp = tic
            node.last_seq = self.req_seq
        return node, prefix_len


__all__ = ["HybridRadixCache", "HybridMatch", "EvictResult", "HybridCacheHandle"]

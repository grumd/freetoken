"""CacheManager under elastic VRAM: lowest-first pages, growth on a shortfall, the idle shrink
that never strands a cached page, and dropping prefixes the last few admissions did not touch."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager


class _Elastic:
    """ElasticVram's scheduler-facing surface, without VRAM."""

    def __init__(self, pages, limit, step, min_pages):
        self.pages, self.limit_pages, self.step_pages, self.min_pages = pages, limit, step, min_pages
        self.calls = []
        self.reclaims = 0  # the lent prefill reserve handed back

    def reclaim(self):
        self.reclaims += 1

    def growable_pages(self):
        return self.limit_pages - self.pages

    def grow(self, need):
        old = self.pages
        self.pages = min(self.limit_pages, old + -(-need // self.step_pages) * self.step_pages)
        self.calls.append(("grow", old, self.pages))
        return old, self.pages

    def shrink(self, pages):
        self.calls.append(("shrink", self.pages, pages))
        self.pages = pages

    def move_pages(self, src, dst):
        self.calls.append(("move", src.tolist(), dst.tolist()))
        if hasattr(self, "payload"):
            self.payload[dst] = self.payload[src]


def _pool():
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    return LinearStatePool(group=g, num_slots=16, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)


def _cm(pages=8, limit=64, step=8, min_pages=8, keep=2):
    el = _Elastic(pages, limit, step, min_pages)
    cm = CacheManager(limit, 1, torch.zeros(4, 128, dtype=torch.int32), "hybrid_radix",
                      linear_state_pool=_pool(), elastic=el, keep_requests=keep)
    return cm, el


def test_pages_come_lowest_first_and_a_shortfall_grows_the_pool():
    cm, el = _cm()
    assert cm.free_slots.tolist() == list(range(8))
    assert cm.available_size == 64  # 8 backed + 56 growable
    a = cm._allocate(5)
    cm._free(a[1:3].flip(0))
    assert cm.free_slots.tolist() == [1, 2, 5, 6, 7]  # sorted, not FIFO
    assert cm._allocate(2).tolist() == [1, 2]
    assert el.reclaims == 0  # served from free pages: a lent reserve stays lent
    got = cm._allocate(10)  # 3 free, so grow by whole steps
    assert el.calls == [("grow", 8, 16)]
    assert el.reclaims == 1  # growth and spills allocate, so the reserve comes back first
    assert got.tolist() == [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    assert cm.free_slots.tolist() == [15]


def test_idle_shrink_returns_free_tail_but_never_a_cached_page():
    cm, el = _cm(pages=40, min_pages=8)
    held = cm._allocate(3)  # pages 0..2 stay in use
    cm.elastic_idle()
    assert el.reclaims == 1  # between turns, before any page moves
    assert el.calls == [("shrink", 40, 16)]  # used 3 + one step of slack, rounded to a step
    assert cm.free_slots.tolist() == list(range(3, 16))
    cm._free(held)

    cm, el = _cm(pages=40, min_pages=8)
    pages = cm._allocate(30)
    cm._free(pages[:29])  # only page 29 is still held
    cm.elastic_idle()
    assert el.calls == [("shrink", 40, 32)]  # stops just above it
    assert int(cm.free_slots.max()) < 32


def _admit(cm, ids, pages, table_idx=0):
    from freetoken.core import Req, SamplingParams

    pend = SimpleNamespace(input_ids=torch.tensor(ids, dtype=torch.int32), input_len=len(ids), mm_embeds=None)
    mr = cm.match_req(pend)
    cm.lock(mr.cuda_handle)
    cm.note_admission(mr.cuda_handle)
    req = Req(input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=table_idx,
              cached_len=len(ids) - 1, output_len=1, uid=0, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    cm.page_table[table_idx, : len(pages)] = pages
    pool = cm.linear_state_pool
    req.linear_slot_idx, req.mamba_ping_pong = pool.alloc(1)[0], tuple(pool.alloc(2))
    req.mamba_next_track_idx = 1
    req.mamba_last_track_seqlen = len(ids) - 1
    cm.cache_req(req, finished=True)
    return req


def test_prefixes_untouched_by_the_last_admissions_are_dropped_first():
    cm, el = _cm(pages=16, limit=16, keep=2)
    _admit(cm, [1, 2, 3, 4, 9], cm._allocate(4))  # admission 1: session A
    _admit(cm, [5, 6, 7, 8, 9], cm._allocate(4))  # admission 2: session B
    _admit(cm, [1, 2, 3, 4, 9], torch.tensor([], dtype=torch.int32))  # 3: A again, a full hit
    free = len(cm.free_slots)
    cm._evict_cold()  # keep=2 -> admissions 2 and 3 are hot, nothing older exists
    assert len(cm.free_slots) == free
    _admit(cm, [1, 2, 3, 4, 9], torch.tensor([], dtype=torch.int32))  # 4: B is now cold
    cm._evict_cold()
    assert len(cm.free_slots) == free + 4
    assert cm.match_req(SimpleNamespace(input_ids=torch.tensor([5, 6, 7, 8, 9], dtype=torch.int32),
                                        input_len=5, mm_embeds=None)).cuda_handle.cached_len == 0
    assert cm.match_req(SimpleNamespace(input_ids=torch.tensor([1, 2, 3, 4, 9], dtype=torch.int32),
                                        input_len=5, mm_embeds=None)).cuda_handle.cached_len == 4
    cm.check_integrity()


def test_idle_compaction_moves_cached_pages_down_so_the_pool_can_shrink():
    cm, el = _cm(pages=48, limit=64, step=8, min_pages=8, keep=8)
    el.payload = torch.arange(48) * 10  # stands in for each page's KV bytes
    first = _admit(cm, [1, 2, 3, 4, 5, 6, 9], cm._allocate(6))  # session A: pages 0..5
    filler = cm._allocate(30)  # pages 6..35, a finished request's tail
    second = _admit(cm, [7, 8, 7, 8, 9], cm._allocate(4))  # session B: pages 36..39
    cm._free(filler)
    assert sorted(cm.prefix_cache._collect_kv(
        cm.prefix_cache.match_prefix(torch.tensor([7, 8, 7, 8], dtype=torch.int32)).node).tolist()) == [36, 37, 38, 39]
    cm.elastic_idle()
    # B's four pages moved into the lowest free pages, then the pool shrank under them
    moves = [c for c in el.calls if c[0] == "move"]
    assert moves == [("move", [36, 37, 38, 39], [6, 7, 8, 9])]
    assert el.calls[-1] == ("shrink", 48, 24)
    kv = cm.prefix_cache.match_prefix(torch.tensor([7, 8, 7, 8], dtype=torch.int32)).kv_indices
    assert kv.tolist() == [6, 7, 8, 9]
    assert el.payload[[6, 7, 8, 9]].tolist() == [360, 370, 380, 390]
    assert int(cm.free_slots.max()) < 24
    cm.check_integrity()

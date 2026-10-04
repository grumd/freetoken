"""P2b integration: CacheManager hybrid path (match_req -> cache_req donate -> prefix hit).
CPU, real LinearStatePool + page_table, hand-built Reqs. Exercises the two-currency wiring
without the full scheduler/engine."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager


def _pool(num_slots=16):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)


def _pend(ids):
    # int32 to match production Req.input_ids dtype (fast_compare_key needs consistent dtype)
    t = torch.tensor(ids, dtype=torch.int32)
    return SimpleNamespace(input_ids=t, input_len=len(ids), mm_embeds=None)


def test_hybrid_cache_manager_donate_then_hit():
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)
    assert cm.is_hybrid

    # cold match on an empty tree
    mr = cm.match_req(_pend([1, 2, 3, 4, 5]))
    assert mr.cuda_handle.cached_len == 0 and mr.mamba_value is None

    # admit req A: allocate live + ping-pong, stage KV pages, mark a ×N snapshot at boundary 4
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    page_table[0, :4] = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
    reqA = Req(input_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32), table_idx=0,
               cached_len=4, output_len=1, uid=0, sampling_params=SamplingParams(),
               cache_handle=mr.cuda_handle)
    reqA.linear_slot_idx, reqA.mamba_ping_pong = live, pp
    reqA.mamba_next_track_idx = 1            # flipped from 0 in build_fla_metadata; frozen = pp[0]
    reqA.mamba_last_track_seqlen = 4
    cm.lock(mr.cuda_handle)

    free_before = pool.num_free_slots
    cm.cache_req(reqA, finished=False)       # donate pp[0] at boundary 4; replace it in the pair
    # pp[0] donated to the tree; a fresh replacement was alloc'd -> net free-slot count unchanged
    assert pool.num_free_slots == free_before - 1  # one replacement alloc'd (donated slot now tree-owned)
    assert reqA.mamba_ping_pong[0] != pp[0]        # slot 0 replaced; pp[0] now lives in the tree

    # req B shares the [1,2,3,4] prefix -> HIT: returns the donated snapshot + reused KV
    mrB = cm.match_req(_pend([1, 2, 3, 4, 9]))
    assert mrB.cuda_handle.cached_len == 4
    assert mrB.mamba_value == pp[0]
    assert mrB.cuda_handle.get_matched_indices().tolist() == [100, 101, 102, 103]


def test_hybrid_finish_donates_live_slot():
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)

    mr = cm.match_req(_pend([7, 8, 9, 10]))
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    page_table[1, :3] = torch.tensor([200, 201, 202], dtype=torch.int32)
    req = Req(input_ids=torch.tensor([7, 8, 9, 10], dtype=torch.int32), table_idx=1,
              cached_len=3, output_len=1, uid=1, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    cm.lock(mr.cuda_handle)

    cm.cache_req(req, finished=True)         # donate the live slot directly (final state)
    # ping-pong pair freed; live slot kept (now owned by the tree)
    mr2 = cm.match_req(_pend([7, 8, 9, 10]))
    assert mr2.cuda_handle.cached_len == 3 and mr2.mamba_value == live


def test_free_req_slots_idempotent():
    """C2: a finish/abort double-free of the same request must NOT push its GDN slots twice."""
    pool = _pool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    req = Req(input_ids=torch.tensor([1, 2, 3], dtype=torch.int32), table_idx=0, cached_len=2,
              output_len=1, uid=0, sampling_params=SamplingParams(), cache_handle=None)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    base = pool.num_free_slots
    cm._free_req_slots(req)
    assert pool.num_free_slots == base + 3        # live + 2 ping-pong returned once
    cm._free_req_slots(req)                        # second free (abort/finish race)
    assert pool.num_free_slots == base + 3         # idempotent: nothing pushed twice


def test_rebuild_reclaims_donated_gdn_slots():
    """C5: a runtime cache rebuild must return the discarded tree's GDN snapshot slots (idle)."""
    pool = _pool(num_slots=16)
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    mr = cm.match_req(_pend([7, 8, 9, 10]))
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    pt[1, :3] = torch.tensor([200, 201, 202], dtype=torch.int32)
    req = Req(input_ids=torch.tensor([7, 8, 9, 10], dtype=torch.int32), table_idx=1, cached_len=3,
              output_len=1, uid=1, sampling_params=SamplingParams(), cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    cm.lock(mr.cuda_handle)
    cm.cache_req(req, finished=True)              # donates `live` to the tree, frees ping-pong
    assert pool.num_free_slots < pool.num_slots - 1   # a slot is now tree-owned
    cm.rebuild(64, pt)                            # idle rebuild discards the tree
    assert pool.num_free_slots == pool.num_slots - 1  # all GDN slots reclaimed (no leak)


def test_prefill_chunk_ends_on_a_page_boundary():
    """A hybrid chunk must end page-aligned: the snapshot commit skips any other boundary."""
    from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool()
    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "hybrid_radix", linear_state_pool=pool)
    assert cm.prefill_chunk_align == 64
    tm = TableManager(max_running_reqs=4, page_table=pt)
    pending = PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))

    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    req = adder.try_add_one(pending)
    assert isinstance(req, ChunkedReq) and req.extend_len == 64

    # a budget below one page keeps the unaligned chunk rather than stalling the request
    adder = PrefillAdder(token_budget=40, reserved_size=0, cache_manager=cm, table_manager=tm)
    assert adder.try_add_one(pending).extend_len == 40


def test_naive_cache_does_not_align_prefill_chunks():
    """The alignment hook is hybrid-only; every other cache keeps the raw budget chunk."""
    from freetoken.scheduler.prefill import PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "radix")
    assert cm.prefill_chunk_align == 1
    tm = TableManager(max_running_reqs=4, page_table=pt)
    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    pending = PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))
    assert adder.try_add_one(pending).extend_len == 100


def test_pool_sizing_covers_4mr_floor():
    """C6: pool must reserve the 4-slot-per-request non-evictable floor even at a tiny ratio."""
    from types import SimpleNamespace
    from freetoken.kvcache.linear_state_pool import _linear_pool_num_slots
    for mr in (1, 8, 64):
        c = SimpleNamespace(max_running_req=mr, cache_type="hybrid_radix",
                            linear_state_cache_ratio=0.1)
        assert _linear_pool_num_slots(c) >= 4 * mr + 1, (mr, _linear_pool_num_slots(c))


def _decoding_req(cm, pool, prompt, total, table_idx=0):
    """A request mid-decode: `total` tokens of ids, KV pages staged for all of them."""
    ids = list(prompt) + [1000 + i for i in range(total - len(prompt))]
    mr = cm.match_req(_pend(ids[: len(prompt)]))
    cm.page_table[table_idx, :total] = torch.arange(5000, 5000 + total, dtype=torch.int32)
    req = Req(input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=table_idx,
              cached_len=len(prompt), output_len=1, uid=table_idx,
              sampling_params=SamplingParams(), cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = pool.alloc(1)[0], tuple(pool.alloc(2))
    cm.lock(mr.cuda_handle)
    return req, ids


def _step_to(cm, pool, req, n, **kw):
    """Advance the request to cached_len == n with a recognizable live state, then run the
    pre-step snapshot hook."""
    req.cached_len = n
    pool.recurrent_states[:, req.linear_slot_idx] = float(n)
    cm.snapshot_decode_states([req], **kw)


def test_decode_boundary_snapshots_become_reuse_points():
    """The next turn echoes the previous output: decode-time boundary snapshots, donated at
    finish, let it resume at the last boundary instead of the end of the previous prompt."""
    pool = _pool()
    cm = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "hybrid_radix",
                      linear_state_pool=pool)
    cm.decode_snapshot_interval = 4
    base = pool.num_free_slots
    req, ids = _decoding_req(cm, pool, prompt=[1, 2, 3, 4, 5, 6], total=19)
    for n in range(6, 19):
        _step_to(cm, pool, req, n, anchors=False, boundaries=True)
    # boundaries 8, 12, 16 crossed; the ping-pong pair keeps the latest two
    assert sorted(req.mamba_decode_tracks.values()) == [12, 16]
    req.cached_len = 17
    cm.cache_req(req, finished=True)
    assert pool.num_free_slots == base - 3       # both snapshots + the live slot tree-owned

    assert cm.match_req(_pend(ids[:17] + [7, 7])).cuda_handle.cached_len == 17
    hit = cm.match_req(_pend(ids[:16] + [7, 7]))
    assert hit.cuda_handle.cached_len == 16
    assert float(pool.recurrent_states[0, hit.mamba_value].max()) == 16.0
    assert hit.cuda_handle.get_matched_indices().tolist() == list(range(5000, 5016))
    early = cm.match_req(_pend(ids[:14] + [7, 7]))  # diverges inside the last interval
    assert early.cuda_handle.cached_len == 12
    assert float(pool.recurrent_states[0, early.mamba_value].max()) == 12.0


def test_decode_boundary_snapshots_off_and_anchor_pinned():
    pool = _pool()
    cm = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "hybrid_radix",
                      linear_state_pool=pool)
    cm.decode_snapshot_interval = 4
    req, _ = _decoding_req(cm, pool, prompt=[1, 2, 3, 4, 5, 6], total=30)
    for n in range(6, 12):
        _step_to(cm, pool, req, n, anchors=False, boundaries=False)
    assert req.mamba_decode_tracks == {}         # speculative decoding: no periodic copies

    req.toolcall_anchor_len = 13
    for n in range(12, 26):
        _step_to(cm, pool, req, n, anchors=True, boundaries=True)
    # the anchor snapshot survives later boundaries; the other slot holds the latest one
    assert req.mamba_pinned_track is not None
    assert req.mamba_decode_tracks[req.mamba_pinned_track] == 13
    assert req.mamba_decode_tracks[1 - req.mamba_pinned_track] == 24
    slot = req.mamba_ping_pong[req.mamba_pinned_track]
    assert float(pool.recurrent_states[0, slot].max()) == 13.0


def test_decode_snapshots_at_page_size_64():
    """qsa_sparse runs hybrid models at page_size 64: the finish state is almost never
    page-aligned, so the last decode boundary is the deepest reuse point of the turn."""
    pool = _pool()
    cm = CacheManager(16, 64, torch.zeros(2, 1024, dtype=torch.int32), "hybrid_radix",
                      linear_state_pool=pool)
    assert cm.decode_snapshot_interval == 64
    base = pool.num_free_slots
    req, ids = _decoding_req(cm, pool, prompt=list(range(1, 71)), total=200)
    for n in range(70, 200):
        _step_to(cm, pool, req, n, anchors=False, boundaries=True)
    assert sorted(req.mamba_decode_tracks.values()) == [128, 192]
    cm.cache_req(req, finished=True)
    assert pool.num_free_slots == base - 2       # snapshots tree-owned, unaligned live freed
    hit = cm.match_req(_pend(ids[:199] + [7] * 40))
    assert hit.cuda_handle.cached_len == 192
    assert float(pool.recurrent_states[0, hit.mamba_value].max()) == 192.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")

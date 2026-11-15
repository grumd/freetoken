"""Elastic expert slots (elastic VRAM): fenced slots are never LRU victims, shrinking keeps what a
smaller LRU would keep and moves the bank bytes with it, growing reopens empty slots, and the
whole thing runs on CUDA-VMM banks whose retired rows are actually unmapped."""

from __future__ import annotations

import math

import pytest
import torch

from freetoken.moe.offload_cache import FENCE_USAGE, _BANK_SCHEMAS, OffloadMoeCache

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

L, E, CAP = 6, 64, 512
# mxfp4_triton's 6-bank schema; the big rows make a few slots fill a 2 MiB chunk
FEATS = [65536, 512, 256, 32768, 512, 256]


def _build(active: int, prefill_overlap: bool = False):
    from freetoken.kernel.vmm import ChunkPool, ElasticTensor

    dev = torch.device("cuda", torch.cuda.current_device())
    pool = ChunkPool(dev)
    banks = []

    def alloc(name, shape, dtype):
        row = math.prod(shape[1:]) * dtype.itemsize
        pool.create(-(-active * row // pool.chunk_bytes) + 1)
        t = ElasticTensor(pool, shape, dtype, [(0, active * row)])
        banks.append((t, row))
        return t.tensor

    cache = OffloadMoeCache(
        num_layers=L, num_experts=E, cache_size=CAP, device=dev, prefill_overlap=prefill_overlap,
        quant_format="mxfp4_triton", bank_alloc=alloc, active_slots=active,
    )
    gen = torch.Generator(device="cuda").manual_seed(0)
    sources = {
        name: list(torch.randint(0, 256, (L * E, feat), dtype=torch.uint8, device=dev, generator=gen)
                   .split(E))
        for name, feat in zip(_BANK_SCHEMAS["mxfp4_triton"], FEATS)
    }
    cache.set_bank_sources(sources)
    assert cache._copy_fused_ok
    return cache, pool, banks


def _remap(banks, slots):
    torch.cuda.synchronize()
    for t, row in banks:
        t.set_required([(0, slots * row)])


def _decode(cache, steps, seed):
    gen = torch.Generator().manual_seed(seed)
    seen_max = -1
    for _ in range(steps):
        for layer in range(L):
            # skewed routing so recency means something
            ids = (torch.rand((1, 10), generator=gen) ** 2 * E).to(torch.int32).cuda()
            cache.ensure_experts(layer, ids)
            cache.copy_missing()
            seen_max = max(seen_max, int(ids.max()))
    torch.cuda.synchronize()
    return seen_max


def _assert_coherent(cache):
    ids = cache.id_of_slot.tolist()
    for slot, gid in enumerate(ids):
        if gid < 0:
            continue
        layer, e = divmod(gid, E)
        assert int(cache.slot_for_id[layer, e]) == slot
        for src, dst in cache.banks:
            assert torch.equal(dst[slot], src[layer][e]), f"slot {slot} holds stale bytes for id {gid}"
    flat = cache.slot_for_id.view(-1).tolist()
    assert sum(s >= 0 for s in flat) == sum(g >= 0 for g in ids)


@CUDA
def test_fenced_slots_are_never_victims_and_reset_refences():
    cache, _, _ = _build(active=200)
    assert int(cache.usage[200:].min()) == FENCE_USAGE
    assert _decode(cache, 300, seed=1) < 200
    assert int(cache.id_of_slot[200:].max()) == -1
    assert int(cache.usage[200:].min()) == FENCE_USAGE
    _assert_coherent(cache)
    cache.reset()
    assert int(cache.usage[200:].min()) == FENCE_USAGE and int(cache.usage[:200].max()) == 0


@CUDA
def test_shrink_keeps_the_most_recent_experts_and_grow_reopens_slots():
    cache, pool, banks = _build(active=300)
    _decode(cache, 200, seed=2)
    before = {int(g): int(u) for g, u in zip(cache.id_of_slot.tolist(), cache.usage.tolist()) if g >= 0}
    moved = cache.shrink_active(120)
    assert moved > 0
    _remap(banks, 120)  # retired rows are now unbacked: touching them would fault
    after = {int(g) for g in cache.id_of_slot.tolist() if g >= 0}
    assert after <= set(before)
    dropped = set(before) - after
    assert min(before[g] for g in after) >= max(before[g] for g in dropped)
    assert int(cache.id_of_slot[120:].max()) == -1 and int(cache.usage[120:].min()) == FENCE_USAGE
    _assert_coherent(cache)

    assert _decode(cache, 100, seed=3) < 120
    _assert_coherent(cache)

    _remap(banks, 400)
    cache.grow_active(400)
    assert int(cache.usage[120:400].max()) == 0
    assert 120 <= _decode(cache, 200, seed=4) < 400
    _assert_coherent(cache)
    assert pool.num_handles - pool.num_free == sum(t.num_mapped for t, _ in banks)


@CUDA
def test_elastic_cache_cannot_rebuild_or_shrink_below_its_floor():
    cache, _, _ = _build(active=200)
    with pytest.raises(RuntimeError, match="elastic"):
        cache.rebuild(300)
    with pytest.raises(AssertionError):
        cache.shrink_active(E - 1)


def _resident(cache):
    return {int(g): int(u) for g, u in zip(cache.id_of_slot.tolist(), cache.usage.tolist()) if g >= 0}


def _routed_prefill(cache, demote: bool) -> int:
    """One routed prefill chunk over every layer: each routes to all E experts and fetches half
    of its misses (the CPU-assisted prefill's capped fetch). Returns the largest per-layer fetch."""
    most = 0
    for layer in range(L):
        ids = torch.arange(E, dtype=torch.int32, device="cuda").view(1, -1)
        cache.ensure_experts_capped(layer, ids, 0.5)
        cache.copy_missing()
        n = int(cache.num_indices[0])
        most = max(most, n)
        if demote:
            cache.demote_fetched(n)
    torch.cuda.synchronize()
    return most


@CUDA
@pytest.mark.parametrize("demote", [False, True])
def test_routed_prefill_fetches_do_not_flush_decode_experts(demote):
    cache, _, _ = _build(active=200)
    _decode(cache, 300, seed=5)
    warm = set(_resident(cache))
    most = _routed_prefill(cache, demote)
    kept = warm & set(_resident(cache))
    _assert_coherent(cache)
    if demote:
        # demoted fetches evict each other: the chunk costs decode about one layer's fetches
        assert len(warm) - len(kept) <= most
    else:
        # stamped as recent, the fetches push most of decode's experts out (the regression)
        assert len(warm) - len(kept) > 2 * most
    # decode keeps working on top of it
    _decode(cache, 50, seed=6)
    _assert_coherent(cache)


@CUDA
def test_streamed_prefill_buffers_evict_the_coldest_experts():
    active = 300
    cache, _, _ = _build(active=active, prefill_overlap=True)
    _decode(cache, 300, seed=7)
    # decode after a streamed chunk refills the empty buffer slots [0, 2E) first, so they hold
    # its most recent fetches: invalidate them, as a chunk does, and decode again
    for b in range(2):
        cache._invalidate_prefill_buffer(b)
    _decode(cache, 40, seed=8)
    before = _resident(cache)
    assert any(s < 2 * E for s in range(active) if int(cache.id_of_slot[s]) >= 0)
    cache.begin_prefill()
    for b in range(2):  # what the chunk's prefetches do to the buffer range
        cache._invalidate_prefill_buffer(b)
    torch.cuda.synchronize()
    after = _resident(cache)
    assert set(after) <= set(before)
    survivors = sorted(before.values(), reverse=True)[: active - 2 * E]
    # the survivors are what an LRU without the buffer range would hold
    assert sorted(after.values(), reverse=True) == survivors
    _assert_coherent(cache)

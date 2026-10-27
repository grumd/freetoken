"""ElasticVram end to end on a real QSA KV pool and a real expert slot cache: startup gives the
spare pool to the experts, KV growth retires expert slots (and back), live KV pages and resident
experts survive every transition byte for byte, and no physical chunk is leaked."""

from __future__ import annotations

import pytest
import torch

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

L, E = 4, 64
FEATS = [65536, 512, 256, 32768, 512, 256]


def _build(cap_pages=1024, spare_mib=16):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.engine.elastic import ElasticVram
    from freetoken.kvcache.mha_pool import elastic_tier_allocator
    from freetoken.kvcache.qsa_pool import QSAKVCache
    from freetoken.moe.offload_cache import _BANK_SCHEMAS, OffloadMoeCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    dev = torch.device("cuda", torch.cuda.current_device())
    el = ElasticVram(dev, min_pages=64, step_pages=32, floor_slots=E, cap_slots=L * E, cap_pages=cap_pages)
    cache = OffloadMoeCache(
        num_layers=L, num_experts=E, cache_size=L * E, device=dev, prefill_overlap=False,
        quant_format="mxfp4_triton", bank_alloc=el.bank, active_slots=el.slots,
    )
    el.cache = cache
    gen = torch.Generator(device="cuda").manual_seed(0)
    cache.set_bank_sources({
        name: list(torch.randint(0, 256, (L * E, f), dtype=torch.uint8, device=dev, generator=gen).split(E))
        for name, f in zip(_BANK_SCHEMAS["mxfp4_triton"], FEATS)
    })
    with elastic_tier_allocator(el.kv_tier):
        kv = QSAKVCache(
            num_kv_heads=1, num_layers=8, head_dim=256, num_pages=cap_pages + 1, page_size=64,
            dtype=torch.bfloat16, device=dev, index_head_dim=128, num_index_layers=2,
            index_ratio=4, num_req_slots=2, layer_ids=[1, 3, 5, 7],
        )
    assert kv.elastic
    el.finalize(spare_mib << 20, 0, 0, None)
    return el, cache, kv


def _decode(cache, steps, seed):
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        for layer in range(L):
            ids = (torch.rand((1, 10), generator=gen) ** 2 * E).to(torch.int32).cuda()
            cache.ensure_experts(layer, ids)
            cache.copy_missing()
            assert int(ids.max()) < cache.active_slots
    torch.cuda.synchronize()


def _coherent(cache):
    for slot, gid in enumerate(cache.id_of_slot.tolist()):
        if gid >= 0:
            layer, e = divmod(gid, E)
            for src, dst in cache.banks:
                assert torch.equal(dst[slot], src[layer][e])


def _stamp(kv, pages):
    """Write each live page's own index into every K/V layer and the index-key rows."""
    for layer in (1, 3, 5, 7):
        for buf in (kv.k_cache(layer), kv.v_cache(layer)):
            buf[:pages] = torch.arange(pages, device="cuda", dtype=torch.bfloat16).view(-1, 1, 1, 1)
    for s in range(2):
        rows = kv.cmp_k_cache(s)
        rows[: pages * 16] = torch.arange(pages, device="cuda").repeat_interleave(16).to(torch.bfloat16).view(-1, 1)


def _check_stamp(kv, pages):
    want = torch.arange(pages, device="cuda", dtype=torch.bfloat16)
    for layer in (1, 3, 5, 7):
        for buf in (kv.k_cache(layer), kv.v_cache(layer)):
            assert torch.equal(buf[:pages].amax(dim=(1, 2, 3)), want)
    for s in range(2):
        assert torch.equal(kv.cmp_k_cache(s)[: pages * 16, 0], want.repeat_interleave(16))


def _no_leak(el):
    mapped = sum(t.num_mapped for t, _ in el._kv) + sum(t.num_mapped for t, _ in el._banks)
    assert el.pool.num_handles - el.pool.num_free == mapped
    assert el.pool.num_free * el.pool.chunk_bytes < 2 * len(el._kv + el._banks) * el.pool.chunk_bytes + el.pool.chunk_bytes * 8


@CUDA
def test_kv_growth_takes_expert_slots_and_gives_them_back_intact():
    el, cache, kv = _build()
    s0 = el.slots
    assert el.pages == 64 and s0 > E and cache.active_slots == s0
    _decode(cache, 100, seed=1)
    _stamp(kv, 64)

    assert el.limit_pages >= 96 and s0 < L * E  # experts, not the slot cap, bound the pool
    old, new = el.grow(20)  # one whole step
    assert (old, new) == (64, 96) and el.slots < s0 and cache.active_slots == el.slots
    _check_stamp(kv, 64)
    # pages the pool just backed start zeroed (the slab's score kernel relies on it)
    assert int(kv.k_cache(1)[64:96].count_nonzero()) == 0
    assert int(kv.cmp_k_cache(0)[64 * 16 : 96 * 16].count_nonzero()) == 0
    _coherent(cache)
    _decode(cache, 50, seed=2)
    _stamp(kv, 96)
    _no_leak(el)

    el.shrink(64)
    assert el.pages == 64 and el.slots == s0
    _check_stamp(kv, 64)
    _decode(cache, 50, seed=3)
    _coherent(cache)
    _no_leak(el)


@CUDA
def test_kv_growth_stops_where_experts_hit_their_floor():
    el, cache, kv = _build(spare_mib=8)
    assert el.limit_pages < 1024
    old, new = el.grow(10_000)
    assert new == el.limit_pages and el.slots >= E
    assert el.growable_pages() == 0
    _decode(cache, 30, seed=4)
    _coherent(cache)
    _no_leak(el)


@CUDA
def test_page_moves_carry_every_kv_tier():
    el, cache, kv = _build()
    el.kv_pool = kv
    _stamp(kv, 64)
    el.move_pages(torch.tensor([60, 61, 63]), torch.tensor([3, 5, 7]))
    for layer in (1, 3, 5, 7):
        for buf in (kv.k_cache(layer), kv.v_cache(layer)):
            assert buf[[3, 5, 7]].amax(dim=(1, 2, 3)).tolist() == [60, 61, 63]
            assert buf[[60, 4, 6]].amax(dim=(1, 2, 3)).tolist() == [60, 4, 6]  # sources and bystanders keep
    for s in range(2):
        rows = kv.cmp_k_cache(s)[:, 0]
        assert rows[3 * 16 : 4 * 16].tolist() == [60.0] * 16 and rows[7 * 16 : 8 * 16].tolist() == [63.0] * 16
        assert rows[4 * 16 : 5 * 16].tolist() == [4.0] * 16

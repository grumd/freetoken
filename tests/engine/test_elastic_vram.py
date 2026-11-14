"""ElasticVram end to end on a real QSA KV pool and a real expert slot cache: startup gives the
spare pool to the experts, KV growth retires expert slots (and back), live KV pages and resident
experts survive every transition byte for byte, and no physical chunk is leaked."""

from __future__ import annotations

import pytest
import torch

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

L, E = 4, 64
FEATS = [65536, 512, 256, 32768, 512, 256]


def _build(cap_pages=1024, spare_mib=16, boost_mib=0):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.engine.elastic import ElasticVram
    from freetoken.kvcache.mha_pool import elastic_tier_allocator
    from freetoken.kvcache.qsa_pool import QSAKVCache
    from freetoken.moe.offload_cache import _BANK_SCHEMAS, OffloadMoeCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    dev = torch.device("cuda", torch.cuda.current_device())
    el = ElasticVram(dev, min_pages=64, step_pages=32, floor_slots=E, cap_slots=L * E, cap_pages=cap_pages,
                     boost_bytes=boost_mib << 20)
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
def test_a_long_decode_borrows_the_prefill_reserve_and_hands_it_back_intact():
    el, cache, kv = _build(spare_mib=8)
    el.lend_after = 3
    s0, handles0 = el.slots, el.pool.num_handles
    _decode(cache, 50, seed=5)
    _stamp(kv, 64)
    el.decode_step(prefill_pending=False)
    el.decode_step(prefill_pending=True)
    el.decode_step(prefill_pending=True)
    assert el.lent == 0  # a waiting prefill keeps the reserve
    el.decode_step(prefill_pending=False)
    assert el.lent > 0 and el.slots == L * E and cache.active_slots == el.slots
    _decode(cache, 50, seed=6)
    _coherent(cache)
    _no_leak(el)

    el.reclaim()
    assert el.lent == 0 and el.slots == s0 and el.pool.num_handles == handles0
    _check_stamp(kv, 64)
    _coherent(cache)
    _decode(cache, 30, seed=7)
    _no_leak(el)
    el.decode_step(prefill_pending=False)
    assert el.lent == 0  # the quiet-step count restarts after a reclaim


@CUDA
def test_lend_after_zero_never_lends():
    el, cache, kv = _build(spare_mib=8)
    for _ in range(200):
        el.decode_step(prefill_pending=False)
    assert el.lent == 0


@CUDA
def test_a_long_prefill_borrows_expert_slots_and_hands_them_back_intact():
    el, cache, kv = _build(spare_mib=16, boost_mib=4)
    assert el.boost_chunks * el.pool.chunk_bytes >= 4 << 20
    assert el.pages <= el.boost_limit_pages < el.limit_pages
    s0, handles0 = el.slots, el.pool.num_handles
    _decode(cache, 50, seed=8)
    _stamp(kv, 64)

    assert el.boost() and el.boosted == el.boost_chunks
    assert el.pool.num_handles == handles0 - el.boost_chunks  # back with the driver for activations
    assert el.slots < s0 and cache.active_slots == el.slots
    assert el.boost() and el.pool.num_handles == handles0 - el.boost_chunks  # a no-op while boosted
    _check_stamp(kv, 64)
    _coherent(cache)
    _no_leak(el)
    # KV can grow only as far as the boost leaves room for
    assert el.growable_pages() == el.boost_limit_pages - el.pages
    old, new = el.grow(10_000)
    assert new == el.boost_limit_pages and el.slots >= E
    _decode(cache, 30, seed=9)
    _coherent(cache)
    _no_leak(el)
    el.shrink(64)

    el.decode_step(prefill_pending=False)  # the first decode step takes the slots back
    assert el.boosted == 0 and el.pool.num_handles == handles0 and el.slots == s0
    _check_stamp(kv, 64)
    _coherent(cache)
    _decode(cache, 30, seed=10)
    _no_leak(el)

    # with more KV backed than a boost leaves room for, a long chunk is refused
    el.grow(el.boost_limit_pages - el.pages + 1)
    assert el.pages > el.boost_limit_pages
    assert not el.boost() and el.boosted == 0 and el.pool.num_handles == handles0


@CUDA
def test_an_unboost_short_of_vram_retries_on_one_gpu_and_fails_loudly_under_tp(monkeypatch):
    el, cache, kv = _build(spare_mib=16, boost_mib=4)
    assert el.boost_chunks >= 2
    handles0, create = el.pool.num_handles, el.pool.create
    made = []

    def flaky(n):
        if made:
            raise RuntimeError("CUDA driver VMM call failed: out of memory")
        made.append(n)
        create(n)

    assert el.boost()
    monkeypatch.setattr(el.pool, "create", flaky)
    el.unboost()  # one GPU: what came back is kept, the rest is retried, KV stays capped meanwhile
    assert el.boosted == el.boost_chunks - 1 and el.growable_pages() == el.boost_limit_pages - el.pages
    _no_leak(el)
    monkeypatch.setattr(el.pool, "create", create)
    el.unboost()
    assert el.boosted == 0 and el.pool.num_handles == handles0
    _coherent(cache)
    _no_leak(el)

    # TP: the ranks vote, so they leave boost together or the server stops (never diverges)
    votes = []

    def all_reduce(t, op=None, group=None):
        votes.append(int(t[0]))
        t.fill_(min(votes[-1], peer_ok))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(el, "_tp_group", object())
    peer_ok = 1
    assert el.boost()
    el.unboost()
    assert votes == [1] and el.boosted == 0 and el.pool.num_handles == handles0
    peer_ok = 0  # this rank got its chunks back, another did not
    assert el.boost()
    with pytest.raises(RuntimeError, match="a TP rank could not take"):
        el.unboost()
    assert votes == [1, 1]


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compact_small_weights_releases_interleaved_holes():
    """--dense-mxfp8-groups frees bf16 weights that sit between per-layer tensors that stay; the
    allocator keeps such holes reserved until the survivors are repacked. Tensors referenced from
    elsewhere (an alias, a view sharing the storage) must stay where they are."""
    from freetoken.engine.engine import _compact_small_weights
    from freetoken.layers.base import BaseOP

    # just over 1 MiB: blocks of <= 1 MiB come from the small pool, which packs them away from
    # the freed 3 MiB blocks and leaves no hole to release
    kept_bytes = (1 << 20) + 4096

    class Layer(BaseOP):
        def __init__(self, i):
            self.freed = torch.full((3 << 20,), i % 251, dtype=torch.uint8, device="cuda")
            self.kept = torch.full((kept_bytes,), i % 7, dtype=torch.uint8, device="cuda")

    class Root(BaseOP):
        def __init__(self):
            self.layers = [Layer(i) for i in range(64)]

    torch.cuda.empty_cache()
    base = torch.cuda.memory_reserved()
    root = Root()
    alias = root.layers[3].kept
    root.layers[4].view = root.layers[4].kept[:10]
    for layer in root.layers:
        layer.freed = None
    torch.cuda.empty_cache()
    before = torch.cuda.memory_reserved() - base
    moved = _compact_small_weights(root)
    after = torch.cuda.memory_reserved() - base
    assert moved == 62 * kept_bytes
    # how much comes back depends on the allocator config (the engine runs expandable segments)
    assert after < before, (before, after)
    assert root.layers[3].kept is alias
    assert root.layers[4].kept.data_ptr() == root.layers[4].view.data_ptr()
    assert all(int(layer.kept[0]) == i % 7 and layer.kept.is_cuda for i, layer in enumerate(root.layers))

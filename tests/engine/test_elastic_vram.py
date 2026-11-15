"""ElasticVram end to end on a real QSA KV pool and a real expert slot cache: startup gives the
spare pool to the experts, KV growth retires expert slots (and back), live KV pages and resident
experts survive every transition byte for byte, and no physical chunk is leaked."""

from __future__ import annotations

import pytest
import torch

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

L, E = 4, 64
FEATS = [65536, 512, 256, 32768, 512, 256]


# prefill workspace in test units: 2048 extra tokens per 8 MiB
PF_MIN, PF_MAX, PF_BPT = 64, 4096, 4096


def _build(cap_pages=1024, spare_mib=16, tp_group=None):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.engine.elastic import ElasticVram
    from freetoken.kvcache.mha_pool import elastic_tier_allocator
    from freetoken.kvcache.qsa_pool import QSAKVCache
    from freetoken.moe.offload_cache import _BANK_SCHEMAS, OffloadMoeCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    dev = torch.device("cuda", torch.cuda.current_device())
    el = ElasticVram(dev, min_pages=64, step_pages=32, floor_slots=E, cap_slots=L * E, cap_pages=cap_pages,
                     prefill_min_tokens=PF_MIN, prefill_max_tokens=PF_MAX, prefill_bytes_per_token=PF_BPT)
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
    el.finalize(spare_mib << 20, 0, 0, tp_group)
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
    el.settle(prefill_pending=False, decoding=True)
    el.decode_step()
    el.decode_step()
    el.settle(prefill_pending=True, decoding=True)
    assert el.lent == 0  # a waiting prefill keeps the reserve
    el.settle(prefill_pending=False, decoding=False)
    assert el.lent == 0  # nor is it lent with nothing decoding
    el.settle(prefill_pending=False, decoding=True)  # before the third decode step
    assert el.lent > 0 and el.slots == L * E and cache.active_slots == el.slots
    _decode(cache, 50, seed=6)
    _coherent(cache)
    _no_leak(el)

    el.hold_prefill(0)  # before KV growth or a spill: the lend ends
    assert el.lent == 0 and el.slots == s0 and el.pool.num_handles == handles0
    _check_stamp(kv, 64)
    _coherent(cache)
    _decode(cache, 30, seed=7)
    _no_leak(el)
    el.settle(prefill_pending=False, decoding=True)
    assert el.lent == 0  # the decode-step count restarts after a hold


@CUDA
def test_lend_after_zero_never_lends():
    el, cache, kv = _build(spare_mib=8)
    for _ in range(200):
        el.decode_step()
        el.settle(prefill_pending=False, decoding=True)
    assert el.lent == 0


def _chunks(el, tokens):
    return -(-PF_BPT * max(0, tokens - PF_MIN) // el.pool.chunk_bytes)


@CUDA
def test_a_short_prompt_takes_nothing_and_a_long_one_only_what_its_chunk_needs():
    el, cache, kv = _build(spare_mib=16)
    s0, handles0 = el.slots, el.pool.num_handles
    _decode(cache, 50, seed=8)
    _stamp(kv, 64)

    assert el.prefill_tokens(50) == 50 and el.prefill_tokens(PF_MIN) == PF_MIN
    el.hold_prefill(50)  # the standing reserve covers it
    assert el.taken == 0 and el.slots == s0 and el.pool.num_handles == handles0

    t = PF_MIN + el.pool.chunk_bytes // PF_BPT  # exactly one chunk beyond the reserve
    assert el.prefill_tokens(t) == t
    el.hold_prefill(t)
    assert el.taken == 1 and el.pool.num_handles == handles0 - 1  # back with the driver for activations
    assert el.slots < s0 and cache.active_slots == el.slots
    one_chunk_slots = s0 - el.slots
    el.hold_prefill(2000)
    assert el.taken == _chunks(el, 2000) == 4 and el.pool.num_handles == handles0 - 4
    el.hold_prefill(t)  # within a prompt the workspace only grows
    assert el.taken == 4
    assert one_chunk_slots < s0 - el.slots
    _check_stamp(kv, 64)
    _coherent(cache)
    _no_leak(el)
    # KV can grow only as far as the workspace leaves room for
    assert el.growable_pages() == el._pages_within(el.room - 4) - el.pages < el.limit_pages - el.pages
    old, new = el.grow(10_000)
    assert new == el._pages_within(el.room - 4) and el.slots >= E
    _decode(cache, 30, seed=9)
    _coherent(cache)
    _no_leak(el)
    el.shrink(64)

    el.settle(prefill_pending=True, decoding=True)
    assert el.taken == 4  # kept while the prompt is still prefilling
    el.settle(prefill_pending=False, decoding=False)  # given back once it is in
    assert el.taken == 0 and el.pool.num_handles == handles0 and el.slots == s0
    _check_stamp(kv, 64)
    _coherent(cache)
    _decode(cache, 30, seed=10)
    _no_leak(el)


@CUDA
def test_the_first_decode_step_gives_the_workspace_back_and_lends_the_reserve_at_once():
    el, cache, kv = _build(spare_mib=16)
    el.lend_after = 1
    s0 = el.slots
    el.hold_prefill(2000)
    assert el.taken == 4
    transitions = el.transitions
    el.settle(prefill_pending=False, decoding=True)
    assert el.taken == 0 and el.lent > 0 and el.slots > s0 and el.transitions == transitions + 1
    el.release_prefill()  # the idle point ends the lend
    assert el.lent == 0 and el.slots == s0
    _coherent(cache)
    _no_leak(el)


@CUDA
def test_chunks_shrink_with_the_context_and_the_minimum_always_fits():
    el, cache, kv = _build(spare_mib=24)
    sizes = []
    while True:
        n = el.prefill_tokens(10**6)
        sizes.append(n)
        assert PF_MIN <= n <= PF_MAX
        spare = el.room - el._kv_chunks(el.pages)
        # the largest chunk that fits: one more token would need a chunk the pool does not have
        assert _chunks(el, n) <= spare and (n == PF_MAX or _chunks(el, n + 1) > spare)
        el.hold_prefill(n)
        assert el.slots >= E and cache.active_slots == el.slots
        _decode(cache, 5, seed=len(sizes))
        _coherent(cache)
        _no_leak(el)
        el.release_prefill()
        if el.pages == el.limit_pages:
            break
        el.grow(1)
    assert sizes[0] == PF_MAX and sizes[-1] < PF_MAX and sizes == sorted(sizes, reverse=True)
    # at the context limit, with the experts at their floor, the minimum chunk takes nothing
    assert el._max_slots(el.pages, el.base - el._workspace_chunks(sizes[-1])) >= E
    assert _chunks(el, PF_MIN) == 0


@CUDA
def test_tp_ranks_size_chunks_alike_whatever_their_expert_banks_hold(monkeypatch):
    """The sizing reads only the page count and the room settled across ranks: a rank with less
    VRAM for experts sets the room, and both ranks then pick the same chunk at every KV size."""
    small, _, _ = _build(spare_mib=12)
    kv = small._kv_chunks(small.cap_pages)
    peer = torch.tensor([small.room, kv, -kv, small.pool.chunk_bytes, -small.pool.chunk_bytes], dtype=torch.int64)

    def all_reduce(t, op=None, group=None):
        if t.numel() == 5:
            t.copy_(torch.minimum(t, peer))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    big, cache, _ = _build(spare_mib=24, tp_group=object())
    assert big.base > small.base and big.room == small.room and big.limit_pages == small.limit_pages
    while True:
        assert big.prefill_tokens(10**6) == small.prefill_tokens(10**6)
        assert big.prefill_tokens(1000) == small.prefill_tokens(1000)
        big.hold_prefill(big.prefill_tokens(10**6))
        small.hold_prefill(small.prefill_tokens(10**6))
        assert big.taken == small.taken and big.slots > small.slots >= E  # slot counts may differ per rank
        big.release_prefill()
        small.release_prefill()
        if big.pages == big.limit_pages:
            break
        big.grow(1)
        small.grow(1)
        assert big.pages == small.pages

    peer[1] -= 1  # a rank whose KV maps differently would size differently: refuse to start
    with pytest.raises(RuntimeError, match="TP ranks differ"):
        _build(spare_mib=24, tp_group=object())


@CUDA
def test_an_explicit_reserve_fixes_every_chunk_at_the_largest():
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.engine.elastic import ElasticVram

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    dev = torch.device("cuda", torch.cuda.current_device())
    el = ElasticVram(dev, min_pages=64, step_pages=32, floor_slots=E, cap_slots=L * E, cap_pages=1024,
                     prefill_min_tokens=PF_MAX, prefill_max_tokens=PF_MAX, prefill_bytes_per_token=PF_BPT)
    assert el.prefill_tokens(100) == 100 and el.prefill_tokens(10**6) == PF_MAX
    assert el._workspace_chunks(PF_MAX) == 0


@CUDA
def test_a_give_back_short_of_vram_retries_on_one_gpu_and_fails_loudly_under_tp(monkeypatch):
    el, cache, kv = _build(spare_mib=16)
    handles0, create = el.pool.num_handles, el.pool.create

    def flaky(n):
        create(n - 1)
        raise RuntimeError("CUDA driver VMM call failed: out of memory")

    el.hold_prefill(2000)
    assert el.taken == 4
    monkeypatch.setattr(el.pool, "create", flaky)
    el.release_prefill()  # one GPU: what came back is kept, the rest is retried, KV stays capped meanwhile
    assert el.taken == 1 and el.growable_pages() == el._pages_within(el.room - 1) - el.pages
    _no_leak(el)
    monkeypatch.setattr(el.pool, "create", create)
    el.settle(prefill_pending=False, decoding=False)
    assert el.taken == 0 and el.pool.num_handles == handles0
    _coherent(cache)
    _no_leak(el)

    # TP: the ranks vote, so they give back together or the server stops (never diverges)
    votes = []

    def all_reduce(t, op=None, group=None):
        votes.append(int(t[0]))
        t.fill_(min(votes[-1], peer_ok))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(el, "_tp_group", object())
    peer_ok = 1
    el.hold_prefill(2000)
    el.release_prefill()
    assert votes == [1] and el.taken == 0 and el.pool.num_handles == handles0
    peer_ok = 0  # this rank got its chunks back, another did not
    el.hold_prefill(2000)
    with pytest.raises(RuntimeError, match="a TP rank could not take"):
        el.release_prefill()
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



def _vram_config(pool, strategy="offload", decode_target="gpu", is_moe=True, flags=(), elastic_kv=None,
                 **settings):
    from types import SimpleNamespace

    model_config = SimpleNamespace(is_moe=is_moe, decode_target=decode_target, pool=pool)
    return SimpleNamespace(model_config=model_config, moe_strategy=strategy, static_vram_flags=flags,
                           elastic_kv=elastic_kv,
                           **{"prefill_batch_min": 2048, "prefill_batch_max": 16384, **settings})


@pytest.fixture
def _pool_from_config(monkeypatch):
    import freetoken.engine.engine as engine_mod

    monkeypatch.setattr(engine_mod, "resolve_pool_class", lambda model_config: model_config.pool)


def test_elastic_vram_is_on_exactly_where_it_is_supported(_pool_from_config):
    from freetoken.engine.engine import _elastic_vram
    from freetoken.kvcache.bsa_pool import BSAKVCache
    from freetoken.kvcache.dsv4_paged_pool import DSV4PagedKVCache
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
    from freetoken.kvcache.mha_pool import MHAKVCache
    from freetoken.kvcache.qsa_pool import QSAKVCache

    assert _elastic_vram(_vram_config(QSAKVCache))
    assert _elastic_vram(_vram_config(MHAKVCache))
    for pool in (BSAKVCache, HybridSWAKVCache, DSV4PagedKVCache):
        assert not _elastic_vram(_vram_config(pool))
    assert not _elastic_vram(_vram_config(QSAKVCache, strategy="hybrid", decode_target="hybrid"))
    assert not _elastic_vram(_vram_config(QSAKVCache, strategy="cpu", decode_target="cpu"))
    assert not _elastic_vram(_vram_config(QSAKVCache, decode_target="cpu"))  # --moe-cpu-layers
    assert not _elastic_vram(_vram_config(QSAKVCache, strategy="fused"))
    assert not _elastic_vram(_vram_config(MHAKVCache, strategy="fused", is_moe=False))


def test_static_sizing_flags_need_the_opt_out_where_elastic_vram_is_on(_pool_from_config, monkeypatch):
    import freetoken.engine.engine as engine_mod
    from freetoken.kvcache.qsa_pool import QSAKVCache

    check, elastic = engine_mod._check_vram_flags, engine_mod._elastic_vram
    for forced in (None, True):
        with pytest.raises(ValueError, match="size VRAM statically; elastic VRAM is on by default for this "
                                              "setup; pass --no-elastic-kv to use them"):
            check(_vram_config(QSAKVCache, flags=("--memory-ratio", "--moe-cache-size"), elastic_kv=forced))
    # the opt-out: the static flags, or upstream's static defaults without any
    for flags in (("--memory-ratio",), ()):
        config = _vram_config(QSAKVCache, flags=flags, elastic_kv=False)
        check(config)
        assert not elastic(config)
    check(_vram_config(QSAKVCache, elastic_kv=True))
    assert elastic(_vram_config(QSAKVCache, elastic_kv=True))


def test_elastic_kv_where_unsupported(_pool_from_config, monkeypatch):
    import freetoken.engine.engine as engine_mod
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
    from freetoken.kvcache.qsa_pool import QSAKVCache

    unsupported = (
        dict(pool=QSAKVCache, strategy="cpu", decode_target="cpu"),
        dict(pool=QSAKVCache, strategy="hybrid", decode_target="hybrid"),
        dict(pool=QSAKVCache, decode_target="cpu"),
        dict(pool=HybridSWAKVCache),
        dict(pool=QSAKVCache, strategy="fused", is_moe=False),
    )
    for setup in unsupported:
        with pytest.raises(ValueError, match="--elastic-kv: elastic VRAM needs"):
            engine_mod._check_vram_flags(_vram_config(**setup, elastic_kv=True))
        # the static flags work as upstream, with or without the opt-out
        for forced in (None, False):
            engine_mod._check_vram_flags(_vram_config(**setup, flags=("--memory-ratio",), elastic_kv=forced))

    warned = []
    monkeypatch.setattr(engine_mod.logger, "warning_rank0", lambda msg: warned.append(msg))
    engine_mod._resolve_prefill_batch(_vram_config(**unsupported[0], prefill_batch_min=4096))
    engine_mod._resolve_prefill_batch(_vram_config(QSAKVCache, elastic_kv=False, prefill_batch_max=8192))
    engine_mod._resolve_prefill_batch(_vram_config(QSAKVCache, elastic_kv=False))  # defaults: silent
    assert len(warned) == 2 and all("--prefill-batch-min/-max ignored" in w for w in warned)


def test_elastic_vram_prefills_up_to_the_batch_max_and_upstream_keeps_its_chunk(_pool_from_config):
    from freetoken.engine.engine import _resolve_prefill_batch
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache
    from freetoken.kvcache.qsa_pool import QSAKVCache

    config = _vram_config(QSAKVCache, max_extend_tokens=8192)
    _resolve_prefill_batch(config)
    assert (config.max_extend_tokens, config.prefill_batch_min) == (16384, 2048)
    config = _vram_config(QSAKVCache, max_extend_tokens=8192, prefill_batch_min=4096, prefill_batch_max=1024)
    _resolve_prefill_batch(config)  # the LLM() path reaches here unvalidated
    assert (config.max_extend_tokens, config.prefill_batch_min) == (1024, 1024)
    config = _vram_config(HybridSWAKVCache, max_extend_tokens=8192)
    _resolve_prefill_batch(config)
    assert config.max_extend_tokens == 8192


def test_llm_records_its_static_sizing_kwargs(monkeypatch):
    from freetoken.llm import llm as llm_mod

    seen = []
    monkeypatch.setattr(llm_mod.Scheduler, "__init__", lambda self, config: seen.append(config))
    llm_mod.LLM("/models/anon", memory_ratio=0.8, moe_cache_size=4096, elastic_kv=False)
    llm_mod.LLM("/models/anon", max_running_req=1)
    assert seen[0].static_vram_flags == ("memory_ratio", "moe_cache_size") and seen[0].elastic_kv is False
    assert seen[1].static_vram_flags == () and seen[1].elastic_kv is None


@pytest.mark.parametrize(
    ("argv", "want"),
    [
        ([], ()),
        (["--memory-ratio", "0.9"], ("--memory-ratio",)),  # the default value, still a choice
        (["--memory-ratio=0.8", "--kv-reserve-tokens", "4096"], ("--memory-ratio", "--kv-reserve-tokens")),
        (["--gpu-memory-ratio", "0:0.8"], ("--gpu-memory-ratio",)),
        (["--moe-cache-rate", "0.5"], ("--moe-cache-rate",)),
        (["--moe-cache-auto"], ("--moe-cache-auto",)),
        (["--num-tokens", "65536"], ("--num-tokens",)),
        (["--vram-keep-free-mb", "512"], ()),
        (["--no-elastic-kv", "--moe-cache-size", "4096"], ("--moe-cache-size",)),
    ],
)
def test_parse_args_records_which_static_sizing_flags_were_given(argv, want):
    from unittest.mock import patch

    from freetoken.engine.config import EngineConfig
    from freetoken.server.args import parse_args

    class _HF:
        def to_dict(self):
            return {"architectures": ["Qwen3_5MoeForConditionalGeneration"], "torch_dtype": "bfloat16"}

    with patch("freetoken.utils.cached_load_hf_config", lambda _path: _HF()):
        args = parse_args(["--model", "/models/anon", *argv])[0]
    assert args.static_vram_flags == want
    assert args.elastic_kv is (False if "--no-elastic-kv" in argv else None)
    if "--memory-ratio" not in " ".join(argv) and "--gpu-memory-ratio" not in argv:
        assert args.memory_ratio == EngineConfig.memory_ratio
    if not want:
        assert args.moe_cache_size == 0 and args.moe_cache_rate is None


@pytest.mark.parametrize(
    ("argv", "want"),
    [
        ([], (8192, 2048, 16384)),
        (["--prefill-batch-min", "4096", "--prefill-batch-max", "8192"], (8192, 4096, 8192)),
        (["--max-prefill-length", "4096"], (4096, 2048, 4096)),  # an explicit cap caps elastic's chunks
        (["--max-prefill-length", "1024"], (1024, 1024, 1024)),
        (["--max-prefill-length", "32768"], (32768, 2048, 16384)),
    ],
)
def test_parse_args_caps_the_prefill_batch_by_an_explicit_max_prefill_length(argv, want):
    from unittest.mock import patch

    from freetoken.server.args import parse_args

    class _HF:
        def to_dict(self):
            return {"architectures": ["Qwen3_5MoeForConditionalGeneration"], "torch_dtype": "bfloat16"}

    with patch("freetoken.utils.cached_load_hf_config", lambda _path: _HF()):
        args = parse_args(["--model", "/models/anon", *argv])[0]
    assert (args.max_extend_tokens, args.prefill_batch_min, args.prefill_batch_max) == want

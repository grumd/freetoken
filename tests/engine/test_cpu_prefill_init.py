"""Engine._maybe_init_wna16_cpu_prefill: the hybrid CPU prefill is on by default wherever the CPU kernel
runs, --moe-prefill-fetch sets its fetched share (1 = off), -t/--threads its thread count."""

from types import SimpleNamespace

import pytest
import torch

import freetoken.layers.moe as moe_mod
from freetoken.engine import engine as engine_mod
from freetoken.engine.engine import Engine
from freetoken.moe import wna16_cpu_prefill as wcp

E, L, H, I, TOP_K = 8, 2, 256, 128, 2


class _Executor:
    BANK_ROLES = wcp.Nvfp4CpuPrefill.BANK_ROLES
    built = []

    def __init__(self, **kwargs):
        self.kwargs, self.layers = kwargs, {}
        _Executor.built.append(self)

    def set_layer(self, layer_id, banks):
        self.layers[layer_id] = sorted(banks)


def _setup(monkeypatch, fetch=0.35, threads=2, activation="silu", avx512_bf16=True):
    _Executor.built = []
    monkeypatch.setattr(wcp, "Nvfp4CpuPrefill", _Executor)
    monkeypatch.setattr(wcp, "cpu_supported", lambda: avx512_bf16)
    monkeypatch.setattr(moe_mod, "_ROUTED_PREFILL_TOKENS", 4096)
    shapes = {"gate_up": (E, 2 * I, H // 2), "gate_up_scale": (E, 2 * I, H // 16), "gate_up_global": (E,),
              "down": (E, H, I // 2), "down_scale": (E, H, I // 16), "down_global": (E,)}
    cache = SimpleNamespace(
        bank_sources={r: [torch.empty(s, dtype=torch.uint8) for _ in range(L)] for r, s in shapes.items()},
        decode_target="gpu", cpu_layer_ids=frozenset(), verify_cpu=False, num_experts=E, num_layers=L,
        cpu_prefill=None, cpu_prefill_fetch=0.35,
    )
    layer = SimpleNamespace(num_experts=E, top_k=TOP_K, activation=activation, apply_router_weight_on_input=False,
                            interleaved=False, has_bias=False)
    eng = Engine.__new__(Engine)
    eng.config = SimpleNamespace(moe_prefill_fetch=fetch, cpu_threads=threads, tp_info=SimpleNamespace(rank=0, size=1))
    eng._maybe_init_wna16_cpu_prefill(cache, [layer] * L)
    return cache


def test_on_by_default_with_buffers_for_the_routed_cutoff(monkeypatch):
    cache = _setup(monkeypatch)
    (ex,) = _Executor.built
    assert cache.cpu_prefill is ex and cache.cpu_prefill_fetch == 0.35
    assert ex.kwargs["max_tokens"] == 4096 and ex.kwargs["threads"] == 2
    assert (ex.kwargs["hidden"], ex.kwargs["inter"]) == (H, I)
    assert sorted(ex.layers) == list(range(L))


def test_the_flag_sets_the_fetched_share(monkeypatch):
    assert _setup(monkeypatch, fetch=0.2).cpu_prefill_fetch == 0.2
    assert _setup(monkeypatch, fetch=0.0).cpu_prefill_fetch == 0.0


@pytest.mark.parametrize("kwargs", [{"fetch": 1.0}, {"activation": "gelu"}, {"avx512_bf16": False}])
def test_off(monkeypatch, kwargs):
    assert _setup(monkeypatch, **kwargs).cpu_prefill is None
    assert not _Executor.built


@pytest.mark.parametrize("threads,tp,want", [(0, 1, 8), (0, 2, 4), (6, 2, 3), (1, 2, 1)])
def test_threads_split_across_tp_ranks(monkeypatch, threads, tp, want):
    import freetoken.moe.cpu_executor as cpu_executor

    monkeypatch.setattr(cpu_executor, "physical_core_cpus", lambda: list(range(8)))
    config = SimpleNamespace(cpu_threads=threads, tp_info=SimpleNamespace(rank=0, size=tp))
    assert engine_mod._rank_cpu_threads(config) == want

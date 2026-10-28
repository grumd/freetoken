"""Engine.tune_cpu_decode, the runtime control calibrate-qwen38.sh drives: only on a server
started for it, only shares below 1 (the decode path is fixed at startup), threads in range."""

import pytest
import torch

from freetoken.engine.engine import CpuDecodeTuneRejected, Engine
from freetoken.moe.offload_cache import OffloadMoeCache


class _Executor:
    max_threads = 8
    isa = "avx512f"

    def __init__(self):
        self.active_threads = 7

    def set_active_threads(self, n):
        self.active_threads = n


def _engine(monkeypatch, tunable=True, split=True):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    cache = OffloadMoeCache(num_layers=2, num_experts=8, cache_size=16, device=torch.device("cpu"),
                            quant_format="nvfp4")
    if split:
        cache.cpu_decode = _Executor()
        cache.set_decode_fetch(0.5)
    eng = Engine.__new__(Engine)
    eng.moe_offload_cache = cache
    eng.device = torch.device("cpu")
    eng._cpu_decode_tunable = tunable
    return eng, cache


def test_tune_changes_share_and_threads(monkeypatch):
    eng, cache = _engine(monkeypatch)
    assert eng.tune_cpu_decode() == {"decode_fetch": 0.5, "threads": 7, "max_threads": 8, "isa": "avx512f"}
    state = eng.tune_cpu_decode(0.1, 4)
    assert state["decode_fetch"] == 0.1 and state["threads"] == 4
    assert int(cache.decode_fetch_q16) == round(0.1 * 65536)
    assert eng.tune_cpu_decode(threads=1)["threads"] == 1 and cache.decode_fetch == 0.1


@pytest.mark.parametrize("kwargs", [{"fetch": 1.0}, {"fetch": -0.1}, {"threads": 0}, {"threads": 9}])
def test_tune_rejects_out_of_range(monkeypatch, kwargs):
    eng, cache = _engine(monkeypatch)
    with pytest.raises(CpuDecodeTuneRejected):
        eng.tune_cpu_decode(**kwargs)
    assert cache.decode_fetch == 0.5 and cache.cpu_decode.active_threads == 7


@pytest.mark.parametrize("tunable,split", [(False, True), (True, False)])
def test_tune_needs_a_tunable_split_server(monkeypatch, tunable, split):
    eng, _ = _engine(monkeypatch, tunable=tunable, split=split)
    with pytest.raises(CpuDecodeTuneRejected, match="FREETOKEN_CPU_DECODE_TUNABLE"):
        eng.tune_cpu_decode(0.1)

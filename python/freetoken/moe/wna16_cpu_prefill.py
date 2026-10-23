"""CPU side of the hybrid W4A16 / NVFP4 routed prefill.

A routed prefill chunk of a few hundred tokens touches most of a layer's experts, and on an
offload setup most of those are VRAM-cache misses whose bytes must cross PCIe before the GPU can
use them. This executor computes a share of the misses on the CPU, straight from the pinned host
banks, while the GPU fetches and computes the rest (``OffloadMoELayer._prefill_cpu_hybrid``).
Kernels: ``kernel/csrc/cpu_moe/wna16_cpu.h`` / ``nvfp4_cpu.h`` (AVX-512 BF16).
"""

from __future__ import annotations

import functools
import os
import time
from pathlib import Path

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)
_now = time.perf_counter

_SRC = Path(__file__).resolve().parents[1] / "kernel" / "csrc" / "cpu_moe"
_REQUIRED_FLAGS = ("avx512f", "avx512bw", "avx512_bf16")


@functools.cache
def cpu_supported() -> bool:
    try:
        with open("/proc/cpuinfo") as f:
            flags = next((line.split(":", 1)[1].split() for line in f if line.startswith("flags")), [])
    except OSError:
        return False
    return all(flag in flags for flag in _REQUIRED_FLAGS)


@functools.cache
def load_extension():
    from torch.utils.cpp_extension import load

    return load(
        name="freetoken_wna16_cpu_prefill",
        sources=[str(_SRC / "wna16_prefill_ext.cpp")],
        extra_include_paths=[str(_SRC)],
        extra_cflags=["-O3", "-march=native", "-std=c++17", "-pthread"],
        verbose=False,
    )


@functools.cache
def load_nvfp4_extension():
    from torch.utils.cpp_extension import load

    return load(
        name="freetoken_nvfp4_cpu_prefill",
        sources=[str(_SRC / "nvfp4_prefill_ext.cpp")],
        extra_include_paths=[str(_SRC)],
        extra_cflags=["-O3", "-march=native", "-std=c++17", "-pthread"],
        verbose=False,
    )


def rank_cpus(threads: int, rank: int, world: int) -> list[int]:
    """This rank's share of the physical cores (one logical CPU each), so the ranks' pools
    never contend for a core."""
    from freetoken.moe.cpu_executor import physical_core_cpus

    cores = physical_core_cpus()
    per = max(1, len(cores) // max(1, world))
    mine = cores[rank * per:(rank + 1) * per] or cores
    return [mine[i % len(mine)] for i in range(threads)]


class Wna16CpuPrefill:
    """Thin owner of the C++ executor plus the pinned staging buffers for one rank."""

    def __init__(self, *, hidden: int, inter: int, down_groups: int, k_off: int, num_experts: int,
                 top_k: int, max_tokens: int, threads: int, cpus: list[int]) -> None:
        ext = load_extension()
        self._init_buffers(ext.Wna16CpuPrefill(threads, cpus, hidden, inter, down_groups, k_off, num_experts),
                           hidden, top_k, max_tokens)
        logger.info(f"wna16 CPU prefill experts: {threads} threads on cpus {cpus}")

    def _init_buffers(self, impl, hidden: int, top_k: int, max_tokens: int) -> None:
        self.impl = impl
        self.hidden, self.top_k, self.max_tokens = hidden, top_k, max_tokens
        pin = torch.cuda.is_available()
        self.x = torch.empty((max_tokens, hidden), dtype=torch.bfloat16, pin_memory=pin)
        # fp32: the caller adds this partial to the GPU routes before rounding once
        self.y = torch.empty((max_tokens, hidden), dtype=torch.float32, pin_memory=pin)
        self.ids = torch.empty((max_tokens, top_k), dtype=torch.int32, pin_memory=pin)
        self.w = torch.empty((max_tokens, top_k), dtype=torch.float32, pin_memory=pin)
        self.meta = torch.empty((4,), dtype=torch.int64, pin_memory=pin)
        # FREETOKEN_WNA16_CPU_PREFILL_STATS=1: one log line per prefill forward
        self.stats = {} if os.environ.get("FREETOKEN_WNA16_CPU_PREFILL_STATS") == "1" else None

    def record(self, layer_id: int, tokens: int, routed: int, fetched: int, cpu_experts: int,
               cpu_s: float, gpu_busy: bool) -> None:
        st = self.stats
        if layer_id == 0 or not st:
            st.clear()
            st.update(t0=_now(), layers=0, routed=0, fetched=0, cpu=0, cpu_s=0.0, gpu_busy=0)
        st["layers"] += 1
        st["routed"] += routed
        st["fetched"] += fetched
        st["cpu"] += cpu_experts
        st["cpu_s"] += cpu_s
        st["gpu_busy"] += int(gpu_busy)
        if layer_id == self.last_layer:
            n = st["layers"]
            logger.info(
                f"cpu prefill T={tokens}: per layer {st['routed'] / n:.0f} routed experts, "
                f"{st['fetched'] / n:.0f} fetched, {st['cpu'] / n:.0f} on CPU "
                f"({1e3 * st['cpu_s'] / n:.2f} ms); GPU still busy after the CPU in "
                f"{st['gpu_busy']}/{n} layers; {1e3 * (_now() - st['t0']):.0f} ms layer 0 -> last")

    def set_layer(self, layer_id: int, banks: dict[str, torch.Tensor]) -> None:
        self.last_layer = max(getattr(self, "last_layer", 0), layer_id)
        self.impl.set_layer(layer_id, banks["gate_up"], banks["gate_up_zero"], banks["gate_up_scale"],
                            banks["down"], banks["down_zero"], banks["down_scale"])

    def forward(self, layer_id: int, x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor,
                y: torch.Tensor) -> int:
        from freetoken.moe.fused_wna16 import prefill_precise

        return self.impl.forward(layer_id, x, ids, w, y, prefill_precise())


class Nvfp4CpuPrefill(Wna16CpuPrefill):
    """The NVFP4 executor (nvfp4_cpu.h) behind the same staging buffers and interface."""

    BANK_ROLES = ("gate_up", "gate_up_scale", "gate_up_global", "down", "down_scale", "down_global")

    def __init__(self, *, hidden: int, inter: int, num_experts: int, top_k: int, max_tokens: int,
                 threads: int, cpus: list[int]) -> None:
        ext = load_nvfp4_extension()
        self._init_buffers(ext.Nvfp4CpuPrefill(threads, cpus, hidden, inter, num_experts), hidden, top_k, max_tokens)
        logger.info(f"nvfp4 CPU prefill experts: {threads} threads on cpus {cpus}")

    def set_layer(self, layer_id: int, banks: dict[str, torch.Tensor]) -> None:
        self.last_layer = max(getattr(self, "last_layer", 0), layer_id)
        self.impl.set_layer(layer_id, *(banks[r] for r in self.BANK_ROLES))


def threads_from_env(default: int) -> int:
    raw = os.environ.get("FREETOKEN_WNA16_CPU_PREFILL_THREADS", "")
    return int(raw) if raw.strip() else default


__all__ = ["Nvfp4CpuPrefill", "Wna16CpuPrefill", "cpu_supported", "load_extension", "load_nvfp4_extension",
           "rank_cpus", "threads_from_env"]

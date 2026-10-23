"""Host side of the RTX 3080-tuned NVFP4 expert kernels (:mod:`freetoken.kernel.triton.nvfp4_tile_moe`).

Same calling convention as :mod:`freetoken.moe.fused_nvfp4` plus the routed-only prefill
``expert_map`` of :func:`freetoken.moe.fused_wna16.fused_experts_wna16`.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import torch
import triton

from freetoken.kernel import moe_sum_reduce_triton
from freetoken.kernel.triton.nvfp4_tile_moe import (
    _gemv_partials_kernel,
    _gemv_partials_v2_kernel,
    _prefill_nvfp4_gate_up_silu_kernel,
    _prefill_nvfp4_tile_kernel,
)
from freetoken.kernel.triton.wna16_decode import _reduce_silu_mul_kernel, _reduce_topk_sum_kernel
from freetoken.kernel.triton.wna16_fused_moe import _tl_dtype
from freetoken.layers import gated_act_and_mul
from freetoken.moe.fused import moe_align_block_size


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8)


_NK_TILE = os.environ.get("FREETOKEN_NVFP4_NK_TILE", "1") != "0"

# Swept with kbench/sweep_nvfp4_prefill.py on an RTX 3080 (q38 TP=2 band: gate_up 2560 x 640,
# down 320 x 2560, top-10 of 512): M=512 / 2048 / 8192 -> fused gate_up+SiLU 1.05 / 1.96 /
# 5.57 ms, down 0.58 / 1.08 / 3.16 ms (WNA16 tile kernels at the same shapes: 4.0 / 4.5 / 14.0
# ms for the pair).
_TILE_GATE_UP = dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128,
                     GROUP_SIZE_M=8, num_warps=4, num_stages=2)


def _tile_down_config(M: int) -> Dict[str, int]:
    return dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
                GROUP_SIZE_M=8, num_warps=4, num_stages=4)


def _tile_silu_config(M: int) -> Dict[str, int]:
    return dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128 if M < 4096 else 64,
                GROUP_SIZE_M=8, num_warps=4, num_stages=2)


def tile_ok(K: int, cfg: Dict[str, int]) -> bool:
    return K % cfg["BLOCK_SIZE_K"] == 0 and cfg["BLOCK_SIZE_K"] % 16 == 0


def _prefill_gemm(a, packed, scale, glob, c, tw, sorted_ids, expert_ids, ntpp, num_valid,
                  kernel_top_k, mul_routed_weight, cfg, nk_tile=None) -> None:
    N, K = packed.shape[1], packed.shape[2] * 2
    EM = sorted_ids.shape[0]
    scale = _u8(scale)
    grid = (triton.cdiv(EM, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),)
    _prefill_nvfp4_tile_kernel[grid](
        a, packed, scale, glob, c, tw, sorted_ids, expert_ids, ntpp,
        N, K, EM, num_valid,
        a.stride(0), a.stride(1),
        packed.stride(0), packed.stride(1),
        scale.stride(0), scale.stride(1),
        glob.stride(0),
        c.stride(1), c.stride(2),
        tw.stride(0),
        MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=kernel_top_k,
        compute_type=_tl_dtype(c.dtype), NK_TILE=_NK_TILE if nk_tile is None else nk_tile, **cfg,
    )


def _prefill_gate_up_silu(a, packed, scale, glob, c, sorted_ids, expert_ids, ntpp, num_valid,
                          top_k, cfg=None, nk_tile=None) -> None:
    cfg = cfg or _tile_silu_config(sorted_ids.shape[0] // max(1, top_k))
    inter, K = packed.shape[1] // 2, packed.shape[2] * 2
    EM = sorted_ids.shape[0]
    scale = _u8(scale)
    grid = (triton.cdiv(EM, cfg["BLOCK_SIZE_M"]) * triton.cdiv(inter, cfg["BLOCK_SIZE_N"]),)
    _prefill_nvfp4_gate_up_silu_kernel[grid](
        a, packed, scale, glob, c, sorted_ids, expert_ids, ntpp,
        inter, K, EM, num_valid,
        a.stride(0), a.stride(1),
        packed.stride(0), packed.stride(1),
        scale.stride(0), scale.stride(1),
        glob.stride(0),
        c.stride(0), c.stride(1),
        top_k=top_k, compute_type=_tl_dtype(c.dtype),
        NK_TILE=_NK_TILE if nk_tile is None else nk_tile, **cfg,
    )


def fused_experts_nvfp4_tile(
    hidden_states: torch.Tensor,
    gate_up_packed: torch.Tensor, gate_up_scale: torch.Tensor, gate_up_global: torch.Tensor,
    down_packed: torch.Tensor, down_scale: torch.Tensor, down_global: torch.Tensor,
    topk_weights: torch.Tensor, topk_ids: torch.Tensor, num_experts: int,
    activation: str = "silu", apply_router_weight_on_input: bool = False,
    act_alpha: float = 1.702, act_limit: float = 7.0,
    expert_map: torch.Tensor | None = None, reduce: bool = True,
    cfg_gu: Dict[str, int] | None = None, cfg_dn: Dict[str, int] | None = None,
    fuse_act: bool | None = None,
) -> torch.Tensor:
    """Grouped NVFP4 prefill. ``expert_map`` (routed-only prefill over the slot cache):
    ``topk_ids`` are compact ids ``[0, num_experts)`` and ``expert_map[id]`` is the bank row
    each reads. ``reduce=False`` returns the per-route outputs ``[M, top_k, H]``."""
    M, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    two_i = gate_up_packed.shape[1]
    inter = two_i // 2
    dev, dt = hidden_states.device, hidden_states.dtype
    cfg_dn = cfg_dn or _tile_down_config(M)
    if fuse_act is None:
        fuse_act = activation == "silu" and not apply_router_weight_on_input
    cfg_gu = cfg_gu or (_tile_silu_config(M) if fuse_act else _TILE_GATE_UP)

    aligned: Dict[int, Any] = {}

    def align(block_m: int):
        if block_m not in aligned:
            a = moe_align_block_size(topk_ids, block_m, num_experts)
            if expert_map is not None:
                a = (a[0], expert_map[a[1].clamp(0, num_experts - 1)].to(torch.int32), a[2])
            aligned[block_m] = a
        return aligned[block_m]

    tw = topk_weights.reshape(-1).contiguous()
    num_valid = topk_ids.numel()
    ic2 = torch.empty((M * top_k, inter), device=dev, dtype=dt)
    if fuse_act:
        _prefill_gate_up_silu(hidden_states, gate_up_packed, gate_up_scale, gate_up_global, ic2,
                              *align(cfg_gu["BLOCK_SIZE_M"]), num_valid, top_k, cfg_gu)
    else:
        ic1 = torch.empty((M, top_k, two_i), device=dev, dtype=dt)
        _prefill_gemm(hidden_states, gate_up_packed, gate_up_scale, gate_up_global, ic1, tw,
                      *align(cfg_gu["BLOCK_SIZE_M"]), num_valid, top_k, apply_router_weight_on_input, cfg_gu)
        gated_act_and_mul(activation, ic1.view(-1, two_i), ic2, alpha=act_alpha, limit=act_limit)
    ic3 = torch.empty((M, top_k, H), device=dev, dtype=dt)
    _prefill_gemm(ic2, down_packed, down_scale, down_global, ic3, tw,
                  *align(cfg_dn["BLOCK_SIZE_M"]), num_valid, 1, not apply_router_weight_on_input, cfg_dn)
    if not reduce:
        return ic3
    out = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(ic3, out)
    return out


# (BLOCK_N, BLOCK_KW, num_warps) and the split-K program target, swept on an RTX 3080 at the
# q38 TP=2 shapes (gate_up 640 x 2560, down 2560 x 320).
_GU_TILE = (64, 16, 2)
_DN_TILE = (64, 8, 4)
_TARGET_PROGRAMS = 544
_GEMV_V2 = os.environ.get("FREETOKEN_NVFP4_GEMV_V2", "1") != "0"


def _gemv(a, packed, scale, glob, ids, R, top_k, a_row_is_route, tile):
    bn, bkw, warps = tile
    N, kw = packed.shape[1], packed.shape[2] // 4
    words = packed.view(torch.int32)
    scale = _u8(scale)
    n_kb = kw // bkw
    split = max(1, min(n_kb, _TARGET_PROGRAMS // max(1, R * triton.cdiv(N, bn))))
    kb_per = triton.cdiv(n_kb, split)
    split = triton.cdiv(n_kb, kb_per)
    part = torch.empty((split, R, N), dtype=torch.float32, device=a.device)
    kernel = _gemv_partials_v2_kernel if _GEMV_V2 else _gemv_partials_kernel
    kernel[(R, triton.cdiv(N, bn), split)](
        a, words, scale, glob, part, ids,
        R, N, kb_per * bkw, kw,
        a.stride(0),
        words.stride(0), words.stride(1),
        scale.stride(0), scale.stride(1),
        glob.stride(0),
        TOP_K=top_k, A_ROW_IS_ROUTE=a_row_is_route,
        BLOCK_N=bn, BLOCK_KW=bkw, num_warps=warps,
    )
    return part, split


def decode_supported(gate_up_packed: torch.Tensor, down_packed: torch.Tensor, gu_tile=None, dn_tile=None) -> bool:
    gu, dn = gu_tile or _GU_TILE, dn_tile or _DN_TILE
    return (gate_up_packed.shape[2] // 4) % gu[1] == 0 and (down_packed.shape[2] // 4) % dn[1] == 0


def nvfp4_decode_silu(
    x: torch.Tensor,
    gate_up_packed: torch.Tensor, gate_up_scale: torch.Tensor, gate_up_global: torch.Tensor,
    down_packed: torch.Tensor, down_scale: torch.Tensor, down_global: torch.Tensor,
    topk_weights: torch.Tensor, topk_ids: torch.Tensor, gu_tile=None, dn_tile=None,
) -> torch.Tensor:
    """SiLU-gated NVFP4 expert MLP for a few tokens; ``topk_ids`` are cache slots, ``x`` [M, H]."""
    M, H = x.shape
    top_k = topk_ids.shape[1]
    R = M * top_k
    inter = gate_up_packed.shape[1] // 2
    ids = topk_ids.reshape(-1)
    part1, s1 = _gemv(x, gate_up_packed, gate_up_scale, gate_up_global, ids, R, top_k, False, gu_tile or _GU_TILE)
    act = torch.empty((R, inter), dtype=x.dtype, device=x.device)
    _reduce_silu_mul_kernel[(R, triton.cdiv(inter, 256))](part1, act, R, inter, SPLIT=s1, BLOCK=256, num_warps=2)
    part2, s2 = _gemv(act, down_packed, down_scale, down_global, ids, R, top_k, True, dn_tile or _DN_TILE)
    out = torch.empty((M, H), dtype=x.dtype, device=x.device)
    _reduce_topk_sum_kernel[(M, triton.cdiv(H, 512))](
        part2, topk_weights, out, R, H, topk_weights.stride(0),
        TOP_K=top_k, SPLIT=s2, BLOCK=512, num_warps=4,
    )
    return out


__all__ = ["decode_supported", "fused_experts_nvfp4_tile", "nvfp4_decode_silu", "tile_ok"]

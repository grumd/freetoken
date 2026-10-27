# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM (vllm/models/qwen4_exp/nvidia/ops/hc.py)
"""NVIDIA HyperConnection kernels for Qwen4Exp."""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from freetoken.utils.arch import is_sm90_supported


@functools.cache
def _pdl_supported() -> bool:
    return is_sm90_supported()


@triton.jit
def _grouped_gemma_rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    stride_x,
    stride_y,
    DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    GROUP_DIM: tl.constexpr = DIM // NUM_GROUPS
    BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(GROUP_DIM)

    pid = tl.program_id(0)
    group_id = pid % NUM_GROUPS
    # row * stride can overflow int32 for large token counts.
    row = (pid // NUM_GROUPS).to(tl.int64)

    offs_g = tl.arange(0, BLOCK_SIZE)
    offsets = group_id * GROUP_DIM + offs_g
    mask = offs_g < GROUP_DIM
    # A [GROUP_DIM] affine is shared; a [DIM] affine follows the grouped
    # checkpoint layout.
    w_offs = offs_g if W_SHARED else offsets

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    x = tl.load(x_ptr + row * stride_x + offsets, mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + w_offs, mask, other=0.0)

    rrms = tl.rsqrt(tl.sum(x * x) / GROUP_DIM + EPS)
    # Gemma's (1 + w) affine is written this way to lower to an FMA.
    y = x * rrms
    y += y * w.to(tl.float32)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(y_ptr + row * stride_y + offsets, y, mask)


def grouped_gemma_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, num_groups: int
) -> torch.Tensor:
    N, DIM = x.shape
    assert x.stride(1) == 1, "grouped Gemma RMSNorm requires unit inner stride"
    assert weight.is_contiguous(), "grouped Gemma RMSNorm weight must be contiguous"
    assert DIM % num_groups == 0
    group_dim = DIM // num_groups
    assert weight.numel() in (group_dim, DIM)

    y = x.new_empty(x.shape)
    _grouped_gemma_rmsnorm_kernel[(N * num_groups,)](
        x,
        weight,
        y,
        x.stride(0),
        y.stride(0),
        DIM,
        num_groups,
        W_SHARED=weight.numel() == group_dim,
        EPS=eps,
        launch_pdl=_pdl_supported(),
    )
    return y


@triton.jit
def _hc_silu_kernel(
    x_ptr,
    y_ptr,
    stride_x,
    stride_y,
    DIM: tl.constexpr,
    HC: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(DIM)

    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < DIM

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    x = tl.load(x_ptr + row * stride_x + offs, mask).to(tl.float32) / HC
    y = x * tl.sigmoid(x)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(y_ptr + row * stride_y + offs, y, mask)


def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    num_tokens, DIM = x.shape
    assert x.stride(1) == 1

    output = x.new_empty(x.shape)
    _hc_silu_kernel[(num_tokens,)](
        x,
        output,
        x.stride(0),
        output.stride(0),
        DIM=DIM,
        HC=hc_count,
        launch_pdl=_pdl_supported(),
    )
    return output


@triton.jit
def _hc_gate_mix_kernel(
    x_ptr,
    g_ptr,
    y_ptr,
    stride_x,
    stride_g,
    stride_y,
    DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    HC_DIM: tl.constexpr = DIM // HC

    row = tl.program_id(0).to(tl.int64)
    tile_id = tl.program_id(1)
    offs_inner = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_inner < HC_DIM

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    # The constexpr loop is unrolled and keeps one stream live at a time.
    # Materializing [HC, BLOCK_SIZE] more than doubles latency at large M.
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for stream in tl.static_range(HC):
        offsets = stream * HC_DIM + offs_inner
        g = tl.load(g_ptr + row * stride_g + offsets, mask, other=0.0)
        x = tl.load(x_ptr + row * stride_x + offsets, mask, other=0.0)
        acc += tl.sigmoid(g.to(tl.float32)) * x.to(tl.float32)
    acc /= HC

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(y_ptr + row * stride_y + offs_inner, acc, mask)


def hc_gate_mix(x: torch.Tensor, gate: torch.Tensor, hc_count: int) -> torch.Tensor:
    N, DIM = gate.shape
    assert x.shape == gate.shape
    assert DIM % hc_count == 0
    assert x.stride(1) == 1
    assert gate.stride(1) == 1

    HC_DIM = DIM // hc_count
    out = x.new_empty(N, HC_DIM)
    BLOCK_SIZE = 512
    _hc_gate_mix_kernel[(N, triton.cdiv(HC_DIM, BLOCK_SIZE))](
        x,
        gate,
        out,
        x.stride(0),
        gate.stride(0),
        out.stride(0),
        DIM,
        hc_count,
        BLOCK_SIZE,
        launch_pdl=_pdl_supported(),
    )
    return out


@triton.jit
def _hc_combine_kernel(
    block_ptr,
    res_ptr,
    inj_ptr,
    out_ptr,
    stride_block,
    stride_res,
    stride_inj,
    stride_out,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    HC_PAD: tl.constexpr = triton.next_power_of_2(HC)

    row = tl.program_id(0).to(tl.int64)
    tile_id = tl.program_id(1)

    offs_inner = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_inner = offs_inner < HC_DIM
    offs_hc = tl.arange(0, HC_PAD)
    mask_hc = offs_hc < HC
    offs = offs_hc[:, None] * HC_DIM + offs_inner[None, :]
    mask = mask_hc[:, None] & mask_inner[None, :]

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    inj = tl.load(inj_ptr + row * stride_inj + offs_hc, mask_hc, other=0.0)
    block = tl.load(block_ptr + row * stride_block + offs_inner, mask_inner, other=0.0)
    res = tl.load(res_ptr + row * stride_res + offs, mask, other=0.0)

    # Keeping HC as a broadcast dimension is faster here than four separate
    # residual load/store sequences.
    inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)
    out = res.to(tl.float32) + block.to(tl.float32)[None, :] * inj[:, None]

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(out_ptr + row * stride_out + offs, out, mask=mask)


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    N, DIM = residual.shape
    assert DIM % hc_count == 0
    hc_dim = DIM // hc_count
    assert block_output.shape == (N, hc_dim)
    assert injection_logits.shape == (N, hc_count)
    assert residual.stride(1) == 1
    assert block_output.stride(1) == 1
    assert injection_logits.stride(1) == 1

    out = residual.new_empty(residual.shape)
    BLOCK_SIZE = 512
    _hc_combine_kernel[(N, triton.cdiv(hc_dim, BLOCK_SIZE))](
        block_output,
        residual,
        injection_logits,
        out,
        block_output.stride(0),
        residual.stride(0),
        injection_logits.stride(0),
        out.stride(0),
        hc_dim,
        hc_count,
        BLOCK_SIZE,
        launch_pdl=_pdl_supported(),
    )
    return out


@triton.jit
def _hc_combine_norm_kernel(
    block_ptr,
    res_ptr,
    inj_ptr,
    w_ptr,
    out_ptr,
    y_ptr,
    stride_block,
    stride_res,
    stride_inj,
    stride_out,
    stride_y,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    HC_PAD: tl.constexpr = triton.next_power_of_2(HC)
    NUM_TILES: tl.constexpr = triton.cdiv(HC_DIM, BLOCK_SIZE)
    NUM_TILES_PAD: tl.constexpr = triton.next_power_of_2(NUM_TILES)

    row = tl.program_id(0).to(tl.int64)
    stream = tl.program_id(1)
    offs_hc = tl.arange(0, HC_PAD)
    mask_hc = offs_hc < HC
    tile_ids = tl.arange(0, NUM_TILES_PAD)
    offs_inner = tile_ids[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    mask_inner = offs_inner < HC_DIM
    offs = stream * HC_DIM + offs_inner
    # Shared norm weights repeat across streams; per-branch weights use the
    # same flattened HC layout as the residual.
    w_offs = offs_inner if W_SHARED else offs

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    # Start the uncached residual load first, then issue the other combine
    # loads before consuming any of them.
    res = tl.load(res_ptr + row * stride_res + offs, mask_inner, other=0.0)
    inj = tl.load(inj_ptr + row * stride_inj + offs_hc, mask_hc, other=0.0)
    block = tl.load(block_ptr + row * stride_block + offs_inner, mask_inner, other=0.0)
    inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)
    inj = tl.sum(tl.where(offs_hc == stream, inj, 0.0))
    # Round the materialized combine result before normalization. This matches
    # the unfused combine -> RMSNorm boundary.
    out = (res.to(tl.float32) + block.to(tl.float32) * inj).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * stride_out + offs, out, mask=mask_inner)

    out = out.to(tl.float32)
    # Keep the two-axis reduction: flattening the padded tile is ~40% slower
    # at decode sizes.
    sum_sq = tl.sum(tl.sum(out * out, axis=1), axis=0)
    rrms = tl.rsqrt(sum_sq / HC_DIM + EPS)

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    # Loading the weight earlier helps decode but keeps the tile live across
    # the reduction and regresses larger batches, so defer it to the norm.
    w = tl.load(w_ptr + w_offs, mask_inner, other=0.0)
    y = out * rrms
    y += y * w.to(tl.float32)
    tl.store(y_ptr + row * stride_y + offs, y, mask_inner)


def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    N, DIM = residual.shape
    assert DIM % hc_count == 0
    hc_dim = DIM // hc_count
    assert block_output.shape == (N, hc_dim)
    assert injection_logits.shape == (N, hc_count)
    assert residual.stride(1) == 1
    assert block_output.stride(1) == 1
    assert injection_logits.stride(1) == 1
    assert norm_weight.is_contiguous()
    assert norm_weight.numel() in (hc_dim, DIM)

    out = residual.new_empty(residual.shape)
    y = residual.new_empty(residual.shape)
    BLOCK_SIZE = 512
    _hc_combine_norm_kernel[(N, hc_count)](
        block_output,
        residual,
        injection_logits,
        norm_weight,
        out,
        y,
        block_output.stride(0),
        residual.stride(0),
        injection_logits.stride(0),
        out.stride(0),
        y.stride(0),
        hc_dim,
        hc_count,
        W_SHARED=norm_weight.numel() == hc_dim,
        EPS=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        launch_pdl=_pdl_supported(),
    )
    return out, y


@triton.jit
def _hc_combine_rmsnorm_kernel(
    block_ptr,
    res_ptr,
    inj_ptr,
    w_ptr,
    out_ptr,
    y_ptr,
    stride_block,
    stride_res,
    stride_inj,
    stride_out,
    stride_y,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    EPS: tl.constexpr,
) -> None:
    # One stream per program over the same [next_pow2(HC_DIM)] tile and sum as
    # _grouped_gemma_rmsnorm_kernel, so both outputs match the unfused pair bit for bit
    # (hc_combine_norm's 2-D tile reduces in another order).
    BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(HC_DIM)
    row = tl.program_id(0).to(tl.int64)
    stream = tl.program_id(1)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < HC_DIM
    offsets = stream * HC_DIM + offs

    inj = 2.0 * tl.sigmoid(tl.load(inj_ptr + row * stride_inj + stream).to(tl.float32) / HC)
    block = tl.load(block_ptr + row * stride_block + offs, mask, other=0.0)
    res = tl.load(res_ptr + row * stride_res + offsets, mask, other=0.0)
    out = (res.to(tl.float32) + block.to(tl.float32) * inj).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * stride_out + offsets, out, mask)

    x = out.to(tl.float32)
    w = tl.load(w_ptr + offsets, mask, other=0.0)
    rrms = tl.rsqrt(tl.sum(x * x) / HC_DIM + EPS)
    y = x * rrms
    y += y * w.to(tl.float32)
    tl.store(y_ptr + row * stride_y + offsets, y, mask)


def hc_combine_rmsnorm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``hc_combine`` and ``grouped_gemma_rmsnorm`` of its result in one launch, bit-identical
    to the two kernels: ``(residual', normed residual')``."""
    N, DIM = residual.shape
    assert DIM % hc_count == 0
    hc_dim = DIM // hc_count
    assert block_output.shape == (N, hc_dim)
    assert injection_logits.shape == (N, hc_count)
    assert residual.stride(1) == 1 and block_output.stride(1) == 1 and injection_logits.stride(1) == 1
    assert norm_weight.is_contiguous() and norm_weight.numel() == DIM

    out = residual.new_empty(residual.shape)
    y = residual.new_empty(residual.shape)
    _hc_combine_rmsnorm_kernel[(N, hc_count)](
        block_output,
        residual,
        injection_logits,
        norm_weight,
        out,
        y,
        block_output.stride(0),
        residual.stride(0),
        injection_logits.stride(0),
        out.stride(0),
        y.stride(0),
        hc_dim,
        hc_count,
        EPS=eps,
    )
    return out, y


@triton.jit
def _hc_up_gate_mix_kernel(
    lora_ptr,
    w_ptr,
    rn_ptr,
    y_ptr,
    K,
    stride_wn,
    h_lo,
    w_h0,
    w_stream,
    HC: tl.constexpr,
    HC_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    # One hidden index per program: its HC gate rows are reduced like gemv.py's _gemv_rows
    # (one row, one K step) over the bf16-rounded hc_silu output, rounded to bf16 as gemv
    # stores them, then summed in hc_gate_mix's stream order. Program i writes y[i] for hidden
    # index h_lo + i, whose stream-s weight row is s * w_stream + h - w_h0.
    i = tl.program_id(0) + tl.arange(0, 1)
    h = h_lo + i
    cols = tl.arange(0, BLOCK_K)
    kmask = cols < K
    a = tl.load(lora_ptr + cols, mask=kmask, other=0.0).to(tl.float32) / HC
    a = (a * tl.sigmoid(a)).to(tl.bfloat16)

    mixed = tl.zeros([1], dtype=tl.float32)
    for stream in tl.static_range(HC):
        rows = stream * w_stream + h - w_h0
        w = tl.load(
            w_ptr + rows[:, None].to(tl.int64) * stride_wn + cols[None, :],
            mask=kmask[None, :], other=0.0, eviction_policy="evict_first",
        )
        acc = tl.zeros((1, BLOCK_K), dtype=tl.float32)
        acc += w.to(tl.float32) * a[None, :].to(tl.float32)
        gate = tl.sum(acc, axis=1).to(tl.bfloat16)
        x = tl.load(rn_ptr + stream * HC_DIM + h)
        mixed += tl.sigmoid(gate.to(tl.float32)) * x.to(tl.float32)
    mixed /= HC
    tl.store(y_ptr + i, mixed.to(y_ptr.dtype.element_ty))


def hc_up_gate_mix_supported(w_up: torch.Tensor, hc_count: int, lora: torch.Tensor | None = None) -> bool:
    """Whether :func:`hc_up_gate_mix` reproduces ``gemv`` on this shape: the bf16 GEMV tile must
    be one row and one K step, the reduction the fused kernel mirrors."""
    from freetoken.kernel.triton.gemv import _config

    if w_up.dim() != 2 or w_up.dtype != torch.bfloat16 or w_up.stride(1) != 1:
        return False
    N, K = w_up.shape
    if lora is not None and (
        lora.dim() != 2 or lora.shape != (1, K) or lora.stride(1) != 1 or lora.dtype != torch.bfloat16
    ):
        return False
    bn, bk, warps = _config(N, K)
    return N % hc_count == 0 and bn == 1 and warps == 1 and K <= bk


def hc_up_gate_mix(
    lora: torch.Tensor,
    w_up: torch.Tensor,
    rn: torch.Tensor,
    hc_count: int,
    h_lo: int = 0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``hc_gate_mix(rn, gemv(hc_silu(lora), w_up), hc_count)`` for one token in one launch,
    bit for bit (see :func:`hc_up_gate_mix_supported`). With ``out`` given, only hidden indices
    ``[h_lo, h_lo + out.shape[1])`` are computed into it."""
    from freetoken.kernel.triton.gemv import _config

    N, K = w_up.shape
    hc_dim = N // hc_count
    assert rn.shape == (1, N) and rn.stride(1) == 1
    _, bk, warps = _config(N, K)
    if out is None:
        out = rn.new_empty((1, hc_dim))
    count = out.shape[1]
    assert out.shape == (1, count) and out.stride(1) == 1 and 0 <= h_lo and h_lo + count <= hc_dim
    _hc_up_gate_mix_kernel[(count,)](
        lora, w_up, rn, out, K, w_up.stride(0), h_lo, 0, hc_dim,
        HC=hc_count, HC_DIM=hc_dim, BLOCK_K=bk, num_warps=warps,
    )
    return out


__all__ = [
    "grouped_gemma_rmsnorm",
    "hc_combine",
    "hc_combine_norm",
    "hc_combine_rmsnorm",
    "hc_gate_mix",
    "hc_silu",
    "hc_up_gate_mix",
    "hc_up_gate_mix_supported",
]

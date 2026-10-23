# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# Adapted from SGLang (python/sglang/srt/models/qwen4_exp.py)
"""UVA row gather for the Qwen3.8-Flash-Next PLE n-gram table.

The table (320,001,536 rows x 160, FP8-e4m3 + one scalar scale = 47.7 GiB) stays in pinned
host memory and the GPU dereferences it in place over PCIe -- at its host VA on Linux/UVA, at
the mapped device address on WDDM (``kernel/pinned.device_ptr``). One program per requested
row: read the row, widen to fp32, apply the per-tensor scale, store bf16.

Ids outside the table store zeros.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import e4m3_native_cx, e4m3_u8_to_f32

# Latency-bound over PCIe, so keep the block small and let many of them be in flight.
_NUM_WARPS = 1


@triton.jit
def _ple_gather_kernel(
    table_ptr,
    ids_ptr,
    out_ptr,
    scale,
    num_rows,
    EMB_DIM: tl.constexpr,
    IS_FP8: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    in_range = (idx >= 0) & (idx < num_rows)
    idx = tl.where(in_range, idx, 0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < EMB_DIM
    # the table is a host allocation: rebuild the typed pointer from the raw address
    if IS_FP8:
        if e4m3_native_cx():
            base = table_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
            values = tl.load(base + idx * EMB_DIM + offsets, mask=mask, other=0.0).to(tl.float32)
        else:
            # pre-sm_89 has no fp8e4nv type: load raw bytes and decode in software
            base = table_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
            values = e4m3_u8_to_f32(tl.load(base + idx * EMB_DIM + offsets, mask=mask, other=0))
    else:
        base = table_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
        values = tl.load(base + idx * EMB_DIM + offsets, mask=mask, other=0.0).to(tl.float32)
    values = tl.where(in_range, values * scale, 0.0)
    tl.store(
        out_ptr + row * EMB_DIM + offsets,
        values.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def ple_gather_rows(
    table_ptr: int,
    num_rows: int,
    embed_dim: int,
    row_ids: torch.Tensor,
    out: torch.Tensor,
    scale: float = 1.0,
    is_fp8: bool = True,
) -> torch.Tensor:
    """Gather ``row_ids`` from the host-resident table at ``table_ptr`` into ``out``.

    ``row_ids`` is a flat device int tensor; ``out`` is ``[row_ids.numel(), embed_dim]``
    bf16 on the same device. ``table_ptr`` is the address the GPU must dereference
    (``kernel/pinned.device_ptr``), not necessarily the host ``data_ptr``.
    """
    n = row_ids.numel()
    assert out.shape == (n, embed_dim) and out.is_contiguous(), out.shape
    if n:
        _ple_gather_kernel[(n,)](
            table_ptr,
            row_ids,
            out,
            float(scale),
            num_rows,
            EMB_DIM=embed_dim,
            IS_FP8=is_fp8,
            BLOCK_D=triton.next_power_of_2(embed_dim),
            num_warps=_NUM_WARPS,
        )
    return out


__all__ = ["ple_gather_rows"]


@triton.jit
def e2m1_to_f32(code):
    """e2m1 nibble -> fp32 (exact): the 3 magnitude bits go to the top of an fp16 exponent/
    mantissa field, which is the value times 2^-14 (incl. the 0.5 subnormal)."""
    h = ((code & 0x8).to(tl.uint16) << 12) | ((code & 0x7).to(tl.uint16) << 9)
    return h.to(tl.float16, bitcast=True).to(tl.float32) * 16384.0


@triton.jit
def _nvfp4_rows_dequant_kernel(
    rows_ptr,
    out_ptr,
    scale,
    n_rows,
    HALF: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Interleaved NVFP4 rows [n, HALF codes | HALF/8 e4m3 scales] -> bf16 [n, 2*HALF]."""
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    k = tl.arange(0, BLOCK_K)
    mask = (r < n_rows)[:, None] & (k < 2 * HALF)[None, :]
    src = rows_ptr + r.to(tl.int64)[:, None] * WIDTH
    byte = tl.load(src + k[None, :] // 2, mask=mask, other=0)
    code = tl.where((k % 2 == 0)[None, :], byte & 0xF, byte >> 4)
    sb = tl.load(src + HALF + k[None, :] // 16, mask=mask, other=0)
    val = e2m1_to_f32(code) * e4m3_u8_to_f32(sb) * scale
    tl.store(out_ptr + r.to(tl.int64)[:, None] * (2 * HALF) + k[None, :], val.to(tl.bfloat16),
             mask=mask)


def nvfp4_rows_dequant(rows: torch.Tensor, head_dim: int, scale: float, out: torch.Tensor | None = None) -> torch.Tensor:
    """``rows`` uint8 [n, head_dim/2 + head_dim/16] (codes, then block scales) -> bf16 [n, head_dim]."""
    half = head_dim // 2
    width = half + head_dim // 16
    rows = rows.view(-1, width)
    n = rows.shape[0]
    if out is None:
        out = torch.empty((n, head_dim), dtype=torch.bfloat16, device=rows.device)
    block_r = 16
    _nvfp4_rows_dequant_kernel[(triton.cdiv(n, block_r),)](
        rows, out, float(scale), n, HALF=half, WIDTH=width, BLOCK_R=block_r,
        BLOCK_K=triton.next_power_of_2(head_dim), num_warps=4,
    )
    return out

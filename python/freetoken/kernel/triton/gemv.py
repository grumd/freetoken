"""Single-token bf16 GEMV ``y[1, N] = x[1, K] @ W[N, K]^T`` for decode.

cuBLAS serves M=1 with split-K tensor-core GEMMs that reach 250-550 GB/s on the skinny
shapes Qwen3.8 issues every step (RTX 3080, 760 GB/s): this is a plain streaming
reduction, one row block per program, the whole K per program (no split-K partials, so
it is deterministic and a single launch). Tiles per (N, K) come from ``_config``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_rows(x_ptr, w_ptr, y_ptr, rows, out_rows, rmask, K, stride_wn,
               BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, EVEN_K: tl.constexpr):
    cols = tl.arange(0, BLOCK_K)
    w_ptrs = w_ptr + rows[:, None].to(tl.int64) * stride_wn + cols[None, :]
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        if EVEN_K:
            x = tl.load(x_ptr + k0 + cols)
            w = tl.load(w_ptrs, mask=rmask[:, None], other=0.0, eviction_policy="evict_first")
        else:
            kmask = k0 + cols < K
            x = tl.load(x_ptr + k0 + cols, mask=kmask, other=0.0)
            w = tl.load(w_ptrs, mask=rmask[:, None] & kmask[None, :], other=0.0,
                        eviction_policy="evict_first")
        acc += w.to(tl.float32) * x[None, :].to(tl.float32)
        w_ptrs += BLOCK_K
    y = tl.sum(acc, axis=1)
    tl.store(y_ptr + out_rows, y.to(y_ptr.dtype.element_ty), mask=rmask)


@triton.jit
def _gemv_kernel(
    x_ptr, w_ptr, y_ptr, N, K, stride_wn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, EVEN_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    _gemv_rows(x_ptr, w_ptr, y_ptr, rows, rows, rows < N, K, stride_wn, BLOCK_N, BLOCK_K, EVEN_K)


@triton.jit
def _gemv3_kernel(
    x_ptr, w0_ptr, w1_ptr, w2_ptr, y_ptr, N0, N1, N2, K, s0, s1, s2,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, EVEN_K: tl.constexpr,
):
    # N0 and N1 are multiples of BLOCK_N, so a row block never spans two matrices
    row0 = tl.program_id(0) * BLOCK_N
    offs = tl.arange(0, BLOCK_N)
    if row0 < N0:
        _gemv_rows(x_ptr, w0_ptr, y_ptr, row0 + offs, row0 + offs, row0 + offs < N0,
                   K, s0, BLOCK_N, BLOCK_K, EVEN_K)
    elif row0 < N0 + N1:
        local = row0 - N0 + offs
        _gemv_rows(x_ptr, w1_ptr, y_ptr, local, row0 + offs, local < N1,
                   K, s1, BLOCK_N, BLOCK_K, EVEN_K)
    else:
        local = row0 - N0 - N1 + offs
        _gemv_rows(x_ptr, w2_ptr, y_ptr, local, row0 + offs, local < N2,
                   K, s2, BLOCK_N, BLOCK_K, EVEN_K)


# Tiles swept with kbench/sweep_gemv.py on an RTX 3080 (q38 TP=2 decode shapes): a single
# warp streaming 512-1024 contiguous K elements per row wins everywhere but the wide-K
# hyper-connection down-projections, which need a full CTA per row to keep enough bytes
# in flight, and the K=320 projections, where rows are too short for one warp.
def _config(N: int, K: int) -> tuple[int, int, int]:
    if K >= 8192:
        return 1, 2048, 8
    if K <= 512:
        return (1, 512, 1) if N >= 8192 else (16, 128, 4)
    if K % 1024 == 0:
        return 1, 1024, 1
    return 2, 512, 1


def gemv(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """``x`` [1, K] bf16 contiguous, ``w`` [N, K] with unit inner stride -> [1, N]."""
    N, K = w.shape
    if out is None:
        out = torch.empty((1, N), dtype=x.dtype, device=x.device)
    bn, bk, warps = _config(N, K)
    _gemv_kernel[(triton.cdiv(N, bn),)](
        x, w, out, N, K, w.stride(0),
        BLOCK_N=bn, BLOCK_K=bk, EVEN_K=K % bk == 0, num_warps=warps,
    )
    return out


def gemv3(
    x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ cat(w0, w1, w2).T`` -> [1, N0 + N1 + N2] in one launch, without the concat."""
    K = x.shape[1]
    N0, N1, N2 = w0.shape[0], w1.shape[0], w2.shape[0]
    bn, bk, warps = _config(N0 + N1 + N2, K)
    while N0 % bn or N1 % bn:
        bn //= 2
    if out is None:
        out = torch.empty((1, N0 + N1 + N2), dtype=x.dtype, device=x.device)
    _gemv3_kernel[(triton.cdiv(N0, bn) + triton.cdiv(N1, bn) + triton.cdiv(N2, bn),)](
        x, w0, w1, w2, out, N0, N1, N2, K, w0.stride(0), w1.stride(0), w2.stride(0),
        BLOCK_N=bn, BLOCK_K=bk, EVEN_K=K % bk == 0, num_warps=warps,
    )
    return out


def gemv_with(bn: int, bk: int, warps: int):
    """A fixed-tile variant for sweeping."""

    def fn(x, w, out=None):
        N, K = w.shape
        if out is None:
            out = torch.empty((1, N), dtype=x.dtype, device=x.device)
        _gemv_kernel[(triton.cdiv(N, bn),)](
            x, w, out, N, K, w.stride(0),
            BLOCK_N=bn, BLOCK_K=bk, EVEN_K=K % bk == 0, num_warps=warps,
        )
        return out

    return fn


@triton.jit
def _gemv_mxfp8_kernel(
    x_ptr, w_ptr, s_ptr, y_ptr, N, K, stride_wn, stride_sn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """MXFP8 rows (e4m3 bytes + one e8m0 code per 32 inputs): the bf16 kernel's streaming
    reduction over half the bytes. e4m3 decodes exactly through an fp16 field (``v * 2^-8``);
    each 32-block's partial takes ``2^(code - 127 + 8)`` once. K % BLOCK_K == 0, BLOCK_K % 32 == 0."""
    KB: tl.constexpr = BLOCK_K // 32
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    cols = tl.arange(0, BLOCK_K)
    w_ptrs = w_ptr + rows[:, None].to(tl.int64) * stride_wn + cols[None, :]
    s_ptrs = s_ptr + rows[:, None].to(tl.int64) * stride_sn + tl.arange(0, KB)[None, :]
    acc = tl.zeros((BLOCK_N, KB), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        x = tl.load(x_ptr + k0 + cols).to(tl.float32)
        v = tl.load(w_ptrs, mask=rmask[:, None], other=0, eviction_policy="evict_first")
        h = ((v & 0x80).to(tl.uint16) << 8) | ((v & 0x7F).to(tl.uint16) << 7)
        w = h.to(tl.float16, bitcast=True).to(tl.float32)
        part = tl.sum(tl.reshape(w * x[None, :], (BLOCK_N, KB, 32)), axis=2)
        code = tl.load(s_ptrs, mask=rmask[:, None], other=0).to(tl.int32)
        # 2^(code - 119) built in the fp32 exponent field (code - 119 >= -126 for real codes)
        scale = ((code - 119 + 127) << 23).to(tl.float32, bitcast=True)
        acc += part * scale
        w_ptrs += BLOCK_K
        s_ptrs += KB
    y = tl.sum(acc, axis=1)
    tl.store(y_ptr + rows, y.to(y_ptr.dtype.element_ty), mask=rmask)


# swept with kbench/bench_mxfp8_gemv.py on an RTX 3080 at the q38 TP=2 dense shapes
def _config_mxfp8(N: int, K: int) -> tuple[int, int, int]:
    if K % 512:
        return 16, 64 if K % 128 else 128, 4
    if N <= 1024:
        return 1, 512, 2
    if K % 1024 == 0 and N <= 4096:
        return 1, 1024, 4
    return 8, 512, 1


def gemv_mxfp8(x: torch.Tensor, w: torch.Tensor, codes: torch.Tensor, out: torch.Tensor | None = None,
               cfg: tuple[int, int, int] | None = None) -> torch.Tensor:
    """``x`` [1, K] bf16 contiguous, ``w`` [N, K] e4m3 (or its uint8 view), ``codes`` [N, K/32]
    uint8 e8m0 -> [1, N]."""
    N, K = w.shape
    if out is None:
        out = torch.empty((1, N), dtype=x.dtype, device=x.device)
    bn, bk, warps = cfg or _config_mxfp8(N, K)
    _gemv_mxfp8_kernel[(triton.cdiv(N, bn),)](
        x, w.view(torch.uint8), codes, out, N, K, w.stride(0), codes.stride(0),
        BLOCK_N=bn, BLOCK_K=bk, num_warps=warps,
    )
    return out

"""Decode-path WNA16 (AutoGPTQ INT4) expert MLP: gate_up GEMV, SiLU*up, down GEMV, top-k sum.

Four launches per layer instead of six: each split-K GEMV writes fp32 partials, and the
reduction that folds them also applies the epilogue (SiLU(gate) * up after gate_up, the
routing weights and the sum over top-k after down), so nothing round-trips through bf16
between the two GEMMs. Each GEMV program streams [BLOCK_KW words x BLOCK_N] int32 tiles,
unpacks the 8 nibbles per word in registers and folds the zero point out of the inner
product: sum_k a_k (q_kn - z_n) s_n = s_n (sum_k a_k q_kn - z_n sum_k a_k), which holds
because a tile never straddles a quantization group.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemv_partials_kernel(
    a_ptr, qw_ptr, qz_ptr, sc_ptr, part_ptr, ids_ptr,
    R, N, kb_per, n_kb, K_OFF,
    stride_a,
    stride_qe, stride_qkw,
    stride_ze, stride_zg,
    stride_se, stride_sg,
    GROUP: tl.constexpr, TOP_K: tl.constexpr, A_ROW_IS_ROUTE: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_KW: tl.constexpr,
):
    r = tl.program_id(0)
    nb = tl.program_id(1)
    s = tl.program_id(2)
    slot = tl.load(ids_ptr + r).to(tl.int64)
    a_row = r if A_ROW_IS_ROUTE else r // TOP_K
    offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = offs_n < N
    offs_w = tl.arange(0, BLOCK_KW)
    offs_j = tl.arange(0, 8)
    shifts = offs_j * 4
    z_shift = (offs_n % 8) * 4
    qw_base = qw_ptr + slot * stride_qe + offs_n[None, :]
    qz_base = qz_ptr + slot * stride_ze + offs_n // 8
    sc_base = sc_ptr + slot * stride_se + offs_n
    a_base = a_ptr + a_row * stride_a
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    kb_end = tl.minimum((s + 1) * kb_per, n_kb)
    for kb in range(s * kb_per, kb_end):
        w0 = kb * BLOCK_KW
        word = tl.load(qw_base + (w0 + offs_w)[:, None] * stride_qkw, mask=nmask[None, :], other=0,
                       eviction_policy="evict_first")
        a = tl.load(a_base + (w0 + offs_w)[:, None] * 8 + offs_j[None, :]).to(tl.float32)
        code = ((word[:, None, :] >> shifts[None, :, None]) & 0xF).to(tl.float32)
        dot = tl.sum(tl.sum(code * a[:, :, None], axis=1), axis=0)
        g = (w0 * 8 + K_OFF) // GROUP
        zero = ((tl.load(qz_base + g * stride_zg, mask=nmask, other=0) >> z_shift) & 0xF) + 1
        scale = tl.load(sc_base + g * stride_sg, mask=nmask, other=0.0).to(tl.float32)
        acc += scale * (dot - zero.to(tl.float32) * tl.sum(tl.sum(a, axis=1), axis=0))
    tl.store(part_ptr + (s * R + r) * N + offs_n, acc, mask=nmask)


@triton.jit
def _reduce_silu_mul_kernel(part_ptr, out_ptr, R, INTER, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = offs < INTER
    gate = tl.zeros((BLOCK,), dtype=tl.float32)
    up = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        row = part_ptr + (s * R + r) * (2 * INTER)
        gate += tl.load(row + offs, mask=m, other=0.0)
        up += tl.load(row + INTER + offs, mask=m, other=0.0)
    y = gate / (1.0 + tl.exp(-gate)) * up
    tl.store(out_ptr + r * INTER + offs, y.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _reduce_topk_sum_kernel(
    part_ptr, w_ptr, out_ptr, R, H, stride_wm,
    TOP_K: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = offs < H
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        r = t * TOP_K + k
        part = tl.zeros((BLOCK,), dtype=tl.float32)
        for s in tl.static_range(SPLIT):
            part += tl.load(part_ptr + (s * R + r) * H + offs, mask=m, other=0.0)
        acc += tl.load(w_ptr + t * stride_wm + k).to(tl.float32) * part
    tl.store(out_ptr + t * H + offs, acc.to(out_ptr.dtype.element_ty), mask=m)


# (BLOCK_N, BLOCK_KW, num_warps) and a target program count for the split-K, swept with
# kbench/bench_wna16_decode.py on an RTX 3080 at the q38 TP=2 shapes.
_GU_TILE = (128, 8, 4)
_DN_TILE = (128, 4, 4)
_TARGET_PROGRAMS = 544


def _split(programs: int, n_kb: int) -> int:
    split = max(1, min(n_kb, _TARGET_PROGRAMS // max(1, programs)))
    return triton.cdiv(n_kb, triton.cdiv(n_kb, split))


def _gemv(a, qw, qz, sc, ids, R, top_k, a_row_is_route, k_off, tile):
    bn, bkw, warps = tile
    kw, N = qw.shape[1], qw.shape[2]
    n_kb = kw // bkw
    split = _split(R * triton.cdiv(N, bn), n_kb)
    kb_per = triton.cdiv(n_kb, split)
    split = triton.cdiv(n_kb, kb_per)
    part = torch.empty((split, R, N), dtype=torch.float32, device=a.device)
    _gemv_partials_kernel[(R, triton.cdiv(N, bn), split)](
        a, qw, qz, sc, part, ids,
        R, N, kb_per, n_kb, k_off,
        a.stride(0),
        qw.stride(0), qw.stride(1),
        qz.stride(0), qz.stride(1),
        sc.stride(0), sc.stride(1),
        GROUP=128, TOP_K=top_k, A_ROW_IS_ROUTE=a_row_is_route,
        BLOCK_N=bn, BLOCK_KW=bkw, num_warps=warps,
    )
    return part, split


def supported(qw_gu: torch.Tensor, qw_dn: torch.Tensor, down_k_off: int) -> bool:
    """Tiles must not straddle a 128-group: K (in words) and the down band offset align."""
    return (
        qw_gu.shape[1] % _GU_TILE[1] == 0
        and qw_dn.shape[1] % _DN_TILE[1] == 0
        and down_k_off % (8 * _DN_TILE[1]) == 0
        and 128 % (8 * _GU_TILE[1]) == 0
        and 128 % (8 * _DN_TILE[1]) == 0
    )


def wna16_decode_silu(
    x: torch.Tensor,
    gate_up_qw: torch.Tensor, gate_up_qz: torch.Tensor, gate_up_sc: torch.Tensor,
    down_qw: torch.Tensor, down_qz: torch.Tensor, down_sc: torch.Tensor,
    topk_weights: torch.Tensor, topk_ids: torch.Tensor, down_k_off: int = 0,
    gu_tile=None, dn_tile=None,
) -> torch.Tensor:
    """SiLU-gated expert MLP for a few tokens; ``topk_ids`` are cache slots, ``x`` [M, H]."""
    M, H = x.shape
    top_k = topk_ids.shape[1]
    R = M * top_k
    inter = gate_up_qw.shape[2] // 2
    ids = topk_ids.reshape(-1)
    part1, s1 = _gemv(x, gate_up_qw, gate_up_qz, gate_up_sc, ids, R, top_k, False, 0, gu_tile or _GU_TILE)
    act = torch.empty((R, inter), dtype=x.dtype, device=x.device)
    _reduce_silu_mul_kernel[(R, triton.cdiv(inter, 256))](part1, act, R, inter, SPLIT=s1, BLOCK=256, num_warps=2)
    part2, s2 = _gemv(act, down_qw, down_qz, down_sc, ids, R, top_k, True, down_k_off, dn_tile or _DN_TILE)
    out = torch.empty((M, H), dtype=x.dtype, device=x.device)
    _reduce_topk_sum_kernel[(M, triton.cdiv(H, 512))](
        part2, topk_weights, out, R, H, topk_weights.stride(0),
        TOP_K=top_k, SPLIT=s2, BLOCK=512, num_warps=4,
    )
    return out

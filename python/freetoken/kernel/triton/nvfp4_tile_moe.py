"""NVFP4 expert kernels for the offload slot cache, tuned on the RTX 3080 (sm_86, no FP4 unit).

Same bank layout as :mod:`freetoken.kernel.triton.nvfp4_fused_moe` (``packed[slot, n, k//2]``
e2m1 codes, low nibble = even k; ``scale[slot, n, k//16]`` e4m3; fp32 ``global[slot, n]``),
the kernel structure of the WNA16 tile / fused-epilogue kernels:

* prefill dequantizes a whole [BLOCK_K, BLOCK_N] weight tile per K step and issues one
  bf16 ``tl.dot``; the gate/up variant keeps both halves in fp32 and fuses SiLU * up;
* decode is a split-K GEMV writing fp32 partials; the reductions apply SiLU * up and the
  routing weights / top-k sum (:mod:`freetoken.kernel.triton.wna16_decode`'s reducers).

Dequant is exact: an e2m1 code placed in an fp16 exponent/mantissa field is ``value * 2^-14``
(the 0.5 subnormal included), the e4m3 scale decodes to ``scale * 2^7`` in fp16, and their
product -- at most 6 significant bits, >= 2^-22 -- is exact in fp16 and in bf16. So the bf16
weight tile carries ``w * 2^-7`` with no rounding, and ``global * 2^7`` is applied to the fp32
accumulator.
"""

from __future__ import annotations

import triton
import triton.language as tl

from freetoken.kernel.triton.e4m3_compat import e4m3_u8_to_f16_x128
from freetoken.kernel.triton.wna16_decode import route_in_flight

# global epilogue factor: undoes the 2^-14 (e2m1 field) * 2^7 (e4m3 x128) of the dequant
DEQUANT_UNDO = 128.0


@triton.jit
def e2m1_f16(code):
    """e2m1 nibble (int) -> fp16 ``value * 2^-14``, exact."""
    h = ((code & 0x8) << 12) | ((code & 0x7) << 9)
    return h.to(tl.uint16).to(tl.float16, bitcast=True)


@triton.jit
def _dequant_tile_kn(p_base, s_base, offs_k, stride_pk, stride_sk):
    """[BLOCK_K, BLOCK_N] weight tile as bf16 ``w * 2^-7``. ``p_base`` / ``s_base`` point at
    the tile's first byte / scale per output row ``[1, BLOCK_N]``; k runs down axis 0."""
    byte = tl.load(p_base + (offs_k[:, None] // 2) * stride_pk)
    code = (byte >> ((offs_k[:, None] % 2) * 4)) & 0xF
    sc = e4m3_u8_to_f16_x128(tl.load(s_base + (offs_k[:, None] // 16) * stride_sk))
    return (e2m1_f16(code.to(tl.int32)) * sc).to(tl.bfloat16)


@triton.jit
def _dequant_tile_nk(p_base, s_base, offs_kb, offs_ks, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """[BLOCK_N, BLOCK_K] weight tile from contiguous byte rows (``p_base`` [BLOCK_N, 1]):
    both nibbles of each byte joined so k stays in natural order."""
    byte = tl.load(p_base + offs_kb[None, :]).to(tl.int32)
    code = tl.reshape(tl.join(byte & 0xF, byte >> 4), (BLOCK_N, BLOCK_K))
    sc = e4m3_u8_to_f16_x128(tl.load(s_base + offs_ks[None, :]))
    w = tl.reshape(e2m1_f16(code), (BLOCK_N, BLOCK_K // 16, 16)) * sc[:, :, None]
    return tl.reshape(w, (BLOCK_N, BLOCK_K)).to(tl.bfloat16)


@triton.jit
def _prefill_nvfp4_tile_kernel(
    a_ptr, p_ptr, s_ptr, g_ptr, c_ptr,
    topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_pe, stride_pn,
    stride_se, stride_sn,
    stride_ge,
    stride_cm, stride_cn,
    stride_tw,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    NK_TILE: tl.constexpr,
):
    """Grouped prefill GEMM ``C[route] = A[token] @ W[slot]^T * global`` over
    ``moe_align_block_size`` blocks. K % BLOCK_K == 0 and BLOCK_K % 16 == 0."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    slot = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k) * stride_am + offs_k[None, :] * stride_ak

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if NK_TILE:
        p_base = p_ptr + slot * stride_pe + offs_bn[:, None] * stride_pn
        s_base = s_ptr + slot * stride_se + offs_bn[:, None] * stride_sn
        offs_kb = tl.arange(0, BLOCK_SIZE_K // 2)
        offs_ks = tl.arange(0, BLOCK_SIZE_K // 16)
        for k0 in range(0, K // BLOCK_SIZE_K):
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = _dequant_tile_nk(p_base, s_base, offs_kb, offs_ks, BLOCK_SIZE_N, BLOCK_SIZE_K)
            acc += tl.dot(a, tl.trans(b))
            a_ptrs += BLOCK_SIZE_K * stride_ak
            p_base += BLOCK_SIZE_K // 2
            s_base += BLOCK_SIZE_K // 16
    else:
        p_base = p_ptr + slot * stride_pe + offs_bn[None, :] * stride_pn
        s_base = s_ptr + slot * stride_se + offs_bn[None, :] * stride_sn
        for k0 in range(0, K // BLOCK_SIZE_K):
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            acc += tl.dot(a, _dequant_tile_kn(p_base, s_base, offs_k, 1, 1))
            a_ptrs += BLOCK_SIZE_K * stride_ak
            p_base += BLOCK_SIZE_K // 2
            s_base += BLOCK_SIZE_K // 16

    g = tl.load(g_ptr + slot * stride_ge + offs_bn) * 128.0
    acc = acc * g[None, :]
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token * stride_tw, mask=token_mask, other=0)
        acc = acc * moe_weight[:, None]
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, acc.to(compute_type), mask=token_mask[:, None] & (offs_cn[None, :] < N))


@triton.jit
def _prefill_nvfp4_gate_up_silu_kernel(
    a_ptr, p_ptr, s_ptr, g_ptr, c_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    INTER, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_pe, stride_pn,
    stride_se, stride_sn,
    stride_ge,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    NK_TILE: tl.constexpr,
):
    """gate_up GEMM with SiLU(gate) * up fused: rows n (gate) and INTER + n (up) accumulate
    side by side in fp32, so neither half is rounded before the activation. Writes the
    activation ``[route, INTER]``."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(INTER, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % INTER
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    slot = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k) * stride_am + offs_k[None, :] * stride_ak

    acc_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if NK_TILE:
        pg = p_ptr + slot * stride_pe + offs_n[:, None] * stride_pn
        sg = s_ptr + slot * stride_se + offs_n[:, None] * stride_sn
        pu = pg + INTER * stride_pn
        su = sg + INTER * stride_sn
        offs_kb = tl.arange(0, BLOCK_SIZE_K // 2)
        offs_ks = tl.arange(0, BLOCK_SIZE_K // 16)
        for k0 in range(0, K // BLOCK_SIZE_K):
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            acc_g += tl.dot(a, tl.trans(_dequant_tile_nk(pg, sg, offs_kb, offs_ks, BLOCK_SIZE_N, BLOCK_SIZE_K)))
            acc_u += tl.dot(a, tl.trans(_dequant_tile_nk(pu, su, offs_kb, offs_ks, BLOCK_SIZE_N, BLOCK_SIZE_K)))
            a_ptrs += BLOCK_SIZE_K * stride_ak
            pg += BLOCK_SIZE_K // 2
            pu += BLOCK_SIZE_K // 2
            sg += BLOCK_SIZE_K // 16
            su += BLOCK_SIZE_K // 16
    else:
        pg = p_ptr + slot * stride_pe + offs_n[None, :] * stride_pn
        sg = s_ptr + slot * stride_se + offs_n[None, :] * stride_sn
        pu = pg + INTER * stride_pn
        su = sg + INTER * stride_sn
        for k0 in range(0, K // BLOCK_SIZE_K):
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            acc_g += tl.dot(a, _dequant_tile_kn(pg, sg, offs_k, 1, 1))
            acc_u += tl.dot(a, _dequant_tile_kn(pu, su, offs_k, 1, 1))
            a_ptrs += BLOCK_SIZE_K * stride_ak
            pg += BLOCK_SIZE_K // 2
            pu += BLOCK_SIZE_K // 2
            sg += BLOCK_SIZE_K // 16
            su += BLOCK_SIZE_K // 16

    gate = acc_g * (tl.load(g_ptr + slot * stride_ge + offs_n) * 128.0)[None, :]
    up = acc_u * (tl.load(g_ptr + slot * stride_ge + INTER + offs_n) * 128.0)[None, :]
    y = gate / (1.0 + tl.exp(-gate)) * up
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, y.to(compute_type), mask=token_mask[:, None] & (offs_cn[None, :] < INTER))


@triton.jit
def _gemv_partials_kernel(
    a_ptr, p_ptr, s_ptr, g_ptr, part_ptr, ids_ptr, copy_slots_ptr, num_copy_ptr,
    R, N, kw_per, n_kw,
    stride_a,
    stride_pe, stride_pn,
    stride_se, stride_sn,
    stride_ge,
    TOP_K: tl.constexpr, A_ROW_IS_ROUTE: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_KW: tl.constexpr,
    ROUTES: tl.constexpr = 0, MAX_COPY: tl.constexpr = 16,
):
    """Split-K decode GEMV over int32 words (``p_ptr`` is the bank viewed as int32
    [S, N, K/8]: 8 codes per word, two words share a scale):
    ``part[s, r, n] = global[n] * sum_{k in split s} a[k] * W[n, k]`` in fp32.
    ``ROUTES`` as in :func:`_gemv_partials_v2_kernel`."""
    r = tl.program_id(0)
    nb = tl.program_id(1)
    s = tl.program_id(2)
    slot = tl.load(ids_ptr + r).to(tl.int64)
    if ROUTES != 0:
        if route_in_flight(slot, copy_slots_ptr, num_copy_ptr, MAX_COPY) != (ROUTES == 2):
            return
    a_row = r if A_ROW_IS_ROUTE else r // TOP_K
    offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = offs_n < N
    offs_w = tl.arange(0, BLOCK_KW)
    p_base = p_ptr + slot * stride_pe + offs_n[:, None] * stride_pn
    s_base = s_ptr + slot * stride_se + offs_n[:, None] * stride_sn
    a_base = a_ptr + a_row * stride_a
    acc = tl.zeros((BLOCK_N, BLOCK_KW), dtype=tl.float32)
    kw_end = tl.minimum((s + 1) * kw_per, n_kw)
    for kw0 in range(s * kw_per, kw_end, BLOCK_KW):
        w_idx = kw0 + offs_w
        word = tl.load(p_base + w_idx[None, :], mask=nmask[:, None], other=0,
                       eviction_policy="evict_first")
        sc = e4m3_u8_to_f16_x128(tl.load(s_base + (w_idx // 2)[None, :], mask=nmask[:, None], other=0))
        part = tl.zeros((BLOCK_N, BLOCK_KW), dtype=tl.float32)
        for j in tl.static_range(8):
            a_j = tl.load(a_base + w_idx * 8 + j).to(tl.float32)
            part += e2m1_f16((word >> (4 * j)) & 0xF).to(tl.float32) * a_j[None, :]
        acc += part * sc.to(tl.float32)
    out = tl.sum(acc, axis=1) * (tl.load(g_ptr + slot * stride_ge + offs_n, mask=nmask, other=0.0) * 128.0)
    tl.store(part_ptr + (s * R + r) * N + offs_n, out, mask=nmask)


@triton.jit
def _half2_to_f32(h2):
    """int32 holding two fp16 -> (low half, high half) as fp32."""
    return tl.inline_asm_elementwise(
        asm="{ .reg .b16 l, h; mov.b32 {l, h}, $2; cvt.f32.f16 $0, l; cvt.f32.f16 $1, h; }",
        constraints="=r,=r,r", args=[h2], dtype=(tl.float32, tl.float32), is_pure=True, pack=1,
    )


@triton.jit
def _gemv_partials_v2_kernel(
    a_ptr, p_ptr, s_ptr, g_ptr, part_ptr, ids_ptr, copy_slots_ptr, num_copy_ptr,
    R, N, kw_per, n_kw,
    stride_a,
    stride_pe, stride_pn,
    stride_se, stride_sn,
    stride_ge,
    TOP_K: tl.constexpr, A_ROW_IS_ROUTE: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_KW: tl.constexpr,
    ROUTES: tl.constexpr = 0, MAX_COPY: tl.constexpr = 16,
):
    """:func:`_gemv_partials_kernel` with Marlin's e2m1 unpack: with a nibble in the top four
    bits of each 16-bit half, ``(q & 0x80008000) | ((q >> 3) & 0x0E000E00)`` is two fp16
    ``value * 2^-14`` at once (sign stays at bit 15, the magnitude lands at bits 9-11), and
    ``q << 4`` brings up the next pair. Word nibble j (k = 8w + j) pairs with j + 4.
    ``ROUTES`` 1 / 2 runs only the routes whose slot is resident / still being copied in."""
    r = tl.program_id(0)
    nb = tl.program_id(1)
    s = tl.program_id(2)
    slot = tl.load(ids_ptr + r).to(tl.int64)
    if ROUTES != 0:
        if route_in_flight(slot, copy_slots_ptr, num_copy_ptr, MAX_COPY) != (ROUTES == 2):
            return
    a_row = r if A_ROW_IS_ROUTE else r // TOP_K
    offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = offs_n < N
    offs_w = tl.arange(0, BLOCK_KW)
    p_base = p_ptr + slot * stride_pe + offs_n[:, None] * stride_pn
    s_base = s_ptr + slot * stride_se + offs_n[:, None] * stride_sn
    a_base = a_ptr + a_row * stride_a
    acc = tl.zeros((BLOCK_N, BLOCK_KW), dtype=tl.float32)
    kw_end = tl.minimum((s + 1) * kw_per, n_kw)
    for kw0 in range(s * kw_per, kw_end, BLOCK_KW):
        w_idx = kw0 + offs_w
        q = tl.load(p_base + w_idx[None, :], mask=nmask[:, None], other=0,
                    eviction_policy="evict_first")
        sc = e4m3_u8_to_f16_x128(tl.load(s_base + (w_idx // 2)[None, :], mask=nmask[:, None], other=0))
        a_ptrs = a_base + w_idx * 8
        part = tl.zeros((BLOCK_N, BLOCK_KW), dtype=tl.float32)
        for p in tl.static_range(4):
            # the top nibble of each half is word nibble 3 - p (low half) / 7 - p (high half)
            h2 = (q & -2147450880) | ((q >> 3) & 0x0E000E00)  # -2147450880 == 0x80008000
            lo, hi = _half2_to_f32(h2)
            part += lo * tl.load(a_ptrs + 3 - p).to(tl.float32)[None, :]
            part += hi * tl.load(a_ptrs + 7 - p).to(tl.float32)[None, :]
            q = q << 4
        acc += part * sc.to(tl.float32)
    out = tl.sum(acc, axis=1) * (tl.load(g_ptr + slot * stride_ge + offs_n, mask=nmask, other=0.0) * 128.0)
    tl.store(part_ptr + (s * R + r) * N + offs_n, out, mask=nmask)


__all__ = [
    "DEQUANT_UNDO",
    "_gemv_partials_kernel",
    "_gemv_partials_v2_kernel",
    "_prefill_nvfp4_gate_up_silu_kernel",
    "_prefill_nvfp4_tile_kernel",
    "e2m1_f16",
]

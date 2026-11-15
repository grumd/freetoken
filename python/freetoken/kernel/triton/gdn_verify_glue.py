"""The MTP verify/commit copies around a GatedDeltaNet layer, one launch each (they are pure
copies, so bit-identical to the torch ops they replace; ~7 small launches per layer were
~1 ms of a 4-token verify round over 36 layers)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _stage_kernel(
    x_ptr, s_x, a_ptr, s_a, b_ptr, s_b, cs_ptr, s_cs_slot, s_cs_c, s_cs_k, slots_ptr,
    win_ptr, s_w_n, s_w_c, mix_ptr, s_m_n, s_m_t, abuf_ptr, bbuf_ptr, C, HV,
    T: tl.constexpr, KM1: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = c < C
    slot = tl.load(slots_ptr + n).to(tl.int64)
    win = win_ptr + n * s_w_n + c.to(tl.int64) * s_w_c
    for j in tl.static_range(KM1):
        v = tl.load(cs_ptr + slot * s_cs_slot + c.to(tl.int64) * s_cs_c + j * s_cs_k, mask=cm)
        tl.store(win + j, v, mask=cm)
    for t in tl.static_range(T):
        v = tl.load(x_ptr + (n * T + t).to(tl.int64) * s_x + c, mask=cm)
        tl.store(win + KM1 + t, v, mask=cm)
        tl.store(mix_ptr + n * s_m_n + t * s_m_t + c, v, mask=cm)
    if tl.program_id(1) == 0:
        h = tl.arange(0, BLOCK_H)
        hm = h < HV
        for t in tl.static_range(T):
            row = (n * T + t).to(tl.int64)
            tl.store(abuf_ptr + row * HV + h, tl.load(a_ptr + row * s_a + h, mask=hm), mask=hm)
            tl.store(bbuf_ptr + row * HV + h, tl.load(b_ptr + row * s_b + h, mask=hm), mask=hm)


def stage_verify(conv_in, a, b, conv_state, slots, conv_win, mixed, abuf, bbuf, n: int, t: int) -> None:
    """``conv_win[:n] = [conv_state[slots] | raw^T]``, ``mixed[:n] = raw``, ``abuf/bbuf[:n] = a/b``
    with ``raw = conv_in`` [n*t, C] (row-strided views are fine)."""
    C = conv_in.shape[1]
    HV = a.shape[1]
    km1 = conv_state.shape[-1]
    assert conv_in.stride(1) == 1 and a.stride(1) == 1 and b.stride(1) == 1 and conv_win.stride(2) == 1
    assert mixed.stride(2) == 1 and abuf.is_contiguous() and bbuf.is_contiguous()
    BLOCK_C = 512
    _stage_kernel[(n, triton.cdiv(C, BLOCK_C))](
        conv_in, conv_in.stride(0), a, a.stride(0), b, b.stride(0),
        conv_state, conv_state.stride(0), conv_state.stride(1), conv_state.stride(2), slots,
        conv_win, conv_win.stride(0), conv_win.stride(1), mixed, mixed.stride(0), mixed.stride(1),
        abuf, bbuf, C, HV, T=t, KM1=km1, BLOCK_C=BLOCK_C, BLOCK_H=triton.next_power_of_2(HV),
        num_warps=4,
    )


@triton.jit
def _commit_conv_kernel(
    win_ptr, s_w_n, s_w_c, lens_ptr, slots_ptr, cs_ptr, s_cs_slot, s_cs_c, s_cs_k, C,
    KM1: tl.constexpr, BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = c < C
    L = tl.load(lens_ptr + n).to(tl.int64)
    slot = tl.load(slots_ptr + n).to(tl.int64)
    for j in tl.static_range(KM1):
        v = tl.load(win_ptr + n * s_w_n + c.to(tl.int64) * s_w_c + L + j, mask=cm)
        tl.store(cs_ptr + slot * s_cs_slot + c.to(tl.int64) * s_cs_c + j * s_cs_k, v, mask=cm)


def commit_conv(conv_win, lens, slots, conv_state, n: int) -> None:
    """``conv_state[slots[i]] = conv_win[i, :, lens[i] : lens[i] + km1]`` (the km1 raw inputs
    ending at the last accepted token)."""
    C, km1 = conv_win.shape[1], conv_state.shape[-1]
    assert conv_win.stride(2) == 1
    BLOCK_C = 512
    _commit_conv_kernel[(n, triton.cdiv(C, BLOCK_C))](
        conv_win, conv_win.stride(0), conv_win.stride(1), lens, slots,
        conv_state, conv_state.stride(0), conv_state.stride(1), conv_state.stride(2), C,
        KM1=km1, BLOCK_C=BLOCK_C, num_warps=4,
    )


__all__ = ["commit_conv", "stage_verify"]

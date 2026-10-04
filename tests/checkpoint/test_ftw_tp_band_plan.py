"""TP-band planning for FTW expert banks at TP>1.

DeepSeek-FTW checkpoints slice cleanly (all banks carry an intermediate axis);
Qwen-family checkpoints (e.g. RadixArk RVN NVFP4) additionally carry 2-D
per-expert global-scale banks (``down_global`` [E, H]) that must be read
full-width on every rank, while ``gate_up_global`` [E, 2I] HAS an intermediate
axis and must band-slice like every other gate_up bank.
"""
from __future__ import annotations

import pytest

from freetoken.checkpoint.ftw import _band_axis, _tp_band_plan


def test_down_packed_slices_the_packed_columns():
    sliced, segs, mem_axis = _tp_band_plan("down_packed", (512, 2560, 320), 0, 1280, 2560, 1280)
    assert sliced == (512, 2560, 160)
    assert segs is None  # down-family: read full entry, slice columns in memory
    assert mem_axis == 2


def test_down_global_reads_full_width():
    sliced, segs, _ = _tp_band_plan("down_global", (512, 2560), 0, 1280, 2560, 1280)
    assert sliced == (512, 2560)  # [E, H]: no packed axis, nothing to slice
    assert segs == "full"         # not None: None would hit the 3-D column slice


def test_gate_up_global_band_scales_offsets_to_bytes():
    # gate_up_global is fp16 (2 bytes): the row stride is in ELEMENTS, so the byte
    # offsets/lengths must be doubled. Without the element-size factor the offsets
    # are halved and only the first half of the experts is read (the rest stay
    # zero) -> degenerate, looping output (Qwen3.8-Flash-Next on 2x3090).
    sliced, segs, _ = _tp_band_plan("gate_up_global", (512, 1280), 0, 640, 1280, 640, elsize=2)
    assert sliced == (512, 1280)  # 2 * I_loc
    assert segs[0] == (0, 1280, 0)        # expert 0 gate: 640 elems * 2 B
    assert segs[1] == (1280, 1280, 1280)  # expert 0 up
    assert segs[2] == (2560, 1280, 2560)  # expert 1 gate: (1*1280) elems * 2 B


def test_gate_up_packed_with_trailing_cols():
    # 1-byte bank: elements == bytes, so elsize=1 leaves the offsets unchanged.
    sliced, segs, _ = _tp_band_plan("gate_up_packed", (512, 1280, 1280), 0, 640, 1280, 640)
    assert sliced == (512, 1280, 1280)
    assert segs[0] == (0, 640 * 1280, 0)


def test_nvfp4_gate_up_band_takes_both_halves():
    # [E, 2I, H/2] with I=640, rank 1 of 2 -> rows [320, 640) of the gate half AND of the
    # up half. Slicing the fused axis as one block ([640, 1280)) handed rank 1 the whole up
    # half and rank 0 the whole gate half: Qwen3.8-Flash-Next-NVFP4 decoded garbage at TP=2.
    sliced, segs, mem_axis = _tp_band_plan("gate_up_packed", (2, 1280, 16), 320, 640, 640, 320)
    assert sliced == (2, 640, 16) and mem_axis is None
    assert segs[:2] == [(320 * 16, 320 * 16, 0), ((640 + 320) * 16, 320 * 16, 320 * 16)]


def test_unknown_bank_is_rejected():
    with pytest.raises(ValueError, match="unknown bank"):
        _tp_band_plan("mystery", (512,), 0, 1, 2, 1)


def test_band_axis_overrides_only_wna16():
    # WNA16 [E, packed_in, out]: gate_up's intermediate is the last axis, down's the packed one
    assert _band_axis("gate_up_qweight", (8, 320, 1280), "w4a16") == 2
    assert _band_axis("down_qweight", (8, 80, 2560), "w4a16") == 1
    assert _band_axis("down_global", (8, 2560), "w4a16") is None
    # NVFP4 keeps the default plan: an axis-1 override on the fused gate_up sliced it as one
    # block and fed each rank the wrong half (garbage output on Flash-Next-NVFP4 at TP=2)
    assert _band_axis("gate_up_packed", (8, 1280, 1280), "nvfp4") is None
    assert _band_axis("gate_up_scale", (8, 1280, 160), None) is None


# --- WNA16 bands: group-aligned and balanced (FREETOKEN_WNA16_BALANCED_TP) ---------------
# Qwen3.8-Flash-Next geometry: E experts, H=2560, I=640, group 128. TP1 bank shapes:
#   gate_up [E, 320, 1280] i32, gate_up_zero [E, 20, 160] i32, gate_up_scale [E, 20, 1280] f16
#   down    [E,  80, 2560] i32, down_zero    [E,  5, 320] i32, down_scale    [E,  5, 2560] f16

import numpy as np
import torch

from freetoken.checkpoint.ftw import _slice_band, _wna16_band_cols

_H, _I, _E = 2560, 640, 4
_W4 = {
    "gate_up": ((_E, _H // 8, 2 * _I), np.int32),
    "gate_up_zero": ((_E, _H // 128, 2 * _I // 8), np.int32),
    "gate_up_scale": ((_E, _H // 128, 2 * _I), np.float16),
    "down": ((_E, _I // 8, _H), np.int32),
    "down_zero": ((_E, _I // 128, _H // 8), np.int32),
    "down_scale": ((_E, _I // 128, _H), np.float16),
}


def test_wna16_group_aligned_band_matches_the_old_plan():
    # rank 1 of the 256 / 384 split: 384 of 640 -> 3 of 5 down groups, 2 for rank 0
    for base, (shape, _) in _W4.items():
        axis = _band_axis(base, shape, "w4a16")
        sliced, segs, mem_axis = _tp_band_plan(base, shape, 256, 640, _I, 384, axis=axis)
        assert segs is None and mem_axis == axis
        assert sliced[axis] == shape[axis] * 384 // 640, base
    sliced, _, _ = _tp_band_plan("down_scale", _W4["down_scale"][0], 0, 256, _I, 256, axis=1)
    assert sliced == (_E, 2, _H)


def test_wna16_balanced_band_keeps_the_straddled_down_group():
    # 320 / 320: both ranks overlap 3 of the 5 down groups (rank 0 groups 0-2, rank 1 groups 2-4),
    # while the packed qweight / gate_up axes split exactly in half.
    for rank, (lo, hi) in enumerate([(0, 320), (320, 640)]):
        for base in ("down_zero", "down_scale"):
            shape = _W4[base][0]
            axis, a0, a1 = _wna16_band_cols(base, shape, lo, hi, _I)
            assert (axis, a0, a1) == (1, 2 * rank, 3 + 2 * rank), (base, rank)
            sliced, _, _ = _tp_band_plan(base, shape, lo, hi, _I, 320, axis=1)
            assert sliced == (_E, 3, shape[2]), (base, rank)
        assert _wna16_band_cols("down", _W4["down"][0], lo, hi, _I) == (1, 40 * rank, 40 * rank + 40)
        assert _tp_band_plan("down", _W4["down"][0], lo, hi, _I, 320, axis=1)[0] == (_E, 40, _H)
        assert _wna16_band_cols("gate_up", _W4["gate_up"][0], lo, hi, _I) == (2, lo, hi)
        assert _tp_band_plan("gate_up", _W4["gate_up"][0], lo, hi, _I, 320, axis=2)[0] == (_E, _H // 8, 640)
        assert _wna16_band_cols("gate_up_zero", _W4["gate_up_zero"][0], lo, hi, _I) == (2, 40 * rank, 40 * rank + 40)
        assert _tp_band_plan("gate_up_zero", _W4["gate_up_zero"][0], lo, hi, _I, 320, axis=2)[0] == (_E, _H // 128, 80)


@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_wna16_slice_band_matches_the_hf_reader(monkeypatch, balanced, rank):
    """The FTW band of a TP1 bank == the bank the HF reader builds at TP2 for that rank:
    the HF path slices each expert piece (``_tp_slice_piece``) and the kernel fuses
    gate/up on the last axis; the FTW path slices the fused full-width bank."""
    import re

    from freetoken.models.wna16_banks import Wna16ExpertSourceSpec, _tp_slice_piece, wna16_tp_bands

    monkeypatch.setenv("FREETOKEN_WNA16_BALANCED_TP", "1" if balanced else "0")
    lo, hi, _, _ = wna16_tp_bands(_I, 2, rank)
    assert (lo, hi) == ((320 * rank, 320 * rank + 320) if balanced else ((0, 256), (256, 640))[rank])
    spec = Wna16ExpertSourceSpec(re.compile(""), {}, lambda layer, cfg: layer, "test")
    rng = np.random.default_rng(rank + 2 * balanced)

    def piece(shape, dt):
        return torch.from_numpy(rng.integers(0, 1000, size=shape).astype(dt))

    # per-expert HF pieces, in the checkpoint's [K/8, N] / [K/128, N/8] / [K/128, N] layouts
    gate = {s: piece((_H // d, _I // n), dt) for s, d, n, dt in
            (("", 8, 1, np.int32), ("_zero", 128, 8, np.int32), ("_scale", 128, 1, np.float16))}
    up = {s: piece(t.shape, t.numpy().dtype) for s, t in gate.items()}
    down = {s: piece((_I // d, _H // n), dt) for s, d, n, dt in
            (("", 8, 1, np.int32), ("_zero", 128, 8, np.int32), ("_scale", 128, 1, np.float16))}
    for suffix in ("", "_zero", "_scale"):
        # TP1 fused bank row for one expert (what the converter writes) ...
        full_gu = torch.cat([gate[suffix], up[suffix]], dim=-1)[None]
        full_dn = down[suffix][None]
        # ... and the HF rank bank row: slice the pieces, then fuse
        want_gu = torch.cat([_tp_slice_piece(spec, "gate" + suffix, gate[suffix], lo, hi),
                             _tp_slice_piece(spec, "up" + suffix, up[suffix], lo, hi)], dim=-1)[None]
        want_dn = _tp_slice_piece(spec, "down" + suffix, down[suffix], lo, hi)[None]
        band = (_I, hi - lo, lo, hi)
        for base, full, want in (("gate_up" + suffix, full_gu, want_gu), ("down" + suffix, full_dn, want_dn)):
            shape = tuple(full.shape)
            axis = _band_axis(base, shape, "w4a16")
            sliced, _, mem_axis = _tp_band_plan(base, shape, lo, hi, _I, hi - lo, axis=axis)
            got = _slice_band(full.numpy(), base, shape, band, mem_axis, "w4a16")
            assert got.shape == tuple(sliced) == tuple(want.shape), (base, got.shape, sliced, want.shape)
            np.testing.assert_array_equal(got, want.numpy())


def test_nvfp4_down_slice_band_keeps_the_column_rule():
    # non-WNA16 banks keep the proportional in-memory column slice
    arr = np.arange(2 * 4 * 320, dtype=np.uint8).reshape(2, 4, 320)
    got = _slice_band(arr, "down_packed", arr.shape, (1280, 640, 640, 1280), 2, "nvfp4")
    np.testing.assert_array_equal(got, arr[:, :, 160:320])

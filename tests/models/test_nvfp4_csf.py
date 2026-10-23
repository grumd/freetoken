# models/nvfp4_csf.py: lossless NVFP4-CSF scale decode, against an independent encoder
# (b12x's byte-window fixture: base = row min capped at 240, offsets > 15 become exceptions).
from __future__ import annotations

import numpy as np
import pytest
import torch

from freetoken.models.nvfp4_csf import CsfPairBuffer, csf_geometry, decode_csf_scales


def _encode(src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = src.shape
    s = src.to(torch.int16)
    base = torch.clamp(s.min(1).values, max=240)
    off = s - base[:, None]
    out = off > 15
    off[out] = 0
    off = off.to(torch.uint8)
    packed = off[:, ::2] | (off[:, 1::2] << 4)
    fixed = torch.cat([base.to(torch.uint8).reshape(-1, 16), packed.reshape(rows // 16, -1)], 1)
    pos = out.flatten().nonzero().flatten()
    words = pos | (s.flatten()[pos].to(torch.int64) << 24)
    return fixed, torch.from_numpy(words.numpy().astype(np.uint32))


@pytest.mark.parametrize("rows,cols", [(640, 160), (2560, 40), (16, 2)])
def test_decode_round_trips_the_reference_encoder(rows, cols):
    g = torch.Generator().manual_seed(rows + cols)
    src = torch.randint(0, 12, (rows, cols), generator=g) + torch.randint(60, 110, (rows, 1), generator=g)
    src[torch.rand(rows, cols, generator=g) < 0.02] = 126  # far outside the window -> exceptions
    fixed, exc = _encode(src.to(torch.uint8))
    assert csf_geometry(tuple(fixed.shape)) == (rows, cols)
    out = decode_csf_scales(fixed, exc)
    assert out.dtype == torch.float8_e4m3fn and out.shape == (rows, cols)
    assert torch.equal(out.view(torch.uint8), src.to(torch.uint8))


def test_decode_without_exceptions():
    src = torch.full((32, 8), 100, dtype=torch.uint8)
    fixed, exc = _encode(src)
    assert exc.numel() == 0
    assert torch.equal(decode_csf_scales(fixed, exc).view(torch.uint8), src)


def test_pair_buffer_pairs_streams_in_any_order():
    src = torch.randint(70, 90, (32, 8), dtype=torch.uint8)
    fixed, exc = _encode(src)
    buf = CsfPairBuffer()
    stem = "model.layers.0.mlp.experts.3.down_proj.weight_scale"
    assert buf.add(stem + ".nvfp4_csf_exceptions", exc) is None
    name, scale = buf.add(stem + ".nvfp4_csf_fixed", fixed)
    assert name == stem and torch.equal(scale.view(torch.uint8), src)
    buf.assert_empty()
    buf.add(stem + ".nvfp4_csf_fixed", fixed)
    with pytest.raises(ValueError):
        buf.assert_empty()


def test_exception_out_of_range_is_rejected():
    fixed, _ = _encode(torch.full((16, 4), 90, dtype=torch.uint8))
    bad = torch.tensor([64 | (100 << 24)], dtype=torch.int64).numpy().astype(np.uint32)
    with pytest.raises(ValueError):
        decode_csf_scales(fixed, torch.from_numpy(bad))

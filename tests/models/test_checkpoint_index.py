"""The checkpoint-format layer: key inventory + generic storage detection/dequant.

CPU-only, synthetic safetensors. The point is that the reader layer can be driven by
what is actually on disk (dtype + companion keys), independent of which exporter
wrote the checkpoint.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from freetoken.models.checkpoint_index import (
    CheckpointIndex,
    StorageKind,
    dequantize,
    dequantize_fp8_block,
    dequantize_fp8_per_row,
    detect_storage,
)

H, I = 8, 12
BLOCK = 4


def _dequant_fp8_block_ref(weight: torch.Tensor, scale: torch.Tensor, block: int) -> torch.Tensor:
    out, inn = weight.shape
    ref = torch.empty(out, inn, dtype=torch.float32)
    for r in range(out // block):
        for c in range(inn // block):
            ref[r * block : (r + 1) * block, c * block : (c + 1) * block] = (
                weight[r * block : (r + 1) * block, c * block : (c + 1) * block].to(torch.float32)
                * scale[r, c].to(torch.float32)
            )
    return ref.to(torch.bfloat16)


def _bf16(shape) -> torch.Tensor:
    return torch.randn(shape, dtype=torch.bfloat16)


def _fp8(shape) -> torch.Tensor:
    return torch.randn(shape).to(torch.float8_e4m3fn)


def test_index_is_global_across_shards(tmp_path):
    # weight in shard A, its per-row scale in shard B: only a global inventory sees it.
    save_file({"a.weight": _fp8((I, H))}, str(tmp_path / "model-00001.safetensors"))
    save_file({"a.weight_scale": torch.ones(I)}, str(tmp_path / "model-00002.safetensors"))
    idx = CheckpointIndex.load(str(tmp_path))
    assert "a.weight" in idx and "a.weight_scale" in idx
    assert detect_storage(idx, "a") is StorageKind.FP8_PER_ROW
    out = dequantize(idx, "a", get=lambda n: _load(tmp_path, idx, n))
    assert out.dtype is torch.bfloat16 and out.shape == (I, H)


def _load(folder, idx: CheckpointIndex, name: str) -> torch.Tensor:
    import safetensors

    with safetensors.safe_open(str(folder / idx.shard(name)), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def test_storage_detection(tmp_path):
    save_file(
        {
            "bf16.weight": _bf16((I, H)),
            "row.weight": _fp8((I, H)),
            "row.weight_scale": torch.ones(I, dtype=torch.bfloat16),
            "blk.weight": _fp8((I, H)),
            "blk.weight_scale_inv": torch.ones(I // BLOCK, H // BLOCK, dtype=torch.bfloat16),
            "nv.weight": torch.zeros(I, H // 2, dtype=torch.uint8),
            "nv.weight_scale": torch.ones(I, H // 16, dtype=torch.float8_e4m3fn),
            "nv.weight_scale_2": torch.tensor(0.5),
        },
        str(tmp_path / "model.safetensors"),
    )
    idx = CheckpointIndex.load(str(tmp_path))
    assert detect_storage(idx, "bf16") is StorageKind.BF16
    assert detect_storage(idx, "row") is StorageKind.FP8_PER_ROW
    assert detect_storage(idx, "blk") is StorageKind.FP8_BLOCK
    assert detect_storage(idx, "nv") is StorageKind.NVFP4


def test_dequantize_fp8_per_row_broadcasts_the_row_scale():
    w = _fp8((I, H))
    scale = torch.rand(I, dtype=torch.bfloat16) + 0.5
    got = dequantize_fp8_per_row(w, scale)
    ref = (w.to(torch.float32) * scale.to(torch.float32).unsqueeze(-1)).to(torch.bfloat16)
    assert torch.equal(got, ref)


def test_dequantize_fp8_block_matches_the_reference():
    w = _fp8((I, H))
    scale = (torch.rand(I // BLOCK, H // BLOCK, dtype=torch.bfloat16) + 0.5)
    got = dequantize_fp8_block(w, scale, block=BLOCK)
    assert got.dtype is torch.bfloat16
    assert torch.equal(got, _dequant_fp8_block_ref(w, scale, BLOCK))


def test_dequantize_rejects_nvfp4_without_a_bf16_target(tmp_path):
    save_file(
        {
            "nv.weight": torch.zeros(I, H // 2, dtype=torch.uint8),
            "nv.weight_scale": torch.ones(I, H // 16, dtype=torch.float8_e4m3fn),
            "nv.weight_scale_2": torch.tensor(0.5),
        },
        str(tmp_path / "model.safetensors"),
    )
    idx = CheckpointIndex.load(str(tmp_path))
    import pytest

    with pytest.raises(ValueError, match="NVFP4"):
        dequantize(idx, "nv", get=lambda n: _load(tmp_path, idx, n))


def test_mxfp8_weight_is_detected_and_dequantized_exactly(tmp_path):
    from safetensors.torch import save_file

    from freetoken.models.checkpoint_index import (
        CheckpointIndex, StorageKind, TensorReader, dequantize, detect_storage,
    )

    g = torch.Generator().manual_seed(0)
    q = (torch.randint(0, 256, (8, 64), generator=g, dtype=torch.uint8) & 0xF7).view(torch.float8_e4m3fn)
    codes = torch.randint(110, 125, (8, 2), generator=g, dtype=torch.uint8)
    save_file({"lin.weight": q, "lin.weight_scale": codes}, str(tmp_path / "model.safetensors"))
    idx = CheckpointIndex.load(str(tmp_path))
    assert detect_storage(idx, "lin") is StorageKind.MXFP8
    rd = TensorReader(idx)
    w = dequantize(idx, "lin", rd.get)
    ref = q.float().view(8, 2, 32) * torch.exp2(codes.float() - 127).unsqueeze(-1)
    assert w.dtype == torch.bfloat16 and torch.equal(w.float(), ref.view(8, 64))


def test_nvfp4_dense_dequant_is_opt_in(tmp_path):
    from safetensors.torch import save_file

    from freetoken.models.checkpoint_index import CheckpointIndex, TensorReader, dequantize

    packed = torch.tensor([[0x21, 0xF7] * 8], dtype=torch.uint8)  # codes 1,2 / 7,15 -> 0.5,1 / 6,-6
    scale = torch.tensor([[2.0, 0.5]], dtype=torch.float32).to(torch.float8_e4m3fn)
    glob = torch.tensor(0.25, dtype=torch.float32)
    save_file({"v.weight": packed, "v.weight_scale": scale, "v.weight_scale_2": glob}, str(tmp_path / "model.safetensors"))
    idx = CheckpointIndex.load(str(tmp_path))
    rd = TensorReader(idx)
    with pytest.raises(ValueError):
        dequantize(idx, "v", rd.get)
    w = dequantize(idx, "v", rd.get, nvfp4_to_bf16=True)
    vals = torch.tensor([0.5, 1.0, 6.0, -6.0] * 8)
    ref = vals * torch.tensor([2.0] * 16 + [0.5] * 16) * 0.25
    assert torch.equal(w.float().view(-1), ref)

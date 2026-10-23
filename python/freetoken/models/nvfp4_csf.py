"""NVFP4-CSF: the lossless compressed form of ModelOpt NVFP4 block scales.

A routed-expert ``weight_scale`` [rows, K/16] (e4m3 bytes) is stored as two tensors:

* ``<name>.weight_scale.nvfp4_csf_fixed`` uint8, one 16-row slab per row of
  ``[16 base bytes | 16 x cols/2 nibble bytes]``: scale = base[row] + nibble (low nibble =
  even column), a byte add on the e4m3 bit pattern;
* ``<name>.weight_scale.nvfp4_csf_exceptions`` uint32 records, sorted by position:
  low 24 bits ``row * cols + col``, high 8 bits the exact scale byte (its nibble is 0).

Format reference: local-inference-lab/b12x ``docs/lossless-scale-compression.md`` and
``b12x/_lib/quant/nvfp4_csf.py`` (codec 0, "byte window"). Decoding happens once, while the
banks are built, so the served banks hold plain e4m3 scales.
"""

from __future__ import annotations

import torch

CSF_FIXED_SUFFIX = ".nvfp4_csf_fixed"
CSF_EXCEPTIONS_SUFFIX = ".nvfp4_csf_exceptions"
CSF_SUFFIXES = (CSF_FIXED_SUFFIX, CSF_EXCEPTIONS_SUFFIX)


def csf_geometry(fixed_shape: tuple[int, ...]) -> tuple[int, int]:
    """``(rows, cols)`` of the scale matrix a fixed stream of ``fixed_shape`` encodes."""
    slabs, width = int(fixed_shape[0]), int(fixed_shape[-1])
    if width % 16 or width < 32:
        raise ValueError(f"NVFP4-CSF fixed stream width {width} is not 16 * (1 + cols/2)")
    return 16 * slabs, 2 * (width // 16 - 1)


def decode_csf_scales(fixed: torch.Tensor, exceptions: torch.Tensor) -> torch.Tensor:
    """Decode one projection's scales -> ``[rows, cols]`` float8_e4m3fn (bit-exact)."""
    rows, cols = csf_geometry(tuple(fixed.shape))
    f = fixed.reshape(rows // 16, 16 + 16 * (cols // 2))
    base = f[:, :16].reshape(rows, 1).to(torch.int32)
    packed = f[:, 16:].reshape(rows, cols // 2).to(torch.int32)
    out = torch.empty((rows, cols), dtype=torch.int32)
    out[:, 0::2] = packed & 15
    out[:, 1::2] = packed >> 4
    out += base
    if exceptions.numel():
        words = exceptions.reshape(-1).view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        pos = words & 0xFFFFFF
        if int(pos[-1]) >= rows * cols:
            raise ValueError(f"NVFP4-CSF exception position {int(pos[-1])} outside {rows}x{cols}")
        out.view(-1)[pos] = (words >> 24).to(torch.int32)
    if int(out.max()) > 255:
        raise ValueError("NVFP4-CSF base + offset overflows a scale byte")
    return out.to(torch.uint8).view(torch.float8_e4m3fn)


class CsfPairBuffer:
    """Pairs up the two streams of each compressed scale as they arrive (in any order,
    possibly from different shards) and hands back the decoded ``weight_scale``."""

    def __init__(self) -> None:
        self._pending: dict[str, dict[str, torch.Tensor]] = {}

    @staticmethod
    def stem(name: str) -> str | None:
        """``...weight_scale`` for a CSF stream key, else None."""
        for suffix in CSF_SUFFIXES:
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return None

    def add(self, name: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor] | None:
        stem = self.stem(name)
        assert stem is not None, name
        parts = self._pending.setdefault(stem, {})
        parts[name[len(stem):]] = tensor
        if len(parts) < 2:
            return None
        del self._pending[stem]
        return stem, decode_csf_scales(parts[CSF_FIXED_SUFFIX], parts[CSF_EXCEPTIONS_SUFFIX])

    def assert_empty(self) -> None:
        if self._pending:
            raise ValueError(f"NVFP4-CSF scales missing a stream: {sorted(self._pending)[:4]}")


__all__ = ["CSF_SUFFIXES", "CsfPairBuffer", "csf_geometry", "decode_csf_scales"]

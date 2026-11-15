"""Round-to-nearest NVFP4 quantization of bf16 expert weights at load.

Produces the checkpoint-native layout the NVFP4 expert kernels read (see
:func:`freetoken.kernel.triton.nvfp4_dequant.dequant_nvfp4`): ``packed [E, N, K/2]`` uint8 with
two E2M1 codes per byte (low nibble first), ``scale [E, N, K/16]`` fp8-e4m3 per 16 inputs and
``global [E, N]`` fp32 per output row, ``w ~= e2m1 * scale * global``. The row global puts the
row's largest block scale at the top of the e4m3 range; each 16-block then takes the e4m3 scale
nearest ``amax / 6`` and its values round to the nearest E2M1 point.
"""

from __future__ import annotations

import torch

# E2M1 magnitudes and the midpoints between neighbours (the rounding thresholds)
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_E4M3_MAX = 448.0


def quantize_nvfp4(w: torch.Tensor, experts_per_chunk: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``w`` [E, N, K] (K % 16 == 0) -> ``(packed, scale, global)``; chunked over experts so the
    fp32 transient stays small."""
    E, N, K = w.shape
    assert K % 16 == 0, "NVFP4 blocks are 16 inputs wide"
    dev = w.device
    packed = torch.empty((E, N, K // 2), dtype=torch.uint8, device=dev)
    scale = torch.empty((E, N, K // 16), dtype=torch.float8_e4m3fn, device=dev)
    glob = torch.empty((E, N), dtype=torch.float32, device=dev)
    mids = torch.tensor(_MIDPOINTS, dtype=torch.float32, device=dev)
    for e0 in range(0, E, experts_per_chunk):
        x = w[e0 : e0 + experts_per_chunk].float().view(-1, N, K // 16, 16)
        block_amax = x.abs().amax(dim=-1)  # [e, N, K/16]
        g = (block_amax.amax(dim=-1) / (6.0 * _E4M3_MAX)).clamp(min=1e-30)  # [e, N]
        s = (block_amax / 6.0 / g[..., None]).clamp(max=_E4M3_MAX).to(torch.float8_e4m3fn)
        step = s.float() * g[..., None]  # the dequant multiplier of each block
        q = x / torch.where(step > 0, step, torch.ones_like(step))[..., None]
        code = torch.bucketize(q.abs().clamp(max=6.0), mids).to(torch.uint8)  # 0..7
        code = code | ((q < 0).to(torch.uint8) << 3)
        code = code.view(-1, N, K // 2, 2)
        packed[e0 : e0 + experts_per_chunk] = code[..., 0] | (code[..., 1] << 4)
        scale[e0 : e0 + experts_per_chunk] = s
        glob[e0 : e0 + experts_per_chunk] = g
    return packed, scale, glob


__all__ = ["quantize_nvfp4"]

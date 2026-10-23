"""bf16 Linear: one kernel (torch), no scheme."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F

from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import LinearKernel, LinearMethod

# Single-token bf16 GEMV on the Triton streaming kernel instead of cuBLAS's split-K
# tensor-core GEMMs (decode is weight-bandwidth bound); FREETOKEN_TRITON_GEMV=0 for A/B.
_TRITON_GEMV = os.getenv("FREETOKEN_TRITON_GEMV", "1") != "0"


class TorchLinearKernel(LinearKernel):
    name = "torch"

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        # an fp32 activation stream (DeepSeek-V4's compressors) upcasts the bf16 weight on the fly, as the reference does
        w, b = layer.weight, layer.bias
        if w.dtype != x.dtype:
            w = w.to(x.dtype)
            b = b.to(x.dtype) if b is not None else None
        if (
            _TRITON_GEMV and b is None and x.dim() == 2 and x.shape[0] == 1 and x.is_cuda
            and x.dtype == torch.bfloat16 and x.is_contiguous() and w.stride(1) == 1
        ):
            from freetoken.kernel.triton.gemv import gemv

            return gemv(x, w)
        return F.linear(x, w, b)


class OnlineFp8LinearKernel(LinearKernel):
    """A bf16 checkpoint weight quantized to fp8-e4m3 per output row at load and run W8A16:
    half the weight bytes for the bandwidth-bound single-token GEMMs of the MTP draft head
    (only drafts are affected; the target verifies every one). Opt-in per layer via
    ``layer.online_fp8``."""

    name = "online-fp8"
    _ROWS = 4096  # quantize in row blocks: no full fp32 copy of a large weight

    def finalize(self, layer: Any) -> None:
        w = layer.weight
        q = torch.empty(w.shape, dtype=torch.float8_e4m3fn, device=w.device)
        scale = torch.empty(w.shape[0], dtype=torch.float32, device=w.device)
        for r in range(0, w.shape[0], self._ROWS):
            blk = w[r : r + self._ROWS].float()
            s = (blk.abs().amax(dim=1) / 448.0).clamp(min=1e-12)
            q[r : r + self._ROWS] = (blk / s[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            scale[r : r + self._ROWS] = s
        layer.weight = q
        layer.weight_scale = scale

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

        return fp8_pertensor_linear(x, layer.weight, layer.weight_scale, layer.bias)


class OnlineMxfp8LinearKernel(LinearKernel):
    """A bf16 weight that came from an MXFP8 checkpoint (e4m3 x 2^k per 32 inputs, dequantized
    exactly at load) re-encoded to MXFP8 at load: half the bytes for the bandwidth-bound decode
    GEMVs. Lossless -- :func:`mxfp8_encode_exact` refuses any weight whose bf16 values do not
    round-trip -- so serving is bit-identical in the weights to the bf16 path."""

    name = "online-mxfp8"

    def __init__(self, weight: torch.Tensor, codes: torch.Tensor) -> None:
        self._enc = (weight, codes)

    def finalize(self, layer: Any) -> None:
        layer.weight, layer.weight_scale_inv = self._enc
        del self._enc

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2 and x.shape[0] == 1 and layer.bias is None and x.is_contiguous() and x.dtype == torch.bfloat16:
            from freetoken.kernel.triton.gemv import gemv_mxfp8

            return gemv_mxfp8(x, layer.weight, layer.weight_scale_inv)
        from freetoken.kernel.triton.mxfp8_linear import mxfp8_linear

        return mxfp8_linear(x, layer.weight, layer.weight_scale_inv, layer.bias)


def mxfp8_encode_exact(w: torch.Tensor, rows: int = 2048) -> tuple[torch.Tensor, torch.Tensor] | None:
    """bf16 [N, K] -> (e4m3 [N, K], e8m0 codes [N, K/32]) when that dequantizes back to ``w``
    bit for bit, else None. Each 32-block takes the smallest power-of-two scale that keeps its
    max |w| <= 448: a block that came from MXFP8 is ``q * 2^k`` with q on the e4m3 grid, and
    for any smaller-or-equal k the rescaled q stays on the grid, so the encoding is exact."""
    N, K = w.shape
    if K % 32 or w.dtype != torch.bfloat16:
        return None
    q = torch.empty((N, K), dtype=torch.float8_e4m3fn, device=w.device)
    codes = torch.empty((N, K // 32), dtype=torch.uint8, device=w.device)
    for r in range(0, N, rows):
        blk = w[r : r + rows].float().view(-1, K // 32, 32)
        amax = blk.abs().amax(dim=2)
        k = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
        k = torch.where(amax / torch.exp2(k) > 448.0, k + 1, k)
        k = torch.where((amax / torch.exp2(k - 1) <= 448.0) & (k > -127), k - 1, k)
        k = torch.where(amax == 0, torch.zeros_like(k), k)
        if bool(((k + 127) < 0).any()) or bool(((k + 127) > 254).any()):
            return None
        enc = (blk / torch.exp2(k).unsqueeze(-1)).to(torch.float8_e4m3fn)
        back = (enc.float() * torch.exp2(k).unsqueeze(-1)).to(torch.bfloat16)
        if not torch.equal(back.view(-1, K), w[r : r + rows]):
            return None
        q[r : r + rows] = enc.view(-1, K)
        codes[r : r + rows] = (k + 127).to(torch.uint8)
    return q, codes


def mark_online_mxfp8(root: Any, skip: frozenset = frozenset()) -> tuple[int, int]:
    """Re-encode every bf16 CUDA linear under ``root`` whose weight round-trips through MXFP8
    exactly (see OnlineMxfp8LinearKernel); the rest stay bf16. ``skip`` names attributes whose
    subtree is left alone. Must run before ``finalize_quant``. Returns (encoded, kept bf16)."""
    from freetoken.layers.base import BaseOP

    seen: set[int] = set()
    stack = [root]
    done = kept = 0
    while stack:
        op = stack.pop()
        if id(op) in seen:
            continue
        seen.add(id(op))
        w = getattr(op, "weight", None)
        if (isinstance(getattr(op, "quant_method", None), UnquantizedLinearMethod) and isinstance(w, torch.Tensor)
                and w.is_cuda and w.dim() == 2):
            enc = mxfp8_encode_exact(w)
            if enc is None:
                kept += 1
            else:
                op.online_mxfp8 = enc
                done += 1
        for name, value in vars(op).items():
            if name in skip:
                continue
            if isinstance(value, BaseOP):
                stack.append(value)
            elif isinstance(value, list):
                stack.extend(v for v in value if isinstance(v, BaseOP))
    return done, kept


def mark_online_fp8(root: Any, skip: frozenset = frozenset()) -> int:
    """Flag every bf16 linear under ``root`` for load-time fp8 quantization (see
    OnlineFp8LinearKernel); ``skip`` names attributes whose subtree stays bf16 (routers).
    Must run before ``finalize_quant``. Returns the number of flagged layers."""
    from freetoken.layers.base import BaseOP

    seen: set[int] = set()
    stack = [root]
    count = 0
    while stack:
        op = stack.pop()
        if id(op) in seen:
            continue
        seen.add(id(op))
        if isinstance(getattr(op, "quant_method", None), UnquantizedLinearMethod) and getattr(op, "weight", None) is not None:
            op.online_fp8 = True
            count += 1
        for name, value in vars(op).items():
            if name in skip:
                continue
            if isinstance(value, BaseOP):
                stack.append(value)
            elif isinstance(value, list):
                stack.extend(v for v in value if isinstance(v, BaseOP))
    return count


@register_method(QuantKind.NONE, LayerKind.LINEAR)
class UnquantizedLinearMethod(LinearMethod):
    candidates = (TorchLinearKernel,)

    def create_weights(self, layer: Any) -> None:
        g = self.cfg
        layer.weight = torch.empty(g.out_features, g.in_features)

    def finalize(self, layer: Any) -> None:
        enc = getattr(layer, "online_mxfp8", None)
        if enc is not None:
            self.kernel = OnlineMxfp8LinearKernel(*enc)
            del layer.online_mxfp8
        elif getattr(layer, "online_fp8", False):
            self.kernel = OnlineFp8LinearKernel()
        self.kernel.finalize(layer)

"""Embedding gather from a pinned host table (``VocabParallelEmbedding.move_to_host``).

Registered host memory is device-addressable on Linux, so the kernel reads the rows it
needs straight over PCIe; ids outside this rank's vocab shard produce zero rows, which the
TP all-reduce fills in from the owning rank.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _host_embedding_kernel(ids_ptr, w_ptr, out_ptr, start, length, D: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    row = tl.load(ids_ptr + t).to(tl.int64) - start
    ok = (row >= 0) & (row < length)
    cols = tl.arange(0, BLOCK)
    for c in range(0, D, BLOCK):
        m = c + cols < D
        v = tl.load(w_ptr + row * D + c + cols, mask=ok & m, other=0.0)
        tl.store(out_ptr + t.to(tl.int64) * D + c + cols, v, mask=m)


def host_embedding(ids: torch.Tensor, weight: torch.Tensor, start: int, length: int) -> torch.Tensor:
    T, D = ids.numel(), weight.shape[1]
    out = torch.empty((T, D), dtype=weight.dtype, device=ids.device)
    if T:
        _host_embedding_kernel[(T,)](ids, weight, out, start, length, D=D, BLOCK=1024)
    return out

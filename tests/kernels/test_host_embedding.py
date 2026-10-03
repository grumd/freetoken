# Copyright (c) 2026 FreeToken contributors
# kernel/triton/host_embedding.py: rows gathered over PCIe from a pinned host vocab shard,
# zero rows for ids owned by another TP rank.
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("start", [0, 1000])
def test_host_gather_matches_shard_lookup(start):
    from freetoken.kernel.pinned import alloc_pinned_tensor
    from freetoken.kernel.triton.host_embedding import host_embedding

    V, D = 1000, 2560
    host = alloc_pinned_tensor(V, D, dtype=torch.bfloat16)
    host.copy_(torch.randn(V, D, dtype=torch.bfloat16))
    ids = torch.tensor([start, start + 5, start + V - 1, start + V, max(start - 1, 0) + 2 * V, 7], device="cuda")
    out = host_embedding(ids, host, start, V).cpu()
    for i, t in enumerate(ids.tolist()):
        expect = host[t - start] if start <= t < start + V else torch.zeros(D, dtype=torch.bfloat16)
        assert torch.equal(out[i], expect), i

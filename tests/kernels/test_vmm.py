"""CUDA-VMM elastic tensors: chunk math, remapping keeps live bytes, zero-on-map, pool accounting."""

import pytest
import torch

from freetoken.kernel.vmm import ChunkPool, ElasticTensor, chunk_ranges, count_chunks

MiB = 1 << 20


def test_chunk_ranges_merge_and_round_outward():
    c = 2 * MiB
    assert chunk_ranges([(0, 1)], c) == [(0, 1)]
    assert chunk_ranges([(c - 1, c + 1)], c) == [(0, 2)]
    # touching and overlapping spans merge; empty ones vanish
    assert chunk_ranges([(5 * c, 6 * c), (0, c), (c, 2 * c), (7 * c, 7 * c)], c) == [(0, 2), (5, 6)]
    assert count_chunks([(0, 3 * c + 1), (10 * c, 10 * c + 5)], c) == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_elastic_tensor_remap_keeps_mapped_bytes_and_zeroes_new_chunks():
    pool = ChunkPool(torch.device("cuda", torch.cuda.current_device()))
    c = pool.chunk_bytes
    rows = 8 * c // 4096
    pool.create(4)
    t = ElasticTensor(pool, (rows, 1024), torch.float32, [(0, 2 * c)], zero_on_map=True)
    try:
        assert t.num_mapped == 2 and pool.num_free == 2
        x = t.tensor
        head = rows // 4  # the first two chunks
        assert int(x[:head].count_nonzero()) == 0
        x[:head] = torch.arange(head * 1024, device="cuda", dtype=torch.float32).view(head, 1024)
        before = x[:head].clone()
        # grow to the 4th chunk only, then shrink back: chunk 0..1 stay intact, chunk 3 is zero
        assert t.set_required([(0, 2 * c), (3 * c, 4 * c)]) == (1, 0)
        x[3 * rows // 8 : rows // 2].fill_(7)
        assert t.set_required([(0, 2 * c)]) == (0, 1)
        assert t.set_required([(0, 2 * c), (3 * c, 4 * c)]) == (1, 0)
        torch.cuda.synchronize()
        assert torch.equal(x[:head], before)
        assert int(x[3 * rows // 8 : rows // 2].count_nonzero()) == 0
        assert pool.num_free == 1 and t.num_mapped == 3
        with pytest.raises(RuntimeError, match="exhausted"):
            t.set_required([(0, 8 * c)])
    finally:
        t.close()
    assert pool.num_free == pool.num_handles == 4
    pool.release(4)
    assert pool.num_handles == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_elastic_tensors_trade_handles_through_one_pool():
    pool = ChunkPool(torch.device("cuda", torch.cuda.current_device()))
    c = pool.chunk_bytes
    pool.create(3)
    a = ElasticTensor(pool, (4 * c,), torch.uint8, [(0, 2 * c)])
    b = ElasticTensor(pool, (4 * c,), torch.uint8, [(0, c)])
    try:
        a.tensor[: 2 * c].fill_(1)
        b.tensor[:c].fill_(2)
        a.set_required([(0, c)])  # hands one chunk back...
        b.set_required([(0, 2 * c)])  # ...which b can now map
        b.tensor[c : 2 * c].fill_(3)
        torch.cuda.synchronize()
        assert int(a.tensor[:c].sum()) == c and int(b.tensor[:c].sum()) == 2 * c
        assert int(b.tensor[c : 2 * c].sum()) == 3 * c
        assert pool.num_free == 0
    finally:
        a.close()
        b.close()
        pool.release(pool.num_free)

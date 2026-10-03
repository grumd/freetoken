# Copyright (c) 2026 FreeToken contributors
# distributed/host_allreduce.py: the 2-rank shared-host-memory all-reduce. Both ranks must
# agree bit for bit with a plain sum (a + b == b + a), eagerly and under CUDA-graph replay,
# where the device-side sequence counter alternates the data slots.
from __future__ import annotations

import os
import socket

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="needs 2 GPUs"
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, port: int, q) -> None:
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=2)
    from freetoken.distributed.host_allreduce import HostAllReduce

    har = HostAllReduce(rank, dist.group.WORLD)
    ok = True
    for n in (8, 2560, 4 * 2560, 32768):
        for step in range(20):
            g = torch.Generator().manual_seed(1000 * n + step)
            both = [torch.randn(n, generator=g).to(torch.bfloat16) for _ in range(2)]
            want = (both[0].float() + both[1].float()).to(torch.bfloat16)
            x = both[rank].cuda()
            har.all_reduce(x)
            ok &= torch.equal(x.cpu(), want)
    # graph replay: same captured kernel, fresh data per replay
    buf = torch.empty(2560, dtype=torch.bfloat16, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        har.all_reduce(buf.zero_())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        har.all_reduce(buf)
        har.all_reduce(buf)
    for step in range(10):
        g = torch.Generator().manual_seed(step)
        both = [torch.randint(-4, 4, (2560,), generator=g).to(torch.bfloat16) for _ in range(2)]
        buf.copy_(both[rank].cuda())
        graph.replay()
        torch.cuda.synchronize()
        ok &= torch.equal(buf.cpu(), 2 * (both[0] + both[1]))
    q.put((rank, bool(ok)))
    dist.destroy_process_group()


def test_host_allreduce_two_ranks():
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_worker, args=(r, port, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=300)
    results = dict(q.get(timeout=5) for _ in range(2))
    assert results == {0: True, 1: True}

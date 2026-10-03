"""Two-rank all-reduce through shared pinned host memory (``kernel/csrc/jit/host_allreduce.cuh``).

Without GPU peer access NCCL routes a 2-rank all-reduce through host memory anyway, and for
the 5 KB decode reductions its latency (ring LL kernel plus the copies in and out of its
buffer) is ~15 us. One kernel that writes the local vector into a mapped ``/dev/shm``
segment and reads the peer's back is a third of that. Small bf16 tensors only; everything
else stays on NCCL.
"""

from __future__ import annotations

import ctypes
import functools
import mmap
import os
import uuid

import torch
import torch.distributed as dist

_FLAGS_BYTES = 4096
_CUDA_HOST_REGISTER_PORTABLE = 1
_CUDA_HOST_REGISTER_MAPPED = 2


@functools.cache
def _module():
    from freetoken.kernel.utils import load_jit

    return load_jit(
        "host_allreduce",
        cuda_files=["host_allreduce.cuh"],
        cuda_wrappers=[("host_allreduce_bf16", "host_allreduce_bf16")],
    )


class HostAllReduce:
    def __init__(self, rank: int, cpu_group: dist.ProcessGroup, slot_bytes: int = 64 * 1024) -> None:
        self.rank = rank
        self.slot_bytes = slot_bytes
        size = _FLAGS_BYTES + 4 * slot_bytes  # 2 ranks x 2 parities
        name = [f"/dev/shm/freetoken-har-{uuid.uuid4().hex}" if rank == 0 else None]
        dist.broadcast_object_list(name, src=0, group=cpu_group)
        path = name[0]
        if rank == 0:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(fd, size)
        dist.barrier(group=cpu_group)
        if rank != 0:
            fd = os.open(path, os.O_RDWR)
        self._mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)
        dist.barrier(group=cpu_group)
        if rank == 0:
            os.unlink(path)  # both ranks hold the mapping; nothing left behind on exit
        self._base = ctypes.addressof(ctypes.c_char.from_buffer(self._mm))
        err = torch.cuda.cudart().cudaHostRegister(
            self._base, size, _CUDA_HOST_REGISTER_PORTABLE | _CUDA_HOST_REGISTER_MAPPED
        )
        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister of the all-reduce segment failed: {err}")
        self._seq = torch.zeros(1, dtype=torch.int64, device="cuda")
        _module()
        dist.barrier(group=cpu_group)

    def eligible(self, x: torch.Tensor) -> bool:
        n = x.numel()
        return (
            x.dtype == torch.bfloat16 and x.is_cuda and x.is_contiguous()
            and n % 8 == 0 and n * 2 <= self.slot_bytes
        )

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        _module().host_allreduce_bf16(
            x.view(-1).view(torch.int16), self._seq, self._base, self._base + _FLAGS_BYTES,
            self.slot_bytes, self.rank,
        )
        return x

"""Two-rank all-reduce over GPU peer access (``kernel/csrc/jit/p2p_allreduce.cuh``).

With peer access NCCL's symmetric LL kernel takes ~9.5 us per 5 KB decode reduction inside a
CUDA graph (copy into its window, kernel, copy out). Pushing LL lines straight into the peer's
VRAM inbox and polling the local one takes ~2.7 us, and degrades least while expert copies
share the PCIe links. Small bf16 tensors only; everything else stays on NCCL.
"""

from __future__ import annotations

import ctypes
import functools

import torch
import torch.distributed as dist

_CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


@functools.cache
def _cudart() -> ctypes.CDLL:
    # the runtime torch already loaded, so the handles belong to torch's context
    lib = ctypes.CDLL(f"libcudart.so.{torch.version.cuda.split('.')[0]}")
    lib.cudaIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), _IpcHandle, ctypes.c_uint]
    return lib


@functools.cache
def _module():
    from freetoken.kernel.utils import load_jit

    return load_jit(
        "p2p_allreduce",
        cuda_files=["p2p_allreduce.cuh"],
        cuda_wrappers=[
            ("p2p_allreduce_bf16", "p2p_allreduce_bf16"),
            ("p2p_allgather_bf16", "p2p_allgather_bf16"),
        ],
    )


class P2PAllReduce:
    def __init__(self, rank: int, cpu_group: dist.ProcessGroup, slot_bytes: int = 128 * 1024) -> None:
        self.rank = rank
        self.slot_bytes = slot_bytes
        size = 2 * slot_bytes  # 2 parities
        rt = _cudart()
        # cudaMalloc rather than a torch tensor: an IPC handle names the whole allocation, and
        # the caching allocator may hand out a sub-block of a larger one
        ptr = ctypes.c_void_p()
        _check(rt.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size)), "cudaMalloc")
        _check(rt.cudaMemset(ptr, 0, ctypes.c_size_t(size)), "cudaMemset")
        handle = _IpcHandle()
        _check(rt.cudaIpcGetMemHandle(ctypes.byref(handle), ptr), "cudaIpcGetMemHandle")
        handles = [None, None]
        # string_at, not handle.reserved: a c_char array field reads back cut at its first NUL
        dist.all_gather_object(handles, ctypes.string_at(ctypes.addressof(handle), 64), group=cpu_group)
        peer_handle = _IpcHandle()
        ctypes.memmove(ctypes.addressof(peer_handle), handles[1 - rank], 64)
        peer = ctypes.c_void_p()
        _check(
            rt.cudaIpcOpenMemHandle(ctypes.byref(peer), peer_handle, _CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS),
            "cudaIpcOpenMemHandle",
        )
        self._mine, self._peer = ptr.value, peer.value
        self._seq = torch.zeros(1, dtype=torch.int64, device="cuda")
        _module()
        torch.cuda.synchronize()
        dist.barrier(group=cpu_group)

    def eligible(self, x: torch.Tensor) -> bool:
        n = x.numel()
        return (
            x.dtype == torch.bfloat16 and x.is_cuda and x.is_contiguous()
            and n % 2 == 0 and n * 4 <= self.slot_bytes
        )

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        _module().p2p_allreduce_bf16(
            x.view(-1).view(torch.int16), self._seq, self._mine, self._peer, self.slot_bytes,
        )
        return x

    def all_gather(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """``out`` = rank 0's ``x`` then rank 1's (bf16, contiguous, ``eligible`` sizes);
        ``x`` may be this rank's half of ``out``."""
        _module().p2p_allgather_bf16(
            x.view(-1).view(torch.int16), out.view(-1).view(torch.int16), self._seq, self.rank,
            self._mine, self._peer, self.slot_bytes,
        )
        return out


def _check(err: int, what: str) -> None:
    if int(err) != 0:
        raise RuntimeError(f"{what} failed for the P2P all-reduce: cudaError {err}")

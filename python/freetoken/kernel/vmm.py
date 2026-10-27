"""Device tensors whose physical backing grows and shrinks at a fixed virtual address.

Each :class:`ElasticTensor` reserves a virtual range sized for its largest shape and maps
only the granularity-sized chunks that currently hold live bytes. Every elastic tensor on a
device draws its chunks from one :class:`ChunkPool`, so moving VRAM from one tensor to
another is an unmap plus a map of the same physical handles. Pointers never change, which
keeps CUDA graphs and device-side pointer descriptors captured against the tensors valid.

The caller owns ordering: nothing may read or write a chunk while it is being unmapped, so
synchronize the device before :meth:`ElasticTensor.set_required` releases anything.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import torch

Interval = Tuple[int, int]


def _drv():
    from cuda.bindings import driver

    return driver


def _check(res):
    drv = _drv()
    err = res[0] if isinstance(res, tuple) else res
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver VMM call failed: {err}")
    return res[1] if isinstance(res, tuple) and len(res) > 1 else None


class ChunkPool:
    """Physical device-memory handles of one fixed size, shared by a device's elastic tensors."""

    def __init__(self, device: torch.device, chunk_bytes: int | None = None) -> None:
        drv = _drv()
        torch.cuda.init()
        self.device = device
        index = device.index if device.index is not None else torch.cuda.current_device()
        prop = drv.CUmemAllocationProp()
        prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = index
        self._prop = prop
        access = drv.CUmemAccessDesc()
        access.location = prop.location
        access.flags = drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self._access = access
        granularity = int(_check(drv.cuMemGetAllocationGranularity(
            prop, drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        )))
        if chunk_bytes is None:
            chunk_bytes = granularity
        if chunk_bytes % granularity:
            raise ValueError(f"chunk_bytes {chunk_bytes} is not a multiple of the {granularity} B granularity")
        self.chunk_bytes = chunk_bytes
        self._free: List[object] = []
        self.num_handles = 0

    @property
    def num_free(self) -> int:
        return len(self._free)

    def create(self, n: int) -> None:
        """Allocate ``n`` more physical chunks from the driver into the free list."""
        drv = _drv()
        for _ in range(n):
            self._free.append(_check(drv.cuMemCreate(self.chunk_bytes, self._prop, 0)))
            self.num_handles += 1

    def release(self, n: int) -> None:
        """Return ``n`` unmapped chunks to the driver."""
        if n > len(self._free):
            raise RuntimeError(f"cannot release {n} chunks, only {len(self._free)} are unmapped")
        drv = _drv()
        for _ in range(n):
            _check(drv.cuMemRelease(self._free.pop()))
            self.num_handles -= 1

    def take(self):
        if not self._free:
            raise RuntimeError("chunk pool exhausted: the elastic budget accounting is off")
        return self._free.pop()

    def give(self, handle) -> None:
        self._free.append(handle)


class _CudaSpan:
    def __init__(self, ptr: int, nbytes: int) -> None:
        self.__cuda_array_interface__ = {
            "shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3,
        }


def chunk_ranges(intervals: Iterable[Interval], chunk_bytes: int) -> List[Tuple[int, int]]:
    """Merged ``[first, last)`` chunk-index ranges covering the byte ``intervals``."""
    spans = sorted(
        (start // chunk_bytes, -(-end // chunk_bytes)) for start, end in intervals if end > start
    )
    merged: List[Tuple[int, int]] = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def count_chunks(intervals: Iterable[Interval], chunk_bytes: int) -> int:
    return sum(hi - lo for lo, hi in chunk_ranges(intervals, chunk_bytes))


class ElasticTensor:
    """A fixed-address device tensor of ``shape`` whose mapped chunks follow :meth:`set_required`.

    ``tensor`` is a normal torch view over the whole range; touching an unmapped chunk is an
    illegal-address fault, so callers must keep every live byte inside the required intervals.
    """

    def __init__(
        self,
        pool: ChunkPool,
        shape: Sequence[int],
        dtype: torch.dtype,
        initial: Iterable[Interval],
        *,
        zero_on_map: bool = False,
    ) -> None:
        drv = _drv()
        self.pool = pool
        self.shape = tuple(int(s) for s in shape)
        self.dtype = dtype
        self.nbytes = math.prod(self.shape) * dtype.itemsize
        chunk = pool.chunk_bytes
        self.va_bytes = max(chunk, -(-self.nbytes // chunk) * chunk)
        self.va = int(_check(drv.cuMemAddressReserve(self.va_bytes, chunk, 0, 0)))
        self.zero_on_map = zero_on_map
        self._mapped: dict[int, object] = {}
        self._bytes: torch.Tensor | None = None
        self.tensor: torch.Tensor | None = None
        # torch derives the tensor's device from the pointer, so the first chunk must be live
        # before the view exists.
        self.set_required(list(initial))
        if 0 not in self._mapped:
            raise ValueError("an elastic tensor must keep its first chunk mapped")
        self._bytes = torch.as_tensor(_CudaSpan(self.va, self.va_bytes), device=pool.device)
        self.tensor = self._bytes[: self.nbytes].view(dtype).view(self.shape)
        if zero_on_map:
            for lo, hi in self._runs(sorted(self._mapped)):
                self._bytes[lo * chunk : hi * chunk].zero_()

    @property
    def num_mapped(self) -> int:
        return len(self._mapped)

    def count_required(self, intervals: Iterable[Interval]) -> int:
        return count_chunks(intervals, self.pool.chunk_bytes)

    def set_required(self, intervals: Iterable[Interval]) -> Tuple[int, int]:
        """Map exactly the chunks covering ``intervals``; returns (mapped, unmapped) counts.

        Unmapping happens first so the freed handles can back this call's new chunks.
        """
        drv = _drv()
        chunk = self.pool.chunk_bytes
        want = set()
        for lo, hi in chunk_ranges(intervals, chunk):
            if hi * chunk > self.va_bytes:
                raise ValueError(f"interval past the reserved range ({hi * chunk} > {self.va_bytes})")
            want.update(range(lo, hi))
        drop = sorted(set(self._mapped) - want)
        for lo, hi in self._runs(drop):
            _check(drv.cuMemUnmap(self.va + lo * chunk, (hi - lo) * chunk))
            for i in range(lo, hi):
                self.pool.give(self._mapped.pop(i))
        add = sorted(want - set(self._mapped))
        for i in add:
            handle = self.pool.take()
            _check(drv.cuMemMap(self.va + i * chunk, chunk, 0, handle, 0))
            self._mapped[i] = handle
        for lo, hi in self._runs(add):
            _check(drv.cuMemSetAccess(self.va + lo * chunk, (hi - lo) * chunk, [self.pool._access], 1))
            if self.zero_on_map and self._bytes is not None:
                self._bytes[lo * chunk : hi * chunk].zero_()
        return len(add), len(drop)

    def close(self) -> None:
        """Unmap everything and free the virtual range (tests; the engine keeps them for life)."""
        drv = _drv()
        torch.cuda.synchronize(self.pool.device)
        self.tensor = None
        self._bytes = None
        chunk = self.pool.chunk_bytes
        for lo, hi in self._runs(sorted(self._mapped)):
            _check(drv.cuMemUnmap(self.va + lo * chunk, (hi - lo) * chunk))
        for handle in self._mapped.values():
            self.pool.give(handle)
        self._mapped.clear()
        _check(drv.cuMemAddressFree(self.va, self.va_bytes))

    @staticmethod
    def _runs(indices: Sequence[int]) -> List[Tuple[int, int]]:
        runs: List[Tuple[int, int]] = []
        for i in indices:
            if runs and runs[-1][1] == i:
                runs[-1] = (runs[-1][0], i + 1)
            else:
                runs.append((i, i + 1))
        return runs


__all__ = ["ChunkPool", "ElasticTensor", "chunk_ranges", "count_chunks"]

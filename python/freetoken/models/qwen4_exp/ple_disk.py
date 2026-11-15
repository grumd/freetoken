"""Disk-backed PLE table (--ple-backend disk): the C++ store hashes n-gram windows and batch-reads rows from the checkpoint's fp8 shard tensors into pinned staging; the captured ``lookup`` is a fixed-shape H2D copy + dequant.

Hash windows are pure functions of ``req.input_ids`` + ``device_len`` (prefix hits, restores and COW forks need no bookkeeping); the decode input token lives device-side under overlap scheduling and is read back here.
"""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

import safetensors
import torch

from freetoken.core import Batch
from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.utils import init_logger

from .weight import (
    _PLE_GLOBAL_SUFFIX,
    _PLE_NVFP4_ROWS_RE,
    _PLE_SCALE_SUFFIX,
    _PLE_SHARD_RE,
    _PLE_SRC_DTYPES,
    _ple_table_files,
    _safetensors_header,
    ple_source_folder,
)

_IO_URING_ENV = "FREETOKEN_PLE_IO_URING"
_SYNC_ENV = "FREETOKEN_PLE_SYNC"  # auto | wait | gate

logger = init_logger(__name__)


def _context(ids: torch.Tensor, position: int, eos: int) -> list[int]:
    """The two token ids before ``position``; eos pads past the start."""
    return [int(ids[position - 2]) if position >= 2 else eos,
            int(ids[position - 1]) if position >= 1 else eos]


@dataclass(frozen=True)
class PleRowSource:
    """On-disk row layout: equal extents, row i of an extent at ``base + i * row_stride`` (a repacked flat file is one extent with its own stride).

    Rows are fp8-e4m3 (one byte/element, with a scalar ``scale``), unquantized bf16/f32
    (``scale == 1.0``) or interleaved NVFP4 (``src_dtype == "NVFP4_ROWS"``: e2m1 codes then
    e4m3 per-16 block scales, ``scale`` the fp32 global); ``row_bytes`` is the byte width of a
    row, ``elem_bytes`` its element size (0 for NVFP4)."""

    paths: list[str]
    extent_file: list[int]
    extent_base: list[int]
    rows_per_extent: int
    row_bytes: int
    row_stride: int
    scale: float
    src_dtype: str
    elem_bytes: int

    @property
    def total_rows(self) -> int:
        return len(self.extent_base) * self.rows_per_extent

    @property
    def head_dim(self) -> int:
        # an NVFP4 row of D values is D/2 code bytes + D/16 scale bytes
        return self.row_bytes * 16 // 9 if self.src_dtype == "NVFP4_ROWS" else self.row_bytes // self.elem_bytes


def source_from_safetensors(folder: str) -> PleRowSource:
    """Map the checkpoint's ``ngram_embedding.shard_<i>`` tensors in place: one extent per shard, no copy."""
    rows = cols = 0
    scale: torch.Tensor | None = None
    src_dtype: str | None = None
    elem_bytes = 0
    paths: list[str] = []
    path_idx: dict[str, int] = {}
    shards: dict[int, tuple[int, int]] = {}
    for path in _ple_table_files(folder):
        header, base = _safetensors_header(path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key.endswith((_PLE_SCALE_SUFFIX, _PLE_GLOBAL_SUFFIX)):
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    scale = f.get_tensor(key).reshape(())
                continue
            match = _PLE_NVFP4_ROWS_RE.search(key)
            if match is not None:
                meta = dict(meta, dtype="NVFP4_ROWS")
            else:
                match = _PLE_SHARD_RE.search(key)
            if match is None:
                continue
            if meta["dtype"] == "U8":
                raise ValueError(
                    "the NVFP4 PLE table must be repacked into interleaved rows for the disk "
                    "backend: convert the checkpoint with `ft checkpoint` and serve the FTW"
                )
            if meta["dtype"] == "NVFP4_ROWS":
                if src_dtype is None:
                    src_dtype, elem_bytes = "NVFP4_ROWS", 0
            elif meta["dtype"] not in _PLE_SRC_DTYPES:
                raise ValueError(
                    f"PLE shard {key} has dtype {meta['dtype']}, expected one of "
                    f"{sorted(_PLE_SRC_DTYPES)}"
                )
            if src_dtype is None:
                src_dtype, elem_bytes = meta["dtype"], _PLE_SRC_DTYPES[meta["dtype"]][1]
            elif meta["dtype"] != src_dtype:
                raise ValueError(
                    f"PLE table mixes {src_dtype} and {meta['dtype']} shards — unsupported layout"
                )
            if rows and tuple(meta["shape"]) != (rows, cols):
                raise ValueError(f"PLE shard {key} is {meta['shape']}, expected {[rows, cols]}")
            rows, cols = meta["shape"]
            if path not in path_idx:
                path_idx[path] = len(paths)
                paths.append(path)
            idx = int(match.group("shard"))
            if idx in shards:
                raise ValueError(f"duplicate PLE shard {idx} in {path}")
            shards[idx] = (path_idx[path], base + meta["data_offsets"][0])
    if sorted(shards) != list(range(len(shards))) or not shards:
        raise ValueError(f"PLE shard indices are not contiguous 0..N-1: {sorted(shards)[:8]}")
    if src_dtype is None:
        raise ValueError("PLE table has no shard tensors")
    if scale is None:
        # Unquantized (bf16/f32) tables carry no scale; fp8 tables must.
        if src_dtype in ("F8_E4M3", "NVFP4_ROWS"):
            raise ValueError("PLE table has no weight_scale")
        scale = torch.tensor(1.0)
    order = [shards[i] for i in range(len(shards))]
    row_bytes = cols * elem_bytes if elem_bytes else cols
    return PleRowSource(
        paths, [f for f, _ in order], [b for _, b in order], rows, row_bytes, row_bytes,
        float(scale), src_dtype, elem_bytes,
    )


def resolve_row_source(folder: str) -> PleRowSource:
    """Pick the row source for a checkpoint; the seam where a repacked format would plug in.

    ``ple_source_folder`` lets an FTW without PLE side tables fall back to its source."""
    return source_from_safetensors(ple_source_folder(folder))


class DiskRowTable:
    """``PLETableBackend`` whose rows are read from disk per fill (--ple-backend disk)."""

    def __init__(
        self,
        source: PleRowSource,
        hash_constants: dict,
        *,
        max_graph_rows: int = 256,
        max_extend_tokens: int = 8192,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        from freetoken.kernel import _ple_store

        self.num_rows = source.total_rows
        self.head_dim = source.head_dim  # elements, not bytes
        self._src_dtype = source.src_dtype
        self.dtype = dtype
        self.heads = int(hash_constants["num_ngram_heads"])
        self.scale = source.scale
        self.eos_token_id = int(hash_constants["eos_token_id"])
        sizes = [int(x) for x in hash_constants["per_head_vocab_sizes"]]
        offsets = [int(x) for x in hash_constants["per_head_offsets"]]
        need = max(o + s for o, s in zip(offsets, sizes))
        if need > source.total_rows:
            raise ValueError(
                f"PLE row source holds {source.total_rows} rows but the hash addresses {need}; incomplete checkpoint?"
            )
        self._store = _ple_store.PleStore(
            paths=list(source.paths),
            extent_file=list(source.extent_file),
            extent_base=list(source.extent_base),
            rows_per_extent=source.rows_per_extent,
            row_bytes=source.row_bytes,
            row_stride=source.row_stride,
            multipliers=[int(x) for x in hash_constants["layer_multipliers"]],
            head_vocab_sizes=sizes,
            head_offsets=offsets,
            eos_token_id=self.eos_token_id,
            use_io_uring=os.getenv(_IO_URING_ENV, "1") != "0",
        )
        self._device = torch.device("cuda", torch.cuda.current_device())
        self._token_bytes = self.heads * source.row_bytes
        # allocated up front: pinned alloc inside stream capture is illegal; one replay consumes it at a time
        self._graph_pinned = alloc_pinned_tensor(max_graph_rows * self._token_bytes, dtype=torch.uint8)
        self._graph_pinned.zero_()  # padded decode lanes read whatever sits here
        # outlives any one graph: a cache rebuild recaptures against the same pointer
        self._graph_dev = torch.empty(
            max_graph_rows * self._token_bytes, dtype=torch.uint8, device=self._device
        )
        eager_bytes = max_extend_tokens * self._token_bytes
        self._eager_pinned = alloc_pinned_tensor(eager_bytes, dtype=torch.uint8)
        self._eager_pinned.zero_()  # the warmup prefill stages nothing and reads whatever sits here
        self._eager_dev = torch.empty(eager_bytes, dtype=torch.uint8, device=self._device)
        # probe picks flag-sync (graph WAITs at the consume, host fills then signals) or launch-gating
        self._wait_sync = self._probe_wait_sync(os.getenv(_SYNC_ENV, "auto"))
        # one flag for all graphs: the readback event orders a fill after the previous graph, so signals never overlap
        self._flag = alloc_pinned_tensor(1, dtype=torch.int64)
        self._flag.zero_()
        self._token_readback = alloc_pinned_tensor(max_graph_rows, dtype=torch.int32)
        self._readback_event = torch.cuda.Event()
        # Prefill rows are read on a worker thread (I/O releases the GIL): the forward launches at
        # once and the PLE layer WAITs for the fill's sequence number; while a chunk computes, the
        # worker reads the request's next chunk into a second buffer.
        self._async = self._wait_sync
        if self._async:
            self._io = ThreadPoolExecutor(1, thread_name_prefix="ple-io")
            self._eager_tokens = max_extend_tokens
            self._ahead_pinned = alloc_pinned_tensor(eager_bytes, dtype=torch.uint8)
            self._eager_seq = alloc_pinned_tensor(1, dtype=torch.int64)
            self._eager_seq.zero_()
            # the worker publishes through numpy: inference mode is per thread, and the model is
            # built under it, so a torch write from the worker would be an illegal inplace update
            self._eager_seq_np = self._eager_seq.numpy()
            self._seq = 0  # last fill submitted
            self._wait_seq = 0  # what the next eager lookup waits for (0: rows already staged)
            self._cur: Future | None = None
            self._ahead: tuple[int, int, int, Future] | None = None  # (uid, start, end, fill)
        sync = "wait-sync" if self._wait_sync else "launch-gating"
        logger.info_rank0(f"PLE disk backend: {self._store.io_backend()}, {sync}")

    def _probe_wait_sync(self, mode: str) -> bool:
        from freetoken.kernel import _ple_store

        if mode == "gate":
            return False
        scratch = alloc_pinned_tensor(1, dtype=torch.int64)
        scratch.zero_()
        stream = torch.cuda.current_stream(self._device)
        ok = (
            _ple_store.memop_write(stream.cuda_stream, scratch.data_ptr(), 7) == 0
            and _ple_store.memop_wait_geq(stream.cuda_stream, scratch.data_ptr(), 7) == 0
        )
        if ok:
            stream.synchronize()
            ok = int(scratch[0]) == 7
        if mode == "wait" and not ok:
            raise RuntimeError("FREETOKEN_PLE_SYNC=wait but stream memops are unavailable")
        return ok

    # ---------------- host side (engine thread, before the forward launches) ----------------

    def _stage_flush(self, runs: Sequence[torch.Tensor], pinned: torch.Tensor, seq: int) -> None:
        """Worker thread: one batched read of ``runs`` into ``pinned``, then publish ``seq``."""
        try:
            offset = 0
            for run in runs:
                self._store.stage(run.data_ptr(), run.numel() - 2, pinned.data_ptr() + offset * self._token_bytes)
                offset += run.numel() - 2
            self._store.flush(0)
        finally:
            if seq:
                # also on failure: the stream must not hang; the join after the launch raises
                self._eager_seq_np[0] = seq

    def _join(self) -> None:
        """Wait for the worker's fills (the store is single-threaded); drops an unused read-ahead."""
        cur, self._cur = self._cur, None
        ahead, self._ahead = self._ahead, None
        if ahead is not None:
            ahead[3].result()
        if cur is not None:
            cur.result()

    def _fill_prefill_async(self, batch: Batch, runs: list[torch.Tensor]) -> None:
        reqs = batch.padded_reqs
        req = reqs[0] if len(reqs) == 1 else None
        ahead, self._ahead = self._ahead, None
        if ahead is not None:
            ahead[3].result()
        if (ahead is not None and req is not None and req.uid == ahead[0]
                and req.cached_len == ahead[1] and req.device_len <= ahead[2]):
            # read while the previous chunk computed: it becomes this chunk's staging
            self._eager_pinned, self._ahead_pinned = self._ahead_pinned, self._eager_pinned
            self._wait_seq = 0
        else:
            cur, self._cur = self._cur, None
            if cur is not None:
                cur.result()
            self._seq += 1
            self._cur = self._io.submit(self._stage_flush, runs, self._eager_pinned, self._seq)
            self._wait_seq = self._seq
        prompt = getattr(req, "prompt_ids", None)
        if prompt is not None and req.device_len < len(prompt):
            # the next chunk is this one's size (a prompt's chunks differ only in the last); 1024 covers rounding
            chunk = -(-(req.device_len - req.cached_len) // 1024) * 1024
            start = req.device_len
            end = min(len(prompt), start + min(chunk, self._eager_tokens))
            run = torch.cat((torch.tensor(_context(prompt, start, self.eos_token_id), dtype=torch.int64),
                             prompt[start:end].to(torch.int64)))
            self._ahead = (req.uid, start, end, self._io.submit(self._stage_flush, [run], self._ahead_pinned, 0))

    def fill(self, runs: Sequence[torch.Tensor], *, graph: bool) -> None:
        """Stage per-request token runs (two context ids, then the new tokens) in batch order."""
        if self._async:
            self._join()
        pinned = self._graph_pinned if graph else self._eager_pinned
        offset = 0
        for run in runs:
            self._store.stage(run.data_ptr(), run.numel() - 2, pinned.data_ptr() + offset * self._token_bytes)
            offset += run.numel() - 2
        self._store.flush(self._flag.data_ptr() if graph and self._wait_sync else 0)

    def host_fill_batch(self, batch: Batch, use_graph: bool):
        """Stage this batch's rows; returns the post-dispatch fill callable under flag-sync, else None."""
        eos = self.eos_token_id
        if batch.is_decode:
            reqs = list(batch.reqs)
            if use_graph and self._wait_sync:
                bs = batch.padded_size
                self._token_readback[:bs].copy_(batch.input_ids, non_blocking=True)
                self._readback_event.record(torch.cuda.current_stream(self._device))

                def _complete() -> None:
                    try:
                        self._readback_event.synchronize()
                        tokens = self._token_readback[:bs].to(torch.int64).tolist()
                        runs = [torch.tensor([*_context(r.input_ids, r.device_len - 1, eos), t], dtype=torch.int64)
                                for r, t in zip(reqs, tokens)]
                        self.fill(runs, graph=True)
                    except BaseException:
                        from freetoken.kernel import _ple_store

                        # unblock the stream before surfacing; the step's output is discarded
                        _ple_store.signal_flag(self._flag.data_ptr())
                        raise

                return _complete
            # launch-gating: this D2H is the step's readback and orders the fill after sampling
            tokens = batch.input_ids.to("cpu").to(torch.int64).tolist()
            runs = [torch.tensor([*_context(r.input_ids, r.device_len - 1, eos), t], dtype=torch.int64)
                    for r, t in zip(reqs, tokens)]
            self.fill(runs, graph=use_graph)
            return None
        runs = [
            torch.cat((
                torch.tensor(_context(req.input_ids, req.cached_len, eos), dtype=torch.int64),
                req.input_ids[req.cached_len : req.device_len].to(torch.int64),
            ))
            for req in batch.padded_reqs
        ]
        if self._async and not use_graph:
            self._fill_prefill_async(batch, runs)
            return self._finish_prefill_fill
        self.fill(runs, graph=False)
        return None

    def _finish_prefill_fill(self) -> None:
        """After the launch: surface a failed read (the forward's output is discarded)."""
        cur, self._cur = self._cur, None
        if cur is not None:
            cur.result()

    @contextmanager
    def forward_host_ctx(self, batch: Batch, use_graph: bool):
        """Around one dispatch: stage on enter, run the deferred fill+signal on exit."""
        deferred = self.host_fill_batch(batch, use_graph)
        yield
        # no try/finally: a failed launch leaves no WAIT pending, so the fill must not run
        if deferred is not None:
            deferred()

    # ---------------- device side (PLETableBackend protocol) ----------------

    def lookup(self, row_ids: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        rows = row_ids.shape[0]
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and self._wait_sync:
            from freetoken.kernel import _ple_store

            _ple_store.memop_wait_reset(
                torch.cuda.current_stream(self._device).cuda_stream, self._flag.data_ptr()
            )
        if not capturing and self._async and self._wait_seq:
            from freetoken.kernel import _ple_store

            stream = torch.cuda.current_stream(self._device).cuda_stream
            if _ple_store.memop_wait_geq(stream, self._eager_seq.data_ptr(), self._wait_seq) != 0:
                self._finish_prefill_fill()  # no stream WAIT: block the host instead
            self._wait_seq = 0
        pinned, dev = (
            (self._graph_pinned, self._graph_dev) if capturing else (self._eager_pinned, self._eager_dev)
        )
        nbytes = rows * self._token_bytes
        dev[:nbytes].copy_(pinned[:nbytes], non_blocking=True)
        if self._src_dtype == "NVFP4_ROWS":
            from freetoken.kernel.triton.ple import nvfp4_rows_dequant

            values = nvfp4_rows_dequant(dev[:nbytes], self.head_dim, self.scale).to(self.dtype)
        else:
            values = dev[:nbytes].view(_PLE_SRC_DTYPES[self._src_dtype][0]).to(self.dtype)
            if self.scale != 1.0:
                values = values * self.scale
        values = values.view(*row_ids.shape[:-1], -1)
        if out is None:
            return values
        out.copy_(values)
        return out

    def prefetch(self, row_ids: torch.Tensor) -> None:
        return None

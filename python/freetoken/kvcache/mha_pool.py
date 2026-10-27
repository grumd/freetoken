from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, List, Sequence, Tuple

import torch
from freetoken.distributed import get_tp_info
from freetoken.utils import div_even

from .base import BaseKVCachePool

# Elastic VRAM (--elastic-kv) installs an allocator around pool construction:
# ``alloc(role, shape, dtype, intervals)`` returns the tensor for one page-scaled tier, where
# ``intervals(num_pages)`` gives the byte ranges that must stay backed while pages
# ``[0, num_pages)`` are usable. None = plain torch allocations.
_TIER_ALLOC: Callable | None = None


@contextmanager
def elastic_tier_allocator(alloc: Callable):
    global _TIER_ALLOC
    _TIER_ALLOC = alloc
    try:
        yield
    finally:
        _TIER_ALLOC = None


def paged_run_intervals(
    num_runs: int, run_bytes: int, page_bytes: int, top_bytes: int
) -> Callable[[int], List[Tuple[int, int]]]:
    """Backed ranges of a tier laid out as ``num_runs`` equal runs of ``run_bytes``, each holding
    one page's ``page_bytes`` per page from offset 0, and ending in ``top_bytes`` that are always
    live (the dummy page, scratch rows)."""

    def intervals(num_pages: int) -> List[Tuple[int, int]]:
        out = []
        for r in range(num_runs):
            base = r * run_bytes
            out.append((base, base + num_pages * page_bytes))
            out.append((base + run_bytes - top_bytes, base + run_bytes))
        return out

    return intervals


def alloc_tier(role, shape, dtype, device, intervals, *, zero: bool = False) -> torch.Tensor:
    if _TIER_ALLOC is not None:
        return _TIER_ALLOC(role, shape, dtype, intervals)
    alloc = torch.zeros if zero else torch.empty
    return alloc(shape, device=device, dtype=dtype)


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.

    ``layer_ids`` lets the pool back only a *subset* of the model's layers while
    callers keep indexing by their global ``layer_id``. Hybrid models (e.g. the
    Qwen3.5 GatedDeltaNet/full-attention stack) interleave linear-attention layers
    that hold no paged KV; passing the full-attention layer ids here allocates one
    storage slab per KV layer (not per model layer) and remaps the global id to its
    dense slot, avoiding a multiple-x over-allocation of unused slabs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_ids: Sequence[int] | None = None,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._num_layers = num_layers
        if layer_ids is None:
            num_storage_layers = num_layers
            self._layer_map: list[int] | None = None
        else:
            num_storage_layers = len(layer_ids)
            # The map is indexed by GLOBAL layer id: an MTP draft block's attention slot
            # continues after the main stack (id >= num_layers), so size it to the highest
            # mapped id, not just num_layers.
            map_size = max(num_layers, max(layer_ids) + 1) if layer_ids else num_layers
            layer_map = [-1] * map_size
            for dense, global_id in enumerate(layer_ids):
                if global_id < 0 or global_id >= map_size:
                    raise ValueError(f"KV layer id {global_id} outside [0, {map_size})")
                layer_map[global_id] = dense
            self._layer_map = layer_map
            self._num_layers = map_size
        self.elastic = _TIER_ALLOC is not None
        page_bytes = page_size * local_kv_heads * head_dim * dtype.itemsize
        self._kv_buffer = alloc_tier(
            "kv",
            (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim),
            dtype,
            device,
            # the last page of every (k/v, layer) run is the dummy page padded rows point at
            paged_run_intervals(2 * num_storage_layers, num_pages * page_bytes, page_bytes, page_bytes),
        )
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def rebuild(self, num_pages: int) -> None:
        """Reallocate the KV buffer for ``num_pages`` pages IN PLACE.

        Geometry (storage layers, page_size, kv heads, head_dim) is taken from the
        existing buffer; only the page count changes. Views and ``_storage_shape`` are
        refreshed. Object identity is preserved so cached backend references stay valid.
        """
        if self.elastic:
            raise RuntimeError("an elastic KV pool resizes by remapping pages, not rebuild")
        _, num_storage_layers, _old_pages, page_size, local_kv_heads, head_dim = self._kv_buffer.shape
        dtype = self._kv_buffer.dtype
        device = self._device
        self._k_buffer = None
        self._v_buffer = None
        self._kv_buffer = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        self._kv_buffer = torch.empty(
            (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token

        per_token = sum(
            spec_kv_bytes_per_token(spec, config)
            for spec in config.model_config.kv_cache_group_specs()
            if not spec.is_swa
        )
        return per_token * config.page_size, 0, config.page_size, 0

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        self.rebuild(num_pages + 1)  # +1 for the dummy page (matches create_kvcache_pool)

    def unit_bytes(self) -> tuple[int, int]:
        buf = self._kv_buffer
        tokens = int(buf.shape[2]) * int(buf.shape[3])
        return int(buf.numel() * buf.element_size()) // tokens, 0

    def _dense(self, layer_id: int) -> int:
        if self._layer_map is None:
            return layer_id
        dense = self._layer_map[layer_id]
        if dense < 0:
            raise KeyError(f"layer {layer_id} has no paged KV storage")
        return dense

    def page_views(self) -> List[torch.Tensor]:
        """Every page-scaled tier viewed with the page on dim 0, so gathering or scattering by
        page index moves all of a page's bytes (elastic compaction, the prefix RAM tier)."""
        return [self._kv_buffer.permute(2, 0, 1, 3, 4, 5)]

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[self._dense(index)]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[self._dense(index)]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        from freetoken.kernel import store_cache

        dense = self._dense(layer_id)
        store_cache(
            k_cache=self._k_buffer[dense].view(self._storage_shape),
            v_cache=self._v_buffer[dense].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers

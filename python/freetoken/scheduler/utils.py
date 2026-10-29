from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from freetoken.core import SamplingParams

    from .mm import MMInputs
    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None
    mm: MMInputs | None = None
    error: str | None = None  # set when admission failed for good; the request is dropped

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]


def cache_key(req, n: int) -> torch.Tensor:
    """The radix-tree key for the first ``n`` tokens of ``req`` (prompt and generated)."""
    mm = getattr(req, "mm", None)
    if mm is None:
        return req.input_ids[:n]
    key = mm.key
    if n <= key.numel():
        return key[:n]
    return torch.cat([key, req.input_ids[key.numel() : n]])

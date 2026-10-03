"""Decode routing log (FREETOKEN_ROUTE_LOG=<path>, diagnostics only).

Records, per decode step and MoE layer, the routed expert ids and the top ids layer L+1's
router gives on layer L's MoE input (a lookahead prediction). Fixed-shape device writes, so
it is captured into the decode CUDA graph; the engine flushes it to ``<path>.rank<r>.pt``.
"""

from __future__ import annotations

import os

import torch

_PATH = os.environ.get("FREETOKEN_ROUTE_LOG")
PRED_K = 32


class RouteLog:
    def __init__(self, num_layers: int, top_k: int, device: torch.device, cap: int = 4096) -> None:
        self.num_layers, self.cap = num_layers, cap
        self.actual = torch.full((cap * num_layers, top_k), -1, dtype=torch.int32, device=device)
        self.pred = torch.full((cap * num_layers, PRED_K), -1, dtype=torch.int32, device=device)
        self.step = torch.zeros(1, dtype=torch.int64, device=device)

    def _row(self, layer_id: int) -> torch.Tensor:
        return (self.step * self.num_layers + layer_id).clamp_(max=self.cap * self.num_layers - 1)

    def record_actual(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        if topk_ids.shape[0] != 1:
            return
        self.actual.index_copy_(0, self._row(layer_id), topk_ids.view(1, -1).to(torch.int32))
        if layer_id == self.num_layers - 1:
            self.step.add_(1)

    def record_pred(self, layer_id: int, logits: torch.Tensor) -> None:
        if logits.shape[0] != 1:
            return
        ids = torch.topk(logits.float(), PRED_K, dim=-1).indices.to(torch.int32)
        self.pred.index_copy_(0, self._row(layer_id + 1), ids.view(1, -1))

    def flush(self, rank: int) -> None:
        n = int(self.step.item())
        rows = min(n, self.cap) * self.num_layers
        torch.save(
            {"steps": n, "actual": self.actual[:rows].cpu(), "pred": self.pred[:rows].cpu()},
            f"{_PATH}.rank{rank}.pt",
        )


_LOG: RouteLog | None = None


def get(num_layers: int, top_k: int, device: torch.device) -> RouteLog | None:
    global _LOG
    if not _PATH:
        return None
    if _LOG is None:
        _LOG = RouteLog(num_layers, top_k, device)
    return _LOG


def current() -> RouteLog | None:
    return _LOG

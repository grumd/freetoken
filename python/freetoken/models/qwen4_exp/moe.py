from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen4ExpMoE(Qwen3_5MoE):
    """Qwen3_5MoE with the shared-expert gate on triton instead of gemv + sigmoid + mul + add.

    Same weights, same state dict. The gate reduction stays ahead of the routed experts, which may write into ``hidden_states`` in place.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # one TP all_reduce for routed + gated shared (see Qwen3_5MoE.forward)
        router_logits = self.gate.forward(hidden_states)
        self._log_route_lookahead(hidden_states, num_tokens)
        shared = self.shared_expert.forward(hidden_states, reduce=False)
        gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
        routed = self.experts.forward(
            hidden_states=hidden_states, router_logits=router_logits, reduce=False
        )
        out = shared_gate_mul_add(routed, shared, gate)
        return self.experts._maybe_all_reduce(out).view(num_tokens, hidden_dim)


    def _log_route_lookahead(self, hidden_states: torch.Tensor, num_tokens: int) -> None:
        """FREETOKEN_ROUTE_LOG diagnostics: the next layer's router scored on this layer's MoE
        input, logged beside the real routing (lookahead-prefetch analysis)."""
        next_gate = getattr(self, "_next_gate", None)
        if next_gate is None or num_tokens != 1:
            return
        from freetoken.core import get_global_ctx
        from freetoken.moe import route_log

        gate, num_layers, layer_id = next_gate
        log = route_log.get(num_layers, self.experts.top_k, hidden_states.device)
        if log is not None and get_global_ctx().batch.is_decode:
            log.record_pred(layer_id, gate.forward(hidden_states))


__all__ = ["Qwen4ExpMoE"]

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

    def fuse_decode_front(self) -> None:
        """Single-token steps run the router, shared-expert gate_up and shared-expert gate
        (all over the MoE input) as one GEMV launch over the three weights in place."""
        gu = self.shared_expert.gate_up_proj
        parts = (self.gate.weight, gu.weight, self.shared_expert_gate.weight)
        if not all(p.is_cuda and p.dtype == torch.bfloat16 and p.dim() == 2 and p.stride(1) == 1 for p in parts):
            return
        if type(gu.quant_method).__name__ != "UnquantizedLinearMethod":
            return
        self._front = True

    def _decode_front(self, hidden_states: torch.Tensor):
        """(router logits, shared-expert output before the gate, raw gate logit column)."""
        from freetoken.kernel.triton.gemv import gemv3
        from freetoken.layers import silu_and_mul

        r, gu = self.gate.weight, self.shared_expert.gate_up_proj.weight
        out = gemv3(hidden_states, r, gu, self.shared_expert_gate.weight)
        a, b = r.shape[0], gu.shape[0]
        shared = self.shared_expert.down_proj.forward(silu_and_mul(out[:, a : a + b]), reduce=False)
        return out[:, :a], shared, out[:, a + b]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if num_tokens == 1 and getattr(self, "_front", False) and hidden_states.is_contiguous():
            router_logits, shared, gate_logit = self._decode_front(hidden_states)
            self._log_route_lookahead(hidden_states, num_tokens)
            routed = self.experts.forward(
                hidden_states=hidden_states, router_logits=router_logits, reduce=False
            )
            out = shared_gate_mul_add(routed, shared, gate_logit, gate_is_logit=True)
            return self.experts._maybe_all_reduce(out).view(num_tokens, hidden_dim)
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

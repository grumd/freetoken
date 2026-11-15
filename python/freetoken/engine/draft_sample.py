"""Sampled MTP drafts: each draft step draws its token from the draft head's own truncated
distribution ``q`` (the request's temperature, top-k and top-p over the draft logits'
``DRAFT_K`` best candidates) and keeps ``q`` for the speculative acceptance
(:func:`freetoken.engine.sample.speculative_accept`). Accepting a sampled draft with
``min(1, p/q)`` keeps the output distribution exactly the target's, and accepts far more
often than a greedy draft (``p(argmax q)``) when the distributions are spread.

Everything here is a device op of fixed shape, so a captured draft step can sample; the
uniforms are drawn outside the graph from the torch generator, which every TP rank advances
identically, and the candidates are gathered across ranks, so all ranks draw the same token.
"""

from __future__ import annotations

import torch

DRAFT_K = 20  # candidates per draft step (the checkpoint's top_k)


def candidates(logits: torch.Tensor, ids: torch.Tensor, comm=None, tp_size: int = 1):
    """Best ``DRAFT_K`` of this rank's ``logits`` [T, n] (column c is token ``ids[c]``; -inf
    columns never win) merged across TP ranks: ``(values [T, K] descending, ids [T, K] int32)``."""
    vals, idx = logits.float().topk(DRAFT_K, dim=-1)
    cid = ids[idx].to(torch.int32)
    if tp_size == 1:
        return vals, cid
    T = vals.shape[0]
    pairs = torch.stack([vals, cid.view(torch.float32)], dim=-1)  # [T, K, 2]
    # as raw bytes: the pynccl transport has no fp32 all_gather, uint8 is byte-transparent
    g = comm.all_gather(pairs.view(torch.uint8)).view(torch.float32).view(tp_size, T, DRAFT_K, 2)
    g = g.permute(1, 0, 2, 3).reshape(T, tp_size * DRAFT_K, 2)
    vals, j = g[..., 0].topk(DRAFT_K, dim=-1)
    cid = g[..., 1].contiguous().view(torch.int32).gather(1, j)
    return vals, cid


def sample(vals: torch.Tensor, cid: torch.Tensor, temp: torch.Tensor, top_p: torch.Tensor,
           top_k: torch.Tensor, u: torch.Tensor):
    """Draw one token per row from the candidates' truncated distribution (the target's
    sampler order: temperature, top-k, softmax, top-p). ``temp <= 0`` rows are greedy (q is
    one-hot on the best candidate). ``u`` [T, K] uniforms. Returns ``(token [T] int32, q [T, K])``."""
    K = vals.shape[1]
    greedy = temp <= 0
    logit = vals / torch.where(greedy, torch.ones_like(temp), temp)[:, None]
    rank = torch.arange(K, device=vals.device)
    logit = logit.masked_fill(rank[None, :] >= top_k[:, None], float("-inf"))
    q = torch.softmax(logit, dim=-1)
    q = q.masked_fill((q.cumsum(dim=-1) - q) >= top_p[:, None], 0.0)  # keep the top-p prefix
    q = q / q.sum(dim=-1, keepdim=True)
    q = torch.where(greedy[:, None], (rank == 0).to(q.dtype)[None, :].expand_as(q), q)
    gumbel = -torch.log(-torch.log(u.clamp(1e-12, 1.0 - 1e-7)))
    score = torch.where(q > 0, q.log() + gumbel, torch.full_like(q, float("-inf")))
    choice = score.argmax(dim=-1, keepdim=True)
    return cid.gather(1, choice).squeeze(1), q


__all__ = ["DRAFT_K", "candidates", "sample"]

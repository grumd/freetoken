# Copyright (c) 2026 FreeToken contributors
# Sampled MTP drafts (engine/draft_sample.py) and the speculative acceptance
# (engine/sample.py speculative_accept): the emitted tokens follow the target distribution
# exactly, whatever the proposal.
from __future__ import annotations

import pytest
import torch


def _first_token_freq(P, drafts_fn, cand, trials):
    from freetoken.engine.sample import speculative_accept

    counts = torch.zeros(P.shape[1])
    accepted = 0
    for _ in range(trials):
        d = drafts_fn()
        a, tok = speculative_accept(P, d, *cand(d))
        first = int(d[0]) if int(a) >= 1 else int(tok)
        counts[first] += 1
        accepted += int(a) >= 1
    return counts / trials, accepted / trials


@pytest.mark.parametrize("proposal", ["sampled", "point"])
def test_speculative_accept_emits_the_target_distribution(proposal):
    torch.manual_seed(0)
    V, k = 6, 2
    P = torch.softmax(torch.randn(k + 1, V) * 1.5, dim=-1)
    q = torch.softmax(torch.randn(k, V) * 1.5, dim=-1)
    cid = torch.arange(V, dtype=torch.int32).expand(k, V).contiguous()
    if proposal == "sampled":
        drafts_fn = lambda: torch.multinomial(q, 1).squeeze(1)  # noqa: E731
        cand = lambda d: (cid, q)  # noqa: E731
        expect_accept = torch.minimum(P[0], q[0]).sum().item()
    else:
        drafts_fn = lambda: q.argmax(dim=1)  # noqa: E731
        cand = lambda d: (None, None)  # noqa: E731
        expect_accept = P[0, q[0].argmax()].item()
    freq, acc = _first_token_freq(P, drafts_fn, cand, 20000)
    assert (freq - P[0]).abs().max().item() < 0.015, (freq, P[0])
    assert abs(acc - expect_accept) < 0.015, (acc, expect_accept)


def test_speculative_accept_sparse_proposal_with_unused_slots():
    """q given over a few candidate ids (unused slots -1), as the draft graph emits it."""
    from freetoken.engine.sample import speculative_accept

    torch.manual_seed(1)
    V = 8
    P = torch.softmax(torch.randn(2, V), dim=-1)
    cid = torch.tensor([[2, 5, -1]], dtype=torch.int32)
    q = torch.tensor([[0.7, 0.3, 0.0]])
    counts = torch.zeros(V)
    for _ in range(20000):
        d = cid[0, torch.multinomial(q[0], 1)].long()
        a, tok = speculative_accept(P, d, cid, q)
        counts[int(d[0]) if int(a) else int(tok)] += 1
    assert (counts / 20000 - P[0]).abs().max().item() < 0.015


def test_draft_sample_truncation_and_greedy_rows():
    from freetoken.engine.draft_sample import sample

    torch.manual_seed(0)
    vals = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0]]).repeat(4, 1)
    cid = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.int32).repeat(4, 1)
    temp = torch.tensor([0.0, 1.0, 1.0, 0.5])
    top_p = torch.tensor([1.0, 1.0, 0.7, 1.0])
    top_k = torch.tensor([5, 2, 5, 5])
    u = torch.rand(4, 5)
    d, q = sample(vals, cid, temp, top_p, top_k, u)
    assert int(d[0]) == 10 and torch.equal(q[0], torch.tensor([1.0, 0, 0, 0, 0]))  # greedy
    assert torch.allclose(q[1, :2], torch.softmax(vals[1, :2], 0)) and not q[1, 2:].any()  # top-k 2
    p2 = torch.softmax(vals[2], 0)  # top-p 0.7: 0.665 < 0.7, so the second candidate stays
    assert torch.allclose(q[2, :2], p2[:2] / p2[:2].sum()) and not q[2, 2:].any()
    assert torch.allclose(q[3], torch.softmax(vals[3] / 0.5, 0))
    # empirical draws follow q
    n = 20000
    rows = sample(vals[3:4].expand(n, 5), cid[3:4].expand(n, 5), temp[3:4].expand(n), top_p[3:4].expand(n),
                  top_k[3:4].expand(n), torch.rand(n, 5))[0]
    freq = torch.bincount(rows.long() - 10, minlength=5).float() / n
    assert (freq - q[3]).abs().max().item() < 0.015


def test_candidates_single_rank_maps_ids():
    from freetoken.engine.draft_sample import DRAFT_K, candidates

    logits = torch.randn(3, 100)
    logits[:, 50:] = float("-inf")  # unused rows of a draft-vocab set
    ids = torch.arange(1000, 1100, dtype=torch.int32)
    vals, cid = candidates(logits, ids)
    ref_v, ref_i = logits.topk(DRAFT_K, dim=-1)
    assert torch.equal(vals, ref_v) and torch.equal(cid, (ref_i + 1000).to(torch.int32))

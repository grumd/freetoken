"""MTP acceptance: greedy match mapping.

Greedy acceptance is pure tensor math and a regression there silently corrupts spec output,
so it is unit-tested on CPU. The sampled rejection path needs the GPU sampling kernels
(``probs_from_logits``); its residual helper is shared with DSpark and covered in
``test_dspark_accept.py``.
"""

from __future__ import annotations

import torch

from freetoken.scheduler.mtp import MTPManager


class _Sampling:
    def __init__(self, greedy: bool) -> None:
        self.is_greedy = greedy


class _Req:
    def __init__(self, greedy: bool = True) -> None:
        self.sampling_params = _Sampling(greedy)


def _accept(preds: list[int], drafts: list[int], k: int):
    """One greedy request: drafts is the k draft ids, preds the target argmax at rows 0..k."""
    logits = torch.zeros(len(preds), 8)
    for j, p in enumerate(preds):
        logits[j, p] = 10.0
    am = logits.argmax(-1).view(1, k + 1)
    draft_mat = torch.tensor([drafts], dtype=torch.int64)
    return MTPManager._accept(
        object.__new__(MTPManager), [_Req()], logits, am, draft_mat, k, torch.device("cpu")
    )


def test_all_drafts_accepted_bonus_is_last_argmax():
    a, bonus = _accept([2, 3, 4, 6], [2, 3, 4], k=3)
    assert a == [3] and bonus == [6]


def test_first_draft_rejected_bonus_is_target_at_c():
    a, bonus = _accept([2, 3, 4, 6], [9, 3, 4], k=3)
    assert a == [0] and bonus == [2]


def test_second_draft_rejected():
    a, bonus = _accept([2, 3, 4, 6], [2, 9, 4], k=3)
    assert a == [1] and bonus == [3]


def test_third_draft_rejected():
    a, bonus = _accept([2, 3, 4, 6], [2, 3, 9], k=3)
    assert a == [2] and bonus == [4]

class _IdsReq:
    def __init__(self, ids: list[int]) -> None:
        self._ids_buf = torch.tensor(ids, dtype=torch.int32)


def test_verify_window_ids_reach_the_host():
    # a host-staged PLE fill reads input_ids[C : C+k+1]: the pending token and the drafts, not
    # the placeholders begin_round grows the buffer with
    r = _IdsReq([5, 6, 7, 0, 0, 0, 0, 0])
    anchor = torch.tensor([7], dtype=torch.int32)
    drafts = [torch.tensor([11]), torch.tensor([12]), torch.tensor([13])]
    MTPManager._stage_window_ids([r], [2], anchor, drafts)
    assert r._ids_buf.tolist() == [5, 6, 7, 11, 12, 13, 0, 0]

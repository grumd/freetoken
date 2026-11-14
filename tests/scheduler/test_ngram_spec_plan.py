"""--ngram-spec planning (scheduler/ngram_spec.py): which drafts a drained request verifies.

The window never commits past the next decode snapshot boundary or the output budget, and stops
at an image placeholder (never generated) or right after a tool-call opener (its anchor). Under
TP every rank plans the same windows from rank 0's synced inputs."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from freetoken.scheduler.ngram_draft import DraftPolicy
from freetoken.scheduler.ngram_spec import NgramSpecManager


class _Req(SimpleNamespace):
    __hash__ = object.__hash__  # requests are compared by identity, like the scheduler's
    __eq__ = object.__eq__


def _mgr(k=7, anchor=None, image=None, interval=64, mode="always", off_file=None, mode_file=None):
    m = NgramSpecManager.__new__(NgramSpecManager)
    m.sched = SimpleNamespace(
        cache_manager=SimpleNamespace(decode_snapshot_interval=interval), toolcall_anchor_id=anchor,
        tp_sync_values=None,
    )
    m.k, m.min_match = k, 3
    m._set_mode(mode)
    m._drafters, m._req_stats = {}, {}
    m._mode_file, m._off_file, m._mode_next, m._paused_flag, m._tp = mode_file, off_file, None, False, False
    m._image_token_id = image
    m.policy = DraftPolicy(k + 1)
    return m


def _req(ids, remain=100, uid=1):
    ids = torch.tensor(ids, dtype=torch.int32)
    n = ids.numel()
    return _Req(uid=uid, table_idx=0, aborted=False, input_ids=ids, device_len=n, cached_len=n - 1,
                remain_len=remain)


# [1..10] then [1, 2, 3] again: the lookup continues with 4, 5, 6, ...
_IDS = list(range(1, 11)) + [1, 2, 3]


def test_drafts_continue_the_repeat():
    _, drafts, match = _mgr().plan_one(_req(_IDS))
    assert drafts == [4, 5, 6, 7, 8, 9, 10] and match == 3


def test_window_stops_before_the_snapshot_boundary():
    # cached_len 12: the window may commit up to position 16 (interval 16) -> 3 drafts
    _, drafts, _ = _mgr(interval=16).plan_one(_req(_IDS))
    assert drafts == [4, 5, 6]


def test_window_respects_the_output_budget():
    _, drafts, _ = _mgr().plan_one(_req(_IDS, remain=3))
    assert drafts == [4, 5]
    assert _mgr().plan_one(_req(_IDS, remain=1)) is None


def test_window_stops_at_an_image_token_and_after_the_tool_call_opener():
    _, drafts, _ = _mgr(image=7).plan_one(_req(_IDS))
    assert drafts == [4, 5, 6]
    _, drafts, _ = _mgr(anchor=6).plan_one(_req(_IDS))
    assert drafts == [4, 5, 6]


def test_in_flight_request_is_not_planned():
    r = _req(_IDS)
    r.device_len += 1  # its next token is still on the GPU
    assert _mgr().plan_one(r) is None


def test_fixed_mode_pads_every_step_to_the_full_width():
    m = _mgr(k=4)
    m._set_mode("fixed")
    _, drafts, _ = m.plan_one(_req(list(range(1, 9))))  # no repeat
    assert drafts == [0, 0, 0, 0]
    m._set_mode("fixed-null")
    _, drafts, _ = m.plan_one(_req(_IDS))
    assert drafts == [0, 0, 0, 0]


def test_pending_tool_call_opener_takes_a_plain_step():
    # the opener is the pending token: a window would commit past its anchor snapshot
    assert _mgr(anchor=3).plan_one(_req(_IDS)) is None
    assert _mgr(anchor=99).plan_one(_req(_IDS)) is not None


# ---------------------------------------------------------------- TP: same plan on every rank


def _ranks(tmp_path=None, **kw):
    """Two ranks whose clocks disagree: rank 0 measures windows that pay, rank 1 ones that don't."""
    ranks = [_mgr(mode="adaptive", **kw), _mgr(mode="adaptive")]
    for t in range(1, 9):
        for _ in range(4):
            ranks[0].policy.observe_cost(t, 20.0 + 4.0 * (t - 1))
            ranks[1].policy.observe_cost(t, 20.0 + 40.0 * (t - 1))
    return ranks


def _requests():
    out, uid = [], 0
    for period in (4, 6, 9, 13):
        for reps in (2, 3, 5):
            base = list(range(100 + period, 100 + 2 * period))
            uid += 1
            out.append(_req((base * reps)[:-1], uid=uid))
    return out


def _plans(m, reqs):
    return [None if p is None else (p[1], p[2]) for p in (m.plan_one(r) for r in reqs)]


def test_tp_ranks_plan_the_same_windows_from_rank0_inputs():
    ranks = _ranks()
    reqs = _requests()
    for m in ranks:  # each rank on its own clock: the plans differ (the hang the sync prevents)
        m.sync(None)
    assert _plans(ranks[0], reqs) != _plans(ranks[1], reqs)
    ranks = _ranks()
    payload = ranks[0].sync_payload()
    assert len(payload) == ranks[0].sync_len
    for _ in range(3):  # probes are spent identically too
        for m in ranks:
            m.sync(payload)
        assert _plans(ranks[0], reqs) == _plans(ranks[1], reqs)
        cands = [m.candidates(reqs, in_flight=reqs[::2]) for m in ranks]
        assert [r.uid for r in cands[0]] == [r.uid for r in cands[1]]


def test_tp_diagnostics_files_are_read_on_rank0_only(tmp_path):
    off, mode = tmp_path / "off", tmp_path / "mode"
    rank0, rank1 = _mgr(off_file=str(off), mode_file=str(mode)), _mgr()
    off.touch()
    mode.write_text("fixed-null\n")
    payload = rank0.sync_payload()
    rank1.sync(payload)
    assert rank1._paused() and rank1._mode_next == "fixed-null"
    rank1.plan_one(_req(_IDS, uid=5))  # a new request's drafter takes the pending policy
    assert rank1.mode == "fixed-null"
    off.unlink()
    rank1.sync(rank0.sync_payload())
    assert not rank1._paused()


def test_rank1_keeps_its_plan_inputs_until_rank0_values_arrive():
    m = _mgr(mode="adaptive")
    m._tp = True
    m.policy.observe_cost(1, 10.0)
    m.policy.observe_cost(1, 10.0)
    m.begin_iteration()  # no broadcast yet: nothing of this rank's clock is adopted
    assert m.policy.cost_n[1] == 0
    m.sched.tp_sync_values = _mgr().sync_payload()
    m.begin_iteration()
    assert m.policy.cost_n[1] == 0 and m.policy.meas_n[1] == 1


def test_broadcast_count_carries_the_sync_values():
    import torch.distributed as dist

    from freetoken.scheduler.io import broadcast_count

    store, out = dist.HashStore(), {}

    def run(rank):
        pg = dist.ProcessGroupGloo(store, rank, 2)
        for extra_len in (0, 4):
            extra = [1.0, 0.0, 12.5, 3.0] if extra_len else None
            got = broadcast_count(pg, 7 if rank == 0 else None, extra if rank == 0 else None, extra_len)
            out[(rank, extra_len)] = got

    threads = [threading.Thread(target=run, args=(r,)) for r in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert out[(0, 0)] == out[(1, 0)] == (7, None)
    assert out[(0, 4)] == out[(1, 4)] == (7, [1.0, 0.0, 12.5, 3.0])


# ---------------------------------------------------------------- padding is never accepted


def _greedy(n):
    return [_Req(sampling_params=SimpleNamespace(is_greedy=True)) for _ in range(n)]


def test_padding_drafts_are_never_accepted_greedy():
    k, vocab = 3, 12
    drafts, real = [[4, 5, 6], [7, 7, 7]], [3, 1]  # request 1 padded with its last draft
    argmax = [[4, 5, 6, 9], [7, 7, 7, 2]]  # the target would "accept" request 1's padding
    logits = torch.zeros(2 * (k + 1), vocab)
    for i, row in enumerate(argmax):
        for j, tok in enumerate(row):
            logits[i * (k + 1) + j, tok] = 1.0
    a, bonus = _mgr()._accept(_greedy(2), logits, drafts, real, k, torch.device("cpu"))
    assert a == [3, 1] and bonus == [9, 7]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the sampling kernels need CUDA")
def test_padding_drafts_are_never_accepted_sampled():
    k, vocab, dev = 3, 32, torch.device("cuda")
    params = SimpleNamespace(is_greedy=False, temperature=1.0, top_k=-1, top_p=1.0)
    logits = torch.full((k + 1, vocab), -30.0, device=dev)
    logits[:, 7] = 30.0  # every position certain of token 7: a draft of 7 is always accepted
    a, bonus = _mgr()._accept([_Req(sampling_params=params)], logits, [[7, 7, 7]], [1], k, dev)
    assert a == [1] and bonus == [7]


# ---------------------------------------------------------------- which models / backends


def test_ngram_spec_needs_a_qwen4_exp_style_verify():
    from freetoken.engine.verify_graph import ngram_spec_unsupported, verify_store_attr

    f = lambda *a: None
    full = SimpleNamespace(mtp=None, forward_logits_all=f, select_verify_width=f, commit_mtp_verify=f)
    assert ngram_spec_unsupported(full, linear_state_pool=object()) is None
    gdn = SimpleNamespace(mtp=None, forward_logits_all=f, commit_mtp_verify=f)  # qwen3.5
    assert "per-width" in ngram_spec_unsupported(gdn, linear_state_pool=object())
    attn_only = SimpleNamespace()
    assert "all-position logits" in ngram_spec_unsupported(attn_only, linear_state_pool=None)
    with_mtp = SimpleNamespace(**{**vars(full), "mtp": object()})
    assert "MTP" in ngram_spec_unsupported(with_mtp, linear_state_pool=object())
    assert verify_store_attr(SimpleNamespace(_graph_verify={})) == "_graph_verify"
    assert verify_store_attr(SimpleNamespace(_verify_graph={})) == "_verify_graph"
    assert verify_store_attr(SimpleNamespace()) is None

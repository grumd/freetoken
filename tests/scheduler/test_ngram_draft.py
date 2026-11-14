"""Self-history n-gram draft combiner (port of llama.cpp's common_ngram_simple_draft)."""

from freetoken.scheduler.ngram_draft import ngram_draft


def test_finds_the_continuation_after_a_repeat():
    # The final [7,8,9] also occurred earlier, followed by [1,2,3].
    tokens = [0, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert ngram_draft(tokens, 3, 3) == [1, 2, 3]
    assert ngram_draft(tokens, 3, 2) == [1, 2]


def test_no_match_returns_empty():
    assert ngram_draft([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 3, 3) == []


def test_match_at_position_zero_is_ignored():
    # llama.cpp ignores position 0; the final pattern only matches there.
    tokens = [7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert ngram_draft(tokens, 3, 3) == []


def test_too_short_history_is_a_noop():
    assert ngram_draft([1, 2, 3], 3, 3) == []
    assert ngram_draft([], 3, 3) == []

# --------------------------------------------------------------------------------------
# SuffixDrafter / DraftPolicy (--ngram-spec)
# --------------------------------------------------------------------------------------

import random

from freetoken.scheduler.ngram_draft import DraftPolicy, SuffixDrafter


def _brute_propose(hist, max_k, min_match=3, max_match=64, ways=4):
    """Oracle: the WAYS most recent earlier ends of the final trigram, longest backward match
    (most recent on ties), continuation read from the history up to the current token."""
    cur = len(hist) - 1
    if len(hist) < 4 or max_k <= 0:
        return [], 0
    ends = [p for p in range(cur - 1, 1, -1) if hist[p - 2 : p + 1] == hist[cur - 2 : cur + 1]][:ways]
    best_end, best_len = -1, 0
    for p in ends:
        length = 0
        while length < max_match and length <= p and hist[p - length] == hist[cur - length]:
            length += 1
        if length > best_len:
            best_len, best_end = length, p
    if best_len < min_match:
        return [], 0
    return hist[best_end + 1 : min(cur + 1, best_end + 1 + max_k)], best_len


def test_suffix_drafter_continues_the_longest_match():
    d = SuffixDrafter()
    # [5,6,7] occurs twice; the older occurrence also matches the token before it (4)
    d.append([4, 5, 6, 7, 10, 11, 9, 5, 6, 7, 20, 21, 4, 5, 6, 7])
    assert d.propose(3) == ([10, 11, 9], 4)


def test_suffix_drafter_prefers_the_most_recent_on_ties():
    d = SuffixDrafter()
    d.append([1, 5, 6, 7, 10, 2, 5, 6, 7, 20, 3, 5, 6, 7])
    assert d.propose(2) == ([20, 3], 3)


def test_suffix_drafter_periodic_text_runs_into_the_suffix():
    d = SuffixDrafter()
    d.append([1, 2, 3, 1, 2, 3, 1, 2, 3])
    drafts, match = d.propose(5)
    assert drafts == [1, 2, 3] and match == 6


def test_suffix_drafter_min_match_and_no_match():
    d = SuffixDrafter(min_match=5)
    d.append([9, 1, 2, 3, 4, 8, 1, 2, 3])
    assert d.propose(3) == ([], 0)
    d = SuffixDrafter()
    d.append(list(range(20)))
    assert d.propose(3) == ([], 0)


def test_suffix_drafter_bulk_and_incremental_match_the_oracle():
    rng = random.Random(7)
    tokens = [rng.randrange(6) for _ in range(400)]
    bulk = SuffixDrafter()
    bulk.append(tokens[:150])  # bulk path (> 64 tokens)
    step = SuffixDrafter()
    step.append(tokens[:10])
    for t in tokens[10:150]:
        step.append([t])
    assert bulk.prev == step.prev and bulk.head == step.head
    for i in range(150, len(tokens)):
        bulk.append([tokens[i]]) if i % 3 else bulk.append(tokens[i : i + 1])
        assert bulk.propose(7) == _brute_propose(tokens[: i + 1], 7), i


def test_suffix_drafter_bulk_append_links_to_older_history():
    rng = random.Random(3)
    tokens = [rng.randrange(5) for _ in range(300)]
    d = SuffixDrafter()
    d.append(tokens[:80])
    d.append(tokens[80:200])  # a second bulk chunk must chain to the first
    d.append(tokens[200:])
    assert d.propose(6) == _brute_propose(tokens, 6)


def test_policy_prefers_plain_steps_when_windows_are_expensive():
    p = DraftPolicy(max_t=8)
    for t in range(1, 9):
        for _ in range(4):
            p.observe_cost(t, 20.0 + 40.0 * (t - 1))
    p.sync()
    assert p.choose(7, match=30) == 0


def test_policy_takes_a_window_that_pays():
    p = DraftPolicy(max_t=8)
    for t in range(1, 9):
        for _ in range(4):
            p.observe_cost(t, 20.0 + 4.0 * (t - 1))
    p.sync()
    assert p.choose(7, match=30) == 7
    # learned: drafts after short matches always fail
    for _ in range(50):
        p.observe_round(drafts=7, accepted=0, match=3)
    assert p.choose(7, match=3) == 0
    assert p.choose(7, match=30) == 7


def test_policy_probes_an_unmeasured_size_a_few_times():
    p = DraftPolicy(max_t=8)
    for _ in range(4):
        p.observe_cost(1, 10.0)
        p.observe_cost(2, 100.0)
    p.sync()
    # scaled from the measured sizes a 5-wide window does not pay, but a confident proposal probes it
    taken = [p.choose(4, match=30) for _ in range(5)]
    assert taken[: DraftPolicy.PROBES] == [4] * DraftPolicy.PROBES
    assert taken[DraftPolicy.PROBES :] == [0, 0]
    assert p.choose(4, match=30, probe=False) == 0


def test_policy_ignores_the_first_round_of_a_width():
    p = DraftPolicy(max_t=4)
    p.observe_cost(3, 5000.0)  # kernel compilation
    assert p.meas_n[3] == 0
    p.observe_cost(3, 30.0)
    assert p.meas[3] == 30.0


def test_policy_prices_only_adopted_costs():
    """TP: two ranks measure different clocks but choose identically once both adopt rank 0's
    snapshot; a rank's own later measurements change nothing until the next adopt."""
    rank0, rank1 = DraftPolicy(max_t=8), DraftPolicy(max_t=8)
    for t in range(1, 9):
        for _ in range(4):
            rank0.observe_cost(t, 20.0 + 4.0 * (t - 1))  # windows pay
            rank1.observe_cost(t, 20.0 + 40.0 * (t - 1))  # windows do not
    grid = [(k, m, probe) for k in range(1, 8) for m in (3, 5, 8, 13, 30) for probe in (False, True)]
    rank0.sync()
    rank1.sync()
    assert [rank0.choose(*g) for g in grid] != [rank1.choose(*g) for g in grid]
    rank0, rank1 = DraftPolicy(max_t=8), DraftPolicy(max_t=8)
    for t in range(1, 9):
        for _ in range(4):
            rank0.observe_cost(t, 20.0 + 4.0 * (t - 1))
            rank1.observe_cost(t, 20.0 + 40.0 * (t - 1))
    snap = rank0.snapshot()
    rank0.adopt(snap)
    rank1.adopt(snap)
    rank1.observe_cost(2, 500.0)  # measured after the sync point: not priced yet
    assert [rank0.choose(*g) for g in grid] == [rank1.choose(*g) for g in grid]
    assert rank0.choose(7, match=30) == 7

"""Self-speculative n-gram drafts: no draft model, the request's own history is the drafter.

* ``ngram_draft`` -- port of llama.cpp's ``common_ngram_simple_draft``; the MTP combiner
  (``FREETOKEN_MTP_NGRAM``) overrides an MTP draft chain with it.
* ``SuffixDrafter`` -- prompt lookup (port of Strata's ``spec/suffix_drafter``): every trigram of
  prompt + output indexes its end positions; a proposal extends the most recent candidates
  backwards and continues the longest match. Drives ``--ngram-spec`` (scheduler/ngram_spec.py).
* ``DraftPolicy`` -- whether a proposal is worth a verify window and how long, from the measured
  round cost per window size and the acceptance seen per match length (port of Strata's
  ``spec/draft_policy``, with a plain decode step as the alternative instead of an MTP window).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def ngram_draft(tokens: Sequence[int], ngram_size: int, draft_len: int) -> List[int]:
    """Draft tokens following the most recent earlier occurrence of the last ``ngram_size``
    tokens of ``tokens``. ``tokens`` includes the pending token as its last element.
    Returns up to ``draft_len`` tokens, or [] when there is no match."""
    cur = len(tokens)
    if ngram_size <= 0 or draft_len <= 0 or cur <= ngram_size + draft_len:
        return []
    pattern = list(tokens[cur - ngram_size:])
    # Search backwards, skipping the current match at the end.
    for start in range(cur - ngram_size - 1, 0, -1):
        if list(tokens[start:start + ngram_size]) == pattern:
            copy_max = min(draft_len, cur - (start + ngram_size))
            if copy_max < 1:
                continue
            return [int(t) for t in tokens[start + ngram_size:start + ngram_size + copy_max]]
    return []


# token ids < 2**20 (Qwen vocab ~248k): a trigram packs into one int64 key
_ID_BITS = 20


class SuffixDrafter:
    """Prompt lookup over one request's token history.

    ``head[key]`` is the last end position of a trigram, ``prev[end]`` the previous end of the
    same trigram, so a lookup walks the ``ways`` most recent occurrences. A proposal extends
    each candidate backwards (up to ``max_match`` tokens) and continues the longest match, the
    most recent one on ties; matches shorter than ``min_match`` propose nothing. The
    continuation may run into the current suffix (periodic text)."""

    WAYS = 4

    def __init__(self, min_match: int = 3, max_match: int = 64) -> None:
        self.min_match = max(3, int(min_match))
        self.max_match = max(self.min_match, int(max_match))
        self.hist: List[int] = []
        self.prev: List[int] = []
        self.head: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self.hist)

    @staticmethod
    def _key(a: int, b: int, c: int) -> int:
        return (((a << _ID_BITS) | b) << _ID_BITS) | c

    def append(self, tokens: Sequence[int]) -> None:
        tokens = [int(t) for t in tokens]
        if len(tokens) > 64:
            self._append_bulk(tokens)
            return
        hist, prev, head = self.hist, self.prev, self.head
        for t in tokens:
            hist.append(t)
            end = len(hist) - 1
            if end < 2:
                prev.append(-1)
                continue
            key = self._key(hist[end - 2], hist[end - 1], t)
            prev.append(head.get(key, -1))
            head[key] = end

    def _append_bulk(self, tokens: List[int]) -> None:
        """Index a long chunk (a prompt) without a per-token Python loop."""
        n0 = len(self.hist)
        self.hist.extend(tokens)
        h = np.asarray(self.hist[max(0, n0 - 2):], dtype=np.int64)
        base = max(0, n0 - 2)
        ends = np.arange(base + 2, len(self.hist), dtype=np.int64)
        ends = ends[ends >= n0]
        prev_new = np.full(len(tokens), -1, dtype=np.int64)
        if ends.size:
            o = ends - base
            keys = (((h[o - 2] << _ID_BITS) | h[o - 1]) << _ID_BITS) | h[o]
            order = np.argsort(keys, kind="stable")
            sk = keys[order]
            same = sk[1:] == sk[:-1]
            # within the chunk: the previous end of the same key in position order
            prev_new[ends[order[1:][same]] - n0] = ends[order[:-1][same]]
            # first occurrence in the chunk links to the history before it
            first = np.ones(len(sk), dtype=bool)
            first[1:] = ~same
            head = self.head
            for key, end in zip(sk[first].tolist(), ends[order[first]].tolist()):
                p = head.get(key)
                if p is not None:
                    prev_new[end - n0] = p
            # later ends overwrite earlier ones: the head is the last occurrence
            head.update(zip(keys.tolist(), ends.tolist()))
        self.prev.extend(prev_new.tolist())

    def propose(self, max_k: int) -> Tuple[List[int], int]:
        """Up to ``max_k`` tokens continuing the history and the match length behind them;
        ``([], 0)`` when nothing matches."""
        hist = self.hist
        n = len(hist)
        if n < 4 or max_k <= 0:
            return [], 0
        cur = n - 1
        p = self.head.get(self._key(hist[cur - 2], hist[cur - 1], hist[cur]), -1)
        best_end, best_len, seen = -1, 0, 0
        prev, max_match = self.prev, self.max_match
        while p >= 0 and seen < self.WAYS:
            if p < cur:
                seen += 1
                length = 3
                while length < max_match and length <= p and hist[p - length] == hist[cur - length]:
                    length += 1
                if length > best_len:
                    best_len, best_end = length, p
            p = prev[p]
        if best_len < self.min_match:
            return [], 0
        return hist[best_end + 1 : min(cur + 1, best_end + 1 + max_k)], best_len


class DraftPolicy:
    """Take a lookup window only where it pays: compare expected committed tokens per ms.

    ``plain``: one token per decode step at the measured step cost ``cost(1)``.
    ``lookup``: a window of ``k`` drafts commits ``E(k) = 1 + q + ... + q^k`` tokens (the
    pending token's bonus plus each draft surviving with probability ``q``, learned per
    match-length bucket from decayed counts) at the measured round cost ``cost(k + 1)``.
    The best ``k`` is taken when ``E(k) / cost(k + 1)`` beats ``1 / cost(1)`` by ``margin``.
    Sizes not measured yet are scaled from the measured ones by a prior shape, and a confident
    proposal tries a size a few times before a guessed cost can veto it. Only chooses which
    drafts to verify: the output does not depend on it.

    Costs are measured on this rank's clock (``meas``) but priced from a copy (``cost``) taken at
    sync points (``adopt``): under TP every rank must choose the same windows (a round is a
    collective), so all ranks adopt rank 0's measurements at the same scheduler iteration."""

    BUCKETS = (6, 12, 24)  # match length < 6, < 12, < 24, longer
    PRIOR_Q = (0.75, 0.88, 0.93, 0.96)
    PRIOR_N = 4.0
    # round cost by window size relative to a plain step, before anything is measured
    SHAPE = (0.0, 1.0, 1.6, 1.9, 2.2, 2.5, 2.8, 3.1, 3.4, 3.7, 4.0, 4.3, 4.6, 4.9, 5.2, 5.5, 5.8)
    COST_ALPHA = 0.1
    DECAY = 0.97
    PROBES = 3
    PROBE_Q = 0.85

    def __init__(self, max_t: int, margin: float = 0.03) -> None:
        self.max_t = max(1, min(int(max_t), len(self.SHAPE) - 1))
        self.margin = margin
        n = self.max_t + 1
        self.meas = [0.0] * n
        self.meas_n = [0.0] * n
        self.cost = [0.0] * n
        self.cost_n = [0.0] * n
        self.tried = [0] * n
        self.warm = [False] * n
        self.ok = [0.0] * (len(self.BUCKETS) + 1)
        self.bad = [0.0] * (len(self.BUCKETS) + 1)

    def bucket(self, match: int) -> int:
        for i, b in enumerate(self.BUCKETS):
            if match < b:
                return i
        return len(self.BUCKETS)

    def rate(self, match: int) -> float:
        b = self.bucket(match)
        return (self.ok[b] + self.PRIOR_N * self.PRIOR_Q[b]) / (self.ok[b] + self.bad[b] + self.PRIOR_N)

    def cost_ms(self, t: int) -> float:
        t = max(1, min(t, self.max_t))
        if self.cost_n[t] > 0:
            return self.cost[t]
        num = den = 0.0
        for u in range(1, self.max_t + 1):
            if self.cost_n[u] > 0:
                w = min(self.cost_n[u], 20.0)
                num += w * self.cost[u] * self.SHAPE[t] / self.SHAPE[u]
                den += w
        return num / den if den > 0 else self.SHAPE[t]

    def choose(self, k_avail: int, match: int, probe: bool = True) -> int:
        """Drafts to verify (0 = a plain decode step) for a proposal of ``k_avail`` tokens.
        ``probe=False`` asks without spending one of a size's probes."""
        k_avail = min(k_avail, self.max_t - 1)
        if k_avail <= 0:
            return 0
        q = self.rate(match)
        base = 1.0 / self.cost_ms(1)
        e = qi = 1.0
        best, best_k = 0.0, 0
        for k in range(1, k_avail + 1):
            qi *= q
            e += qi
            r = e / self.cost_ms(k + 1)
            if r > best:
                best, best_k = r, k
        if best_k and best > base * (1.0 + self.margin):
            return best_k
        t_full = k_avail + 1
        if q >= self.PROBE_Q and self.cost_n[t_full] + self.tried[t_full] < self.PROBES:
            if probe:
                self.tried[t_full] += 1
            return k_avail
        return 0

    def observe_cost(self, t: int, ms: float) -> None:
        """Record a measured round (``t`` > 1) or plain step (``t`` == 1); priced after ``adopt``."""
        if not 1 <= t <= self.max_t or ms <= 0:
            return
        if not self.warm[t]:
            # the first round of a width compiles its kernels: it says nothing about the cost
            self.warm[t] = True
            return
        if self.meas_n[t] > 0:
            ms = min(ms, 2.0 * self.meas[t])  # one stalled round cannot price a width out
        # a running mean for the first samples, then an EMA
        alpha = max(self.COST_ALPHA, 1.0 / (self.meas_n[t] + 1.0))
        self.meas[t] = ms if self.meas_n[t] == 0 else (1.0 - alpha) * self.meas[t] + alpha * ms
        self.meas_n[t] += 1.0

    def snapshot(self) -> List[float]:
        """The measured costs, ``2 * max_t`` values for ``adopt`` (on every rank)."""
        return self.meas[1:] + self.meas_n[1:]

    def adopt(self, values: Sequence[float]) -> None:
        m = self.max_t
        self.cost[1:] = [float(v) for v in values[:m]]
        self.cost_n[1:] = [float(v) for v in values[m : 2 * m]]

    def sync(self) -> None:
        """Single rank: price with this rank's own measurements."""
        self.adopt(self.snapshot())

    def observe_round(self, drafts: int, accepted: int, match: int) -> None:
        b = self.bucket(match)
        self.ok[b] = self.DECAY * self.ok[b] + accepted
        self.bad[b] = self.DECAY * self.bad[b] + (1.0 if accepted < drafts else 0.0)

    def cost_table(self) -> str:
        return " ".join(
            f"T{t}:{self.meas[t]:.1f}(n={self.meas_n[t]:.0f})"
            for t in range(1, self.max_t + 1) if self.meas_n[t] > 0
        )


__all__ = ["DraftPolicy", "SuffixDrafter", "ngram_draft"]

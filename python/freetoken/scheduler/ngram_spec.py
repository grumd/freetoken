"""Prompt-lookup speculative decoding (``--ngram-spec``): drafts without a draft head.

When the text being written repeats an earlier stretch of the request's own context (code
edits, quoted input, tool-call JSON), the tokens that followed that stretch are a free draft.
``SuffixDrafter`` proposes them and ``DraftPolicy`` decides whether a verify window pays
against a plain decode step (measured round costs, acceptance per match length).

A round is synchronous and reuses the MTP verify machinery:

1. entry: ``cached_len == device_len - 1``, the pending token on the host and in ``token_pool``;
2. one prefill-phase extend over ``[C, C+k+1)`` (pending token + k drafts) with
   ``mtp_verify`` set: the GDN layers run the decode-exact verify and leave their live state
   alone, the PLE layer keeps its conv history / n-gram context aside, the MoE layers take the
   CPU-assisted decode split;
3. acceptance: greedy -- the target argmax at ``C+j`` must equal draft ``j``; sampled -- the
   draft is a point-mass proposal, accepted with probability ``p_target(d)``, the bonus drawn from
   the residual on rejection (the output distribution is the target's);
4. publish the accepted drafts and the bonus, roll back the rejected pages and commit the
   accepted prefix into the GDN / PLE state (``commit_mtp_verify``).

Plain decode steps keep the overlap loop and the CUDA graph. A request whose step is still in
flight is a *candidate* when the history up to its last drained token continues a match (the
in-flight token is guessed); only then is the step drained early and a real proposal made, so
ordinary text pays one dictionary lookup per step. A window never commits past the next
decode snapshot boundary, so the hybrid prefix cache keeps its reuse points.

Under TP every rank runs this manager and the rounds are collectives, so every rank must plan
the same windows. The plan depends only on host state all ranks share (ids, lengths, accepted
counts) plus the policy's priced costs and the diagnostics files, which rank 0 sends along with
its per-iteration message-count broadcast (``sync_payload`` -> ``sync``).
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Dict, Iterable, List, Sequence, Set, Tuple

import torch
from freetoken.core import Batch, Req
from freetoken.engine.sample import probs_from_logits
from freetoken.env import ENV
from freetoken.message import DetokenizeMsg
from freetoken.utils import init_logger

from .ngram_draft import DraftPolicy, SuffixDrafter

if TYPE_CHECKING:
    from .scheduler import Scheduler

logger = init_logger(__name__)

Plan = Tuple[Req, List[int], int]  # (request, drafts, match length)
MODES = ("adaptive", "always", "fixed", "fixed-null")


class NgramSpecManager:
    is_ngram = True
    uses_streams = False
    overlaps = True

    def __init__(self, sched: "Scheduler") -> None:
        from freetoken.kernel.fla.chunk import CHUNK_SIZE

        self.sched = sched
        self.engine = sched.engine
        self.enabled = True
        self.k = max(1, int(ENV.NGRAM_SPEC_MAX_DRAFT.value))
        # a longer window would get hybrid-radix track metadata the GDN verify never writes
        assert self.k + 1 <= CHUNK_SIZE, f"--ngram-spec-max-draft + 1 must fit one GDN chunk ({CHUNK_SIZE})"
        self.min_match = max(3, int(ENV.NGRAM_SPEC_MIN_MATCH.value))
        self._set_mode(str(ENV.NGRAM_SPEC_POLICY.value))
        # diagnostics: a file whose content replaces the policy for the requests that start after
        self._mode_file = os.environ.get("FREETOKEN_NGRAM_SPEC_MODE_FILE") or None
        self._mode_next: str | None = None
        self._paused_flag = False
        self._tp = sched.config.tp_info.size > 1
        self.policy = DraftPolicy(self.k + 1)
        self._drafters: Dict[int, SuffixDrafter] = {}
        self._req_stats: Dict[int, List[int]] = {}  # uid -> [rounds, drafted, accepted, round tokens]
        self._deferred: Set[Req] = set()
        # diagnostics A/B inside one server: no drafting while this file exists
        self._off_file = os.environ.get("FREETOKEN_NGRAM_SPEC_OFF_FILE") or None
        self._image_token_id = getattr(sched.config.model_config, "image_token_id", None)
        self.stats_drafted = 0
        self.stats_accepted = 0
        self.stats_rounds = 0
        self._rounds_logged = 0
        self._a_hist = [0] * (self.k + 1)
        self._last_plain_t: float | None = None
        self._debug = os.environ.get("FREETOKEN_NGRAM_SPEC_DEBUG") == "1"
        logger.info_rank0(
            f"n-gram speculative decoding: drafts up to {self.k} tokens from context repeats of "
            f">= {self.min_match} tokens, policy {self.mode}"
        )

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self.always = mode == "always"
        # diagnostics: every decode step is a full-width round, drafts padded with token 0
        # ("fixed": lookup drafts first, "fixed-null": padding only) -- two runs whose rounds all
        # have the same shape, so their outputs must match bit for bit if the rollback is exact
        self.fixed = mode in ("fixed", "fixed-null")

    # ------------------------------------------------------------------ drafting

    def eligible(self, r: Req) -> bool:
        return r.table_idx != -1 and not r.aborted and r.remain_len >= 2

    def _drafter(self, r: Req) -> SuffixDrafter:
        d = self._drafters.get(r.uid)
        if d is None:
            d = self._drafters[r.uid] = SuffixDrafter(self.min_match)
            self._req_stats[r.uid] = [0, 0, 0, 0]
            if self._mode_next is not None and self._mode_next != self.mode:
                logger.info_rank0(f"ngram-spec policy -> {self._mode_next}")
                self._set_mode(self._mode_next)
        n = r.input_ids.numel()
        if len(d) < n:
            d.append(r.input_ids[len(d) : n].tolist())
        return d

    def _paused(self) -> bool:
        return self._paused_flag

    # ------------------------------------------------------------------ TP sync

    @property
    def sync_len(self) -> int:
        return 2 + 2 * self.policy.max_t

    def sync_payload(self) -> List[float]:
        """Rank 0 (or the only rank): the values every rank plans with this iteration."""
        paused = self._off_file is not None and os.path.exists(self._off_file)
        mode = -1
        if self._mode_file is not None and os.path.exists(self._mode_file):
            text = open(self._mode_file).read().strip()
            mode = MODES.index(text) if text in MODES else -1
        return [float(paused), float(mode)] + self.policy.snapshot()

    def begin_iteration(self) -> None:
        """Before any planning in a scheduler iteration (after its message poll)."""
        if not self._tp:
            self.sync(None)
        elif self.sched.tp_sync_values is not None:
            self.sync(self.sched.tp_sync_values)

    def sync(self, values: Sequence[float] | None) -> None:
        """Start of an iteration: adopt rank 0's values (``None``: single rank, take our own)."""
        if values is None:
            values = self.sync_payload()
        self._paused_flag = bool(values[0])
        self._mode_next = MODES[int(values[1])] if values[1] >= 0 else None
        self.policy.adopt(values[2:])

    def candidates(self, reqs: Iterable[Req], in_flight: Iterable[Req]) -> List[Req]:
        """Requests worth draining early: a request whose decode step is in flight continues a
        match as of its last drained token (the in-flight token is guessed to follow it); a
        drained request has a proposal right now."""
        self._gc(reqs)
        if self._paused():
            return []
        if self.fixed:
            return [r for r in reqs if self.eligible(r)]
        flying = set(in_flight)
        out = []
        for r in reqs:
            if not self.eligible(r):
                continue
            if r in flying:
                drafts, match = self._drafter(r).propose(self.k + 1)
                if len(drafts) >= 2 and (self.always or self.policy.choose(len(drafts) - 1, match, probe=False)):
                    out.append(r)
            elif self.plan_one(r, probe=False) is not None:
                out.append(r)
        return out

    def plan_one(self, r: Req, probe: bool = True) -> Plan | None:
        """The drafts to verify for a drained request, or None for a plain decode step
        (``probe=False``: a look-ahead that spends none of the policy's probes)."""
        if not self.eligible(r) or r.input_ids.numel() != r.device_len:
            return None
        sched = self.sched
        anchor = sched.toolcall_anchor_id
        if anchor is not None and int(r.input_ids[-1]) == anchor:
            # a pending tool-call opener: a plain step lands the live state on its anchor snapshot
            return None
        k = min(self.k, r.remain_len - 1)
        if self.fixed and k > 0:
            drafts = self._drafter(r).propose(k)[0] if self.mode == "fixed" else []
            return (r, (self._truncate(drafts) + [0] * k)[:k], 0)
        # never commit past the next decode snapshot boundary (its GDN snapshot is taken on entry)
        interval = sched.cache_manager.decode_snapshot_interval
        if interval:
            k = min(k, (r.cached_len // interval + 1) * interval - r.cached_len - 1)
        if k <= 0:
            return None
        drafts, match = self._drafter(r).propose(k)
        drafts = self._truncate(drafts)
        if not drafts:
            return None
        n = len(drafts) if self.always else self.policy.choose(len(drafts), match, probe)
        if n <= 0:
            return None
        return (r, drafts[:n], match)

    def _truncate(self, drafts: List[int]) -> List[int]:
        """Image placeholders never come from the model; a tool-call opener ends the window so a
        round lands on its anchor snapshot."""
        if self._image_token_id is not None and self._image_token_id in drafts:
            drafts = drafts[: drafts.index(self._image_token_id)]
        anchor = self.sched.toolcall_anchor_id
        if anchor is not None and anchor in drafts:
            drafts = drafts[: drafts.index(anchor) + 1]
        return drafts

    def plan(self, reqs: Sequence[Req]) -> List[Plan]:
        return [p for p in (self.plan_one(r) for r in reqs) if p is not None]

    def _gc(self, reqs: Iterable[Req]) -> None:
        """Drop the drafters of finished requests (logging their counts) and start new ones."""
        reqs = list(reqs)
        live = {r.uid for r in reqs}
        for r in reqs:
            if r.uid not in self._drafters and self.eligible(r):
                self._drafter(r)
        for uid in [u for u in self._drafters if u not in live]:
            del self._drafters[uid]
            rounds, drafted, accepted, toks = self._req_stats.pop(uid, [0, 0, 0, 0])
            if rounds:
                logger.info_rank0(
                    f"ngram-spec request {uid}: {rounds} rounds, {drafted} drafted, {accepted} accepted "
                    f"(rate {accepted / max(1, drafted):.2f}, {accepted / rounds:.2f} per round), "
                    f"{toks} tokens from rounds"
                )

    # ------------------------------------------------------------- scheduler hooks

    def draft_prefill(self, batch, pre_lens) -> None:
        """No draft model to feed: the lookup index reads the host ids when a request decodes."""

    def run_round(self) -> None:
        """normal_loop: every decode request is drained here, so plan directly."""
        self._deferred = set()
        sched = self.sched
        self.begin_iteration()
        if sched.prefill_manager.runnable:
            return
        running = list(sched.decode_manager.running_reqs)
        self._gc(running)
        if self._paused():
            return
        plans = self.plan(running)
        if plans:
            self.run_plans(plans)

    def run_plans(self, plans: List[Plan]) -> None:
        """One synchronous round over ``plans``; requests with another proposal right after it
        skip this iteration's plain decode step (the next iteration's round takes them)."""
        t0 = time.perf_counter()
        width = 1 + max(len(d) for _, d, _ in plans)
        a_list = self._round(plans)
        ms = (time.perf_counter() - t0) * 1e3
        self.policy.observe_cost(width, ms)
        if self._debug:
            for (r, d, match), a in zip(plans, a_list):
                logger.info_rank0(
                    f"ngram-spec round: uid={r.uid} pos={r.cached_len} drafts={len(d)} match={match} "
                    f"accepted={a} ms={ms:.1f}"
                )
        self._last_plain_t = None
        reqs = [r for r, _, _ in plans if r in self.sched.decode_manager.running_reqs]
        self._deferred = {r for r in reqs if self.plan_one(r, probe=False) is not None}

    def note_plain_step(self, single: bool) -> None:
        """Overlap loop, after a plain decode step launched: the time between two consecutive
        plain launches of a lone request is its decode step cost."""
        now = time.perf_counter()
        if single and self._last_plain_t is not None:
            self.policy.observe_cost(1, (now - self._last_plain_t) * 1e3)
        self._last_plain_t = now if single else None

    # ------------------------------------------------------------------ the round

    def _round(self, plans: List[Plan]) -> None:
        sched, eng = self.sched, self.engine
        dev = eng.device
        reqs = [r for r, _, _ in plans]
        n = len(reqs)
        k = max(len(d) for _, d, _ in plans)
        # shorter windows are padded to the batch's width; padding is never accepted (a request
        # cut at its tool-call opener must not commit past it)
        real = [len(d) for _, d, _ in plans]
        drafts = [d + [d[-1]] * (k - len(d)) for _, d, _ in plans]
        C = [r.cached_len for r in reqs]
        for r, c, d in zip(reqs, C, drafts):
            assert r.device_len == c + 1 and r.input_ids.numel() == c + 1, (
                "an n-gram round expects a drained request with its pending token at cached_len"
            )
            r._ids_buf[c + 1 : c + 1 + k] = torch.tensor(d, dtype=r._ids_buf.dtype)
            r.input_ids = r._ids_buf[: c + 1 + k]
            r.device_len = c + 1 + k
        # the live GDN slot holds exactly C tokens: a decode snapshot boundary is taken here
        sched.cache_manager.snapshot_decode_states(
            reqs, anchors=sched.toolcall_anchor_id is not None, boundaries=True
        )
        verify_batch = Batch(reqs=reqs, phase="prefill")
        sched._prepare_batch(verify_batch)
        window = torch.tensor(
            [r._ids_buf[c : c + 1 + k].tolist() for r, c in zip(reqs, C)], dtype=torch.int32,
            pin_memory=True,
        ).to(dev, non_blocking=True)
        for i, (r, c) in enumerate(zip(reqs, C)):
            sched.token_pool[r.table_idx, c : c + 1 + k] = window[i].to(sched.token_pool.dtype)
        verify_batch.input_ids = window.view(-1)
        verify_batch.mtp_verify = True
        verify_batch.mtp_verify_n = n
        verify_batch.mtp_verify_t = k + 1
        graphs = getattr(eng, "ngram_verify_graphs", None)
        if graphs is not None and graphs.can_use(n, k + 1):
            logits, _ = graphs.replay(verify_batch, k + 1)
        else:
            logits, _ = eng.extend_forward(verify_batch)

        a_list, bonus_list = self._accept(reqs, logits, drafts, real, k, dev)
        for (_, d, match), a in zip(plans, a_list):
            self.policy.observe_round(len(d), a, match)
        self._publish(reqs, C, k, drafts, real, a_list, bonus_list)
        return a_list

    def _accept(self, reqs, logits, drafts, real, k, dev):
        """Per request (accepted drafts a <= its real draft count, bonus token).
        Row ``i*(k+1)+j`` predicts ``C+j+1``."""
        n = len(reqs)
        draft_t = torch.tensor(drafts, dtype=torch.int64).to(dev, non_blocking=True)  # [n, k]
        greedy = [r.sampling_params.is_greedy for r in reqs]
        am = torch.argmax(logits, dim=-1).view(n, k + 1) if any(greedy) else None
        am_host = am.tolist() if am is not None else None
        a_list, bonus_list = [], []
        for i, r in enumerate(reqs):
            if greedy[i]:
                row = am_host[i]
                a = next((j for j in range(real[i]) if row[j] != drafts[i][j]), real[i])
                a_list.append(a)
                bonus_list.append(int(row[a]))
                continue
            probs = probs_from_logits(logits[i * (k + 1) : (i + 1) * (k + 1)], r.sampling_params)
            p_draft = probs[torch.arange(k, device=dev), draft_t[i]]
            ok = (torch.rand(k, device=dev) < p_draft).tolist()
            a = next((j for j in range(real[i]) if not ok[j]), real[i])
            if a < real[i]:
                resid = probs[a].clone()
                resid[drafts[i][a]] = 0.0
                total = float(resid.sum())
                # the rejected draft held all the mass only if p(d) == 1, which accepts it
                bonus = torch.multinomial(resid / total, 1).view(()) if total > 0 else probs[a].argmax()
            else:
                bonus = torch.multinomial(probs[a], 1).view(())
            a_list.append(a)
            bonus_list.append(int(bonus))
        return a_list, bonus_list

    def _publish(self, reqs, C, k, drafts, real, a_list, bonus_list) -> None:
        """Publish accepted drafts + bonus, roll back the rejected pages, commit the accepted
        prefix into the GDN / PLE slots (same bookkeeping as ``MTPManager.finish_round``)."""
        sched, eng = self.sched, self.engine
        cm = sched.cache_manager
        dev = eng.device
        reply: List[DetokenizeMsg] = []
        plans: List[Tuple[Req, int, int, bool, bool]] = []
        for i, r in enumerate(reqs):
            a, bonus = a_list[i], bonus_list[i]
            m = 0
            finished, finish_reason, matched_stop = False, None, None
            r.input_ids = r._ids_buf[: C[i] + 1]

            def _check(tok: int):
                hit_length = r.output_budget_exhausted
                hit_eos = not r.sampling_params.ignore_eos and tok in sched.eos_token_ids
                ms = sched._match_stop_str(r) if (not hit_eos and r.sampling_params.stop_strs) else None
                fin = hit_length or hit_eos or ms is not None
                return fin, ("stop" if (hit_eos or ms is not None) else "length") if fin else None, ms

            def _emit(tok: int) -> None:
                nonlocal finished, finish_reason, matched_stop
                r.append_host(torch.tensor([tok]))
                finished, finish_reason, matched_stop = _check(tok)
                if tok == sched.toolcall_anchor_id and r.toolcall_anchor_len is None and not finished:
                    r.toolcall_anchor_len = r.input_ids.numel()
                reply.append(DetokenizeMsg(
                    uid=r.uid, next_token=tok, finished=finished, finish_reason=finish_reason,
                    matched_stop=matched_stop, stop_strs=r.sampling_params.stop_strs or None,
                ))

            for j in range(a):
                if r.output_budget_exhausted:
                    finished, finish_reason = True, "length"
                    break
                _emit(drafts[i][j])
                m += 1
                if finished:
                    break
            bonus_published = not finished
            if bonus_published and not r.output_budget_exhausted:
                _emit(bonus)
            finished_at_draft = finished and not bonus_published
            # committed: the pending token + the published drafts (not the bonus)
            rext_end = C[i] + 1 + (m if finished_at_draft else a)
            keep_to = rext_end + (0 if finished_at_draft else 1)
            r.device_len = C[i] + k + 1
            cm.rollback_last(r, rext_end)
            r.device_len = keep_to
            r.input_ids = r._ids_buf[:keep_to]
            plans.append((r, rext_end, keep_to, finished, bonus_published))
            st = self._req_stats.get(r.uid)
            if st is not None:
                st[0] += 1
                st[1] += real[i]
                st[2] += a
                st[3] += rext_end - C[i] + (1 if bonus_published else 0) - 1
            self.stats_rounds += 1
            self.stats_drafted += real[i]
            self.stats_accepted += a
            self._a_hist[a] += 1

        bonus_gpu = torch.tensor(bonus_list, dtype=torch.int32).to(dev, non_blocking=True)
        for i, (r, rext_end, _keep, _fin, bonus_published) in enumerate(plans):
            if bonus_published:
                sched.token_pool[r.table_idx, rext_end] = bonus_gpu[i].to(sched.token_pool.dtype)
        pool = eng.linear_state_pool
        graphs = getattr(eng, "ngram_verify_graphs", None)
        if pool is not None:
            slots = [r.linear_slot_idx if r.linear_slot_idx is not None else r.table_idx for r, *_ in plans]
            lens = [p[1] - C[i] for i, p in enumerate(plans)]
            if graphs is None or not graphs.commit(slots, lens, k + 1):
                eng.model.commit_mtp_verify(
                    pool,
                    torch.tensor(slots, dtype=torch.int32).to(dev, non_blocking=True),
                    torch.tensor(lens, dtype=torch.int32).to(dev, non_blocking=True),
                )
        for r, rext_end, keep_to, finished, _bp in plans:
            r.cached_len = rext_end
            if finished:
                sched.decode_manager.remove_req(r)
                sched._free_req_resources(r)
                sched.finished_reqs.add(r)

        if reply:
            used, total = sched._kv_usage_pages()
            mamba_used, mamba_total = sched._mamba_slot_usage() or (0, 0)
            swa_used, swa_total = sched._swa_token_usage() or (0, 0)
            mem = sched._gpu_mem_bytes()
            for msg in reply:
                msg.kv_used_pages, msg.kv_total_pages = used, total
                msg.mamba_used_slots, msg.mamba_total_slots = mamba_used, mamba_total
                msg.swa_used_tokens, msg.swa_total_tokens = swa_used, swa_total
                msg.gpu_mem_bytes = mem
                msg.mtp_enabled = True
                msg.mtp_drafted = self.stats_drafted
                msg.mtp_accepted = self.stats_accepted
            sched.send_result(reply)
        self._rounds_logged += len(reqs)
        if self._rounds_logged >= 100:
            self._rounds_logged = 0
            ran = max(1, sum(self._a_hist))
            cond = " ".join(f"a>={j}:{sum(self._a_hist[j:]) / ran:.2f}" for j in range(1, self.k + 1))
            logger.info_rank0(
                f"ngram-spec: {self.stats_rounds} rounds, drafted {self.stats_drafted}, accepted "
                f"{self.stats_accepted} (rate {self.stats_accepted / max(1, self.stats_drafted):.2f}) | "
                f"{cond} | cost ms {self.policy.cost_table()}"
            )


__all__ = ["NgramSpecManager"]

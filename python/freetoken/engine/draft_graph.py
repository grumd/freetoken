"""CUDA-graph capture for the MTP draft chain (Phase 3b).

The draft chain runs ``k`` sequential single-token forwards of the MTP block (a plain
bf16 resident MoE + QSA attention), each an extend of ``t=1`` token per request. The
attention backend only captures decode batches and the ``k+1`` verify extend, so this
runner captures the *single draft step* with its own static geometry -- one graph per
padded batch size (mirroring ``GraphRunner``/``VerifyGraphRunner``), replayed ``k``
times per round with the per-step positions / out_loc / MTP-KV length staged in
between. The ``R_next``/token hand-off between steps stays outside the graph (a device
copy), so nothing here needs to know ``k``. Each step has a greedy variant and a sampled
one (the draft drawn from the head's truncated distribution, engine/draft_sample.py).

Everything inside the captured region is a device kernel; the eager Python (metadata
build, page allocation, per-step staging) stays outside. Captured whenever the model has an
MTP head.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

import torch
from freetoken.core import Batch, get_global_ctx
from freetoken.utils import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.engine.graph import GraphRunner
    from freetoken.models import BaseLLMModel


class _DraftReq:
    """Minimal request view for the synthetic draft batch: the QSA metadata reads only
    ``table_idx`` (``ring_slots``) off the reqs; positions/out_loc ride on the batch."""

    __slots__ = ("table_idx", "cached_len", "device_len")

    def __init__(self, table_idx: int, cached_len: int, device_len: int) -> None:
        self.table_idx = table_idx
        self.cached_len = cached_len
        self.device_len = device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len


class _DraftBatch:
    """Synthetic prefill-phase batch for one captured draft step (t=1 token/req)."""

    def __init__(
        self, reqs: Sequence[_DraftReq], positions: torch.Tensor, out_loc: torch.Tensor
    ) -> None:
        self.reqs: List[_DraftReq] = list(reqs)
        self.positions = positions
        self.out_loc = out_loc
        self.phase = "prefill"
        self.attn_metadata = None

    @property
    def padded_reqs(self) -> List[_DraftReq]:
        return self.reqs

    @property
    def is_prefill(self) -> bool:
        return True

    @property
    def is_decode(self) -> bool:
        return False

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.reqs)


class DraftGraphRunner:
    """Captures (once) and replays one MTP draft step for each padded batch size."""

    def __init__(
        self,
        graph_runner: "GraphRunner",
        model: "BaseLLMModel",
        attn_backend: "BaseAttnBackend",
        bs_list: List[int],
        width: int,
        device: torch.device,
        draft_vocab=None,
    ) -> None:
        self.gr = graph_runner
        self.model = model
        self.attn = attn_backend
        self.mtp = model.mtp
        self.embed = model.model.embed_tokens.forward
        self.lm_head = model.lm_head
        self.draft_vocab = draft_vocab
        self.bs_list = sorted(bs_list)
        self.n_max = max(self.bs_list)
        self.width = width
        self.device = device
        self.dummy_table_idx = graph_runner.dummy_req.table_idx
        # (padded bs, draft-vocab subset head?, sampled drafts?) -> graph
        self.graph_map: Dict[Tuple[int, bool, bool], torch.cuda.CUDAGraph] = {}
        # Private memory pool: the draft graphs must NOT share the decode/verify pool, or a
        # replay of one overwrites the other's baked transients.
        self.pool = None

        self.R_in = torch.zeros(self.n_max, width, dtype=torch.bfloat16, device=device)
        self.token_in = torch.zeros(self.n_max, dtype=torch.int32, device=device)
        self.positions = torch.zeros(self.n_max, dtype=torch.int32, device=device)
        self.out_loc = torch.zeros(self.n_max, dtype=torch.int32, device=device)
        # The MTP block's output stream is the resident bf16 draft dtype.
        self.R_out = torch.empty(self.n_max, width, dtype=torch.bfloat16, device=device)
        self.draft_ids = torch.zeros(self.n_max, dtype=torch.int32, device=device)
        from .draft_sample import DRAFT_K

        # sampled variants: per-request sampling params (staged per round) and per-step
        # uniforms in; the draft's candidates and their probabilities q out, for the acceptance
        self.temp = torch.zeros(self.n_max, dtype=torch.float32, device=device)
        self.top_p = torch.ones(self.n_max, dtype=torch.float32, device=device)
        self.top_k = torch.full((self.n_max,), DRAFT_K, dtype=torch.int64, device=device)
        self.u = torch.full((self.n_max, DRAFT_K), 0.5, dtype=torch.float32, device=device)
        self.cand_ids = torch.zeros(self.n_max, DRAFT_K, dtype=torch.int32, device=device)
        self.cand_q = torch.zeros(self.n_max, DRAFT_K, dtype=torch.float32, device=device)
        # Padding rows: position 0 and the dummy page's slot (safe no-op writes).
        self.pad_positions = torch.zeros(self.n_max, dtype=torch.int32, device=device)
        self.pad_out_loc = (
            get_global_ctx().page_table[self.dummy_table_idx, :1].to(torch.int32)
        ).repeat(self.n_max).contiguous()

        self.attn.init_capture_draft(self.n_max)
        for bs in self.bs_list:
            self._capture(bs)
        # The QSA/FLA kernels call @tensor_cache helpers whose result tensors are baked into
        # these graphs; pin the current entries so a later eager call cannot evict them.
        from freetoken.kernel.fla.utils import pin_all_tensor_caches

        pin_all_tensor_caches()
        logger.info_rank0(f"MTP draft CUDA graphs captured: bs={self.bs_list}")

    # ------------------------------------------------------------------ capture

    def _capture(self, bs: int) -> None:
        # one graph per head variant: the full-vocab argmax and, when a draft vocab is
        # configured, the subset argmax (chosen per round by whether the set is active)
        for subset in (False, True) if self.draft_vocab is not None else (False,):
            for sampled in (False, True):
                self._capture_variant(bs, subset, sampled)

    def _capture_variant(self, bs: int, subset: bool, sampled: bool = False) -> None:
        dummy = self.gr.dummy_req
        reqs = [_DraftReq(dummy.table_idx, 0, 1) for _ in range(bs)]
        self.positions[:bs] = self.pad_positions[:bs]
        self.out_loc[:bs] = self.pad_out_loc[:bs]
        sb = _DraftBatch(reqs, self.positions[:bs], self.out_loc[:bs])
        self.attn.set_draft_n(bs)
        self.attn.stage_draft_round(
            [dummy.table_idx] * bs, [1] * bs, pad_table_idx=dummy.table_idx
        )
        graph = torch.cuda.CUDAGraph()
        with self.attn.draft_scratch():
            sb.attn_metadata = self.attn.make_draft_metadata()
            with get_global_ctx().forward_batch(sb):
                self._forward(bs, sb, subset, sampled)  # eager warmup (materializes scratch)
                # The MTP block is the only QSA layer here, so qsa_forward's slot-0
                # ``_plan_index_writes`` guard does not fire during capture; clear the
                # warmup plan so the per-step index computation is captured (its stale
                # warmup rows would otherwise be replayed against real geometry).
                sb.attn_metadata.cmp_rows = None
                with torch.cuda.graph(graph, pool=self.pool, stream=self.gr.stream):
                    self._forward(bs, sb, subset, sampled)
        self.graph_map[(bs, subset, sampled)] = graph
        if self.pool is None:
            self.pool = graph.pool()  # reuse this graph's mempool for the remaining sizes

    def _forward(self, bs: int, sb: _DraftBatch, subset: bool, sampled: bool = False) -> None:
        R_next, sample_hidden = self.mtp.draft_step(
            self.embed, self.R_in[:bs], self.token_in[:bs], sb
        )
        self.R_out[:bs].copy_(R_next)
        # All rows, NOT forward: the draft batch is phase="prefill", and forward's last-row
        # gather (``x = x[attn_metadata.last_indices]``) indexes a capture-time temporary
        # that is freed once the capture returns (device-side OOB on replay). At t=1 every
        # row IS the last row.
        if sampled:
            from .draft_sample import candidates, sample

            if subset:
                logits, ids = self.draft_vocab.local_logits(sample_hidden), self.draft_vocab.ids
            else:
                start, count = self.lm_head.vocab_range
                logits = self.lm_head._logits(sample_hidden)[:, :count]
                ids = torch.arange(start, start + count, dtype=torch.int32, device=logits.device)
            vals, cid = candidates(logits, ids, getattr(self.lm_head, "_comm", None), self.lm_head.tp_size)
            d, q = sample(vals, cid, self.temp[:bs], self.top_p[:bs], self.top_k[:bs], self.u[:bs])
            self.draft_ids[:bs] = d
            self.cand_ids[:bs] = cid
            self.cand_q[:bs] = q
        elif subset:
            self.draft_ids[:bs] = self.draft_vocab.argmax(sample_hidden)
        else:
            self.draft_ids[:bs] = self.lm_head.forward_argmax(sample_hidden)

    # ------------------------------------------------------------------ replay

    def can_use(self, n: int) -> bool:
        return 0 < n <= self.n_max

    def _select_bs(self, n: int) -> int:
        return next(bs for bs in self.bs_list if bs >= n)

    @torch.inference_mode()
    def replay_step(
        self,
        table_idx: Sequence[int],
        kvlen: Sequence[int],
        positions: torch.Tensor,
        out_loc: torch.Tensor,
        R_in: torch.Tensor,
        tokens: torch.Tensor,
        subset: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage one draft step's geometry + inputs into the static buffers and replay the
        smallest captured size ``>=`` the request count. Returns ``(R_next[:n], draft ids[:n])``."""
        n = len(table_idx)
        assert self.can_use(n), f"draft n={n} outside captured range (max {self.n_max})"
        bs = self._select_bs(n)
        self.R_in[:bs] = 0
        self.R_in[:n].copy_(R_in)
        self.token_in[:bs] = 0
        self.token_in[:n].copy_(tokens)
        self.positions[:bs] = self.pad_positions[:bs]
        self.positions[:n].copy_(positions)
        self.out_loc[:bs] = self.pad_out_loc[:bs]
        self.out_loc[:n].copy_(out_loc)
        self.attn.set_draft_n(bs)
        self.attn.stage_draft_round(list(table_idx), list(kvlen), pad_table_idx=self.dummy_table_idx)
        self.graph_map[(bs, subset, False)].replay()
        return self.R_out[:n], self.draft_ids[:n]

    def can_chain(self) -> bool:
        return getattr(self.attn, "can_stage_draft_dev", lambda: False)()

    @torch.inference_mode()
    def replay_chain(
        self,
        table_idx: Sequence[int],
        cached: Sequence[int],
        R_in: torch.Tensor,
        tokens: torch.Tensor,
        k: int,
        subset: bool = False,
        sampling: torch.Tensor | None = None,
    ) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]] | None]:
        """The whole k-step draft chain with no host sync. Every step's geometry is known at
        round entry (step ``j`` of request ``i`` sits at position ``cached[i] + j`` with KV
        length ``cached[i] + j + 1``), so it goes to the device in one pinned copy and the k
        replays run back to back, each step's outputs feeding the next on the device. (Staging
        step by step from host lists made every step wait for the previous replay.)
        ``sampling`` [3, n] fp32 on the device (temperature, top_p, top_k per request; temperature
        0 = greedy) draws each draft from the head's truncated distribution. Returns the k draft id tensors [n] and, when sampled, each step's
        ``(candidate ids [n, K], q [n, K])``."""
        n = len(table_idx)
        assert self.can_use(n), f"draft n={n} outside captured range (max {self.n_max})"
        bs = self._select_bs(n)
        pad = bs - n
        rows = [list(table_idx) + [self.dummy_table_idx] * pad]
        rows += [[c + j for c in cached] + [0] * pad for j in range(k)]
        rows += [[c + j + 1 for c in cached] + [1] * pad for j in range(k)]
        geo = torch.tensor(rows, dtype=torch.int64).pin_memory().to(self.device, non_blocking=True)
        ti, pos, kvlen = geo[0], geo[1 : 1 + k], geo[1 + k :].to(torch.int32)
        out_loc = get_global_ctx().page_table[ti.unsqueeze(0).expand(k, bs), pos].to(torch.int32)
        pos = pos.to(torch.int32)
        self.R_in[:bs] = 0
        self.R_in[:n].copy_(R_in)
        self.token_in[:bs] = 0
        self.token_in[:n].copy_(tokens)
        self.attn.set_draft_n(bs)
        self.attn.stage_draft_rows_dev(ti)
        sampled = sampling is not None
        if sampled:
            self.temp[:bs] = 0.0
            self.temp[:n].copy_(sampling[0])
            self.top_p[:n].copy_(sampling[1])
            self.top_k[:n].copy_(sampling[2].to(torch.int64))
            # drawn eagerly from the torch generator, which every TP rank advances identically
            u = torch.rand((k, bs, self.u.shape[1]), device=self.device)
        drafts, qs = [], ([] if sampled else None)
        for j in range(k):
            if j:
                self.R_in[:n].copy_(self.R_out[:n])
                self.token_in[:n].copy_(self.draft_ids[:n])
            self.positions[:bs].copy_(pos[j])
            self.out_loc[:bs].copy_(out_loc[j])
            self.attn.stage_draft_kvlen_dev(kvlen[j])
            if sampled:
                self.u[:bs].copy_(u[j])
            self.graph_map[(bs, subset, sampled)].replay()
            drafts.append(self.draft_ids[:n].clone())
            if sampled:
                qs.append((self.cand_ids[:n].clone(), self.cand_q[:n].clone()))
        return drafts, qs


__all__ = ["DraftGraphRunner"]

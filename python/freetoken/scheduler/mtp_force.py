"""Teacher-forced MTP rounds for KLD scoring of the verify path (diagnostics only).

``FREETOKEN_MTP_FORCE_IDS=<file>`` (space-separated token ids of the whole text, re-read when the
file changes): every round's pending token and drafts are the file's tokens at their positions
and every draft is accepted, so the decode follows the text exactly through the captured verify.
``FREETOKEN_DUMP_VERIFY_LOGITS=<file>``: rank 0 appends each round's k+1 verify rows (bf16, row
j = logits after the token at C+j) -- with FREETOKEN_DUMP_PREFILL_LOGITS for the prompt rows this
gives the per-position layout kld/jyo/score.py reads. One request at a time.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch


class MTPForce:
    def __init__(self, ids_path: str, dump_path: Optional[str], primary: bool) -> None:
        self._path = ids_path
        self._dump = dump_path if primary else None
        self._stamp = None
        self._ids: List[int] = []

    @classmethod
    def from_env(cls, primary: bool) -> Optional["MTPForce"]:
        path = os.environ.get("FREETOKEN_MTP_FORCE_IDS")
        return cls(path, os.environ.get("FREETOKEN_DUMP_VERIFY_LOGITS"), primary) if path else None

    def _tokens(self, start: int, count: int) -> List[int]:
        st = os.stat(self._path)
        if (st.st_mtime_ns, st.st_size) != self._stamp:
            self._ids = [int(x) for x in open(self._path).read().split()]
            self._stamp = (st.st_mtime_ns, st.st_size)
        return [self._ids[p] if p < len(self._ids) else 0 for p in range(start, start + count)]

    def apply(self, sched, reqs, C, anchor: torch.Tensor, drafts_gpu: List[torch.Tensor], k: int) -> None:
        """Overwrite the round's pending token (device + host) and drafts with the text's."""
        for i, (r, c) in enumerate(zip(reqs, C)):
            t = self._tokens(c, k + 1)
            sched.token_pool[r.table_idx, c] = t[0]
            anchor[i] = t[0]
            for j in range(k):
                drafts_gpu[j][i] = t[j + 1]
            r._ids_buf[c : c + k + 1] = torch.tensor(t, dtype=r._ids_buf.dtype)

    def accept(self, C, k: int) -> Tuple[List[int], List[int]]:
        return [k] * len(C), [self._tokens(c + k + 1, 1)[0] for c in C]

    def dump(self, logits: torch.Tensor, C) -> None:
        """Append the round's rows; ``<dump>.pos`` gets each round's first position C."""
        if self._dump:
            with open(self._dump, "ab") as f:
                f.write(logits.to(torch.bfloat16).cpu().view(torch.int16).numpy().tobytes())
            with open(self._dump + ".pos", "a") as f:
                f.write(f"{C[0]}\n")

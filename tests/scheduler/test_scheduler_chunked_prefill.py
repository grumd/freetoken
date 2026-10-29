"""Regression test for the chunked-prefill page-accounting bug.

An earlier gemma4 perf change added ``cache_manager.cache_req(req, finished=False)`` to the
``ChunkedReq`` branch of ``Scheduler._process_last_data`` -- caching every *intermediate*
chunk. Under overlap scheduling the next chunk is created (snapshotting the prior chunk's
``cache_handle``) BEFORE that cache_req runs, so each continuation carries a stale handle
whose ``cached_len`` is behind reality and ``cache_req`` re-frees the prior chunk's pages.
Any multi-chunk prefill (codex's large prompts) then crashes the scheduler at the next idle
with ``CacheManager integrity check failed``.

The fix reverts to mini-sglang's behavior: do NOT cache intermediate chunks; the whole
prompt is inserted once when the final (non-chunked) chunk is processed.
"""

from __future__ import annotations

import torch

CHUNK = 8
WIDTH = 64
MAX_RUNNING = 4
UID = 7


def _setup_context() -> None:
    from freetoken.core import Context, get_global_ctx, set_global_ctx

    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=1))


def _build_managers(num_pages):
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import PrefillManager
    from freetoken.scheduler.table import TableManager

    _setup_context()
    pt = torch.zeros((MAX_RUNNING + 1, WIDTH), dtype=torch.int32, device="cpu")
    cm = CacheManager(num_pages=num_pages, page_size=1, page_table=pt, type="radix")
    tm = TableManager(max_running_reqs=MAX_RUNNING, page_table=pt)
    dm = DecodeManager(page_size=1)
    pm = PrefillManager(cm, tm, dm)
    return cm, tm, dm, pm


def _drive_chunked_prefill(cm, tm, pm, n_chunks):
    """Drive one chunked-prefill request through ``n_chunks``, faithfully replicating overlap
    ordering: each iteration schedules+forwards the NEXT chunk (reading the prior chunk's
    cache_handle) BEFORE the PREVIOUS chunk is cached. Intermediate chunks are NOT cached
    (like the real scheduler's ``continue``); the whole prompt is cached once when the final
    non-chunked chunk is processed."""
    from freetoken.core import SamplingParams
    from freetoken.scheduler.prefill import ChunkedReq
    from freetoken.scheduler.utils import PendingReq

    prompt_len = n_chunks * CHUNK
    pm.pending_list = [PendingReq(uid=UID, input_ids=torch.arange(prompt_len, dtype=torch.int32),
                                  sampling_params=SamplingParams(max_tokens=4))]
    last_batch = None
    final_req = None
    while pm.runnable or last_batch is not None:
        batch = pm.schedule_next_batch(CHUNK)          # step 2: schedule next chunk
        if batch is not None:
            cm.allocate_paged(batch.reqs)              # _prepare_batch allocates pages
            for r in batch.reqs:
                r.complete_one()                       # forward advances cached_len
        if last_batch is not None:                     # step 3: process the PREVIOUS batch
            for r in last_batch.reqs:
                if not isinstance(r, ChunkedReq):
                    cm.cache_req(r, finished=False)    # final chunk cached once
                    final_req = r
        last_batch = batch
    cm.cache_req(final_req, finished=True)             # request finishes -> release
    tm.free(final_req.table_idx)


def test_multichunk_overlap_no_double_free():
    """The fix: not caching intermediate chunks keeps page accounting consistent across a
    multi-chunk prefill, and still caches the whole prompt (reusable) exactly once."""
    cm, tm, _dm, pm = _build_managers(num_pages=64)
    _drive_chunked_prefill(cm, tm, pm, n_chunks=4)

    cm.check_integrity()  # free_slots + cache == num_pages; no chunk freed twice
    si = cm.prefix_cache.size_info
    assert si.protected_size == 0  # request released -> no leaked locks / ref_count drift
    assert si.evictable_size == 4 * CHUNK  # whole prompt retained in the prefix cache
    assert len(cm.free_slots) == cm.num_pages - 4 * CHUNK


def test_radix_hit_admission_reports_nonzero_cached_tokens():
    """A second request sharing a cached prefix admits with the prefix-cache hit recorded."""
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    cm, tm, _dm, pm = _build_managers(num_pages=64)
    _drive_chunked_prefill(cm, tm, pm, n_chunks=4)  # whole 32-token prompt now in the radix cache

    cached_len = 4 * CHUNK
    prompt_len = cached_len + CHUNK
    pm.pending_list = [
        PendingReq(uid=UID + 1, input_ids=torch.arange(prompt_len, dtype=torch.int32),
                   sampling_params=SamplingParams(max_tokens=4))
    ]
    batch = pm.schedule_next_batch(prompt_len)
    assert batch is not None
    assert batch.prompt_admissions == [(UID + 1, prompt_len, cached_len)]
    assert batch.log_cached_tokens == cached_len


def test_chunked_prompt_admission_reports_complete_length_once():
    """The first prepared chunk carries the full prompt usage; continuations carry none."""
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    cm, _tm, _dm, pm = _build_managers(num_pages=64)
    prompt_len = 3 * CHUNK
    pm.pending_list = [
        PendingReq(
            uid=UID,
            input_ids=torch.arange(prompt_len, dtype=torch.int32),
            sampling_params=SamplingParams(max_tokens=4),
        )
    ]

    admissions = []
    while pm.runnable:
        batch = pm.schedule_next_batch(CHUNK)
        assert batch is not None
        admissions.append(list(batch.prompt_admissions))
        cm.allocate_paged(batch.reqs)
        for req in batch.reqs:
            req.complete_one()

    assert admissions[0] == [(UID, prompt_len, 0)]
    assert all(items == [] for items in admissions[1:])


def test_batched_prefill_carries_each_new_prompt_admission():
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    _cm, _tm, _dm, pm = _build_managers(num_pages=64)
    pm.pending_list = [
        PendingReq(1, torch.arange(3, dtype=torch.int32), SamplingParams(max_tokens=2)),
        PendingReq(2, torch.arange(5, dtype=torch.int32), SamplingParams(max_tokens=2)),
    ]
    batch = pm.schedule_next_batch(16)
    assert batch is not None
    assert batch.prompt_admissions == [(1, 3, 0), (2, 5, 0)]



IMG = 900  # image placeholder id; the text ids below stay well under it


def _mm_sched():
    from types import SimpleNamespace

    from freetoken.scheduler.scheduler import Scheduler

    sched = Scheduler.__new__(Scheduler)
    sched.config = SimpleNamespace(model_config=SimpleNamespace(image_token_id=IMG))
    return sched


def _image(seed):
    """One 6-soft-token image: grid 1x4x6 = 24 patch rows, 4 rows per soft token."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(24, 3, generator=g), torch.tensor([1, 4, 6])


class _Tower:
    """Stands in for Scheduler._embed_images: rows derived from the digest, every call
    recorded, optionally failing."""

    def __init__(self):
        self.calls = []
        self.fail = None

    def __call__(self, images):
        self.calls.append([img.digest for img in images])
        if self.fail is not None:
            return self.fail
        for img in images:
            img.embeds = self.rows(img)
        return None

    @staticmethod
    def rows(img):
        return float(img.digest[0]) + torch.arange(img.count * 4, dtype=torch.float32).view(-1, 4)


def _mm_managers(num_pages=128):
    cm, tm, dm, pm = _build_managers(num_pages)
    pm.embed_images = tower = _Tower()
    return cm, tm, pm, tower


def _mm_pending(sched, uid, segments, images):
    """A PendingReq for ``segments`` (lists of text ids, or None for the next image) built by
    the real intake path, Scheduler._build_mm."""
    from types import SimpleNamespace

    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    ids = []
    for seg in segments:
        ids += [IMG] * 6 if seg is None else seg
    msg = SimpleNamespace(
        uid=uid, input_ids=torch.tensor(ids, dtype=torch.int32), mm_embeds=None,
        pixel_values=torch.cat([p for p, _ in images]),
        image_grid_thw=torch.stack([g for _, g in images]),
    )
    mm, error = sched._build_mm(msg)
    assert error is None
    return PendingReq(uid, msg.input_ids, SamplingParams(max_tokens=4), mm=mm)


def test_multimodal_prompt_chunks_with_each_chunks_soft_tokens():
    """An image prompt longer than the chunk budget used to raise NotImplementedError in the
    scheduler loop and take the server down. Chunk boundaries now fall anywhere, inside
    images too, and each chunk scatters exactly the soft tokens of its own placeholders."""
    from freetoken.scheduler.scheduler import Scheduler

    cm, _tm, pm, tower = _mm_managers()
    sched = _mm_sched()
    pend = _mm_pending(sched, UID, [[1, 2, 3], None, [4, 5, 6, 7], None, [8, 9, 10]],
                       [_image(0), _image(1)])
    pm.pending_list = [pend]

    gathered = []
    while pm.runnable:
        batch = pm.schedule_next_batch(CHUNK)
        assert batch is not None
        # admitted: the pixels are dropped right away, only the soft tokens ride along
        assert all(img.pixel_values is None for img in pend.mm.images)
        cm.allocate_paged(batch.reqs)
        Scheduler._gather_multimodal(sched, batch)
        (req,) = batch.reqs
        slots = int((req.input_ids[req.cached_len : req.device_len] == IMG).sum())
        rows = 0 if batch.mm_embeds is None else batch.mm_embeds.shape[0]
        assert rows == slots  # what the model's masked_scatter needs
        if rows:
            gathered.append(batch.mm_embeds)
        req.complete_one()

    assert len(gathered) == 3  # 22 tokens in chunks of 8: both images straddle a boundary
    expect = torch.cat([_Tower.rows(img) for img in pend.mm.images])
    assert torch.equal(torch.cat(gathered), expect)
    assert len(tower.calls) == 1  # both images embedded once, at admission
    assert all(img.embeds is None for img in pend.mm.images)  # VRAM released after prefill


def test_pixels_survive_a_refused_admission():
    """A request the pool cannot seat yet stays queued with its pixels: its images may still
    need the vision tower when it is finally admitted."""
    cm, _tm, pm, tower = _mm_managers(num_pages=8)
    pend = _mm_pending(_mm_sched(), UID, [[1, 2, 3], None, [4, 5, 6, 7]], [_image(0)])
    pm.pending_list = [pend]
    assert pm.schedule_next_batch(64) is None  # 13 prompt + 4 output tokens > 8 pages
    assert pm.pending_list == [pend] and not pm.failed
    assert tower.calls == [] and pend.mm.images[0].pixel_values is not None


def _run_to_finish(cm, tm, pm, pend):
    from freetoken.scheduler.scheduler import Scheduler

    pm.pending_list = [pend]
    batch = pm.schedule_next_batch(64)
    cm.allocate_paged(batch.reqs)
    Scheduler._gather_multimodal(_mm_sched(), batch)
    (req,) = batch.reqs
    req.complete_one()
    cm.cache_req(req, finished=True)
    tm.free(req.table_idx)
    return batch


def test_multimodal_prefix_reused_only_for_the_same_image():
    """Image requests used to bypass the prefix cache, so every turn after an image
    re-prefilled the whole conversation. They now key the tree on image content: the next
    turn with the same image reuses past it, while a different image -- same placeholder
    ids -- or a text-only prompt stops at the first placeholder."""
    cm, tm, pm, _tower = _mm_managers()
    sched = _mm_sched()
    head, tail = [1, 2, 3], [4, 5, 6, 7, 8, 9, 10]
    first = _mm_pending(sched, 1, [head, None, tail], [_image(0)])
    _run_to_finish(cm, tm, pm, first)
    prompt_len = first.input_len

    more = [11, 12, 13]
    same = _mm_pending(sched, 2, [head, None, tail + more], [_image(0)])
    batch = _run_to_finish(cm, tm, pm, same)
    assert batch.log_cached_tokens == prompt_len

    other = _mm_pending(sched, 3, [head, None, tail + more], [_image(1)])
    batch = _run_to_finish(cm, tm, pm, other)
    assert batch.log_cached_tokens == len(head)

    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    text = PendingReq(4, same.input_ids.clone(), SamplingParams(max_tokens=4))
    batch = _run_to_finish(cm, tm, pm, text)
    assert batch.log_cached_tokens == len(head)
    cm.check_integrity()


def test_images_inside_the_prefix_hit_are_never_embedded():
    """Re-encoding every image in the history on every turn cost seconds per image. Only
    images past the prefix-cache hit reach the vision tower now."""
    cm, tm, pm, tower = _mm_managers()
    sched = _mm_sched()
    turn1 = _mm_pending(sched, 1, [[1, 2], None, [3, 4]], [_image(0)])
    _run_to_finish(cm, tm, pm, turn1)
    assert len(tower.calls) == 1

    turn2 = _mm_pending(sched, 2, [[1, 2], None, [3, 4, 5, 6]], [_image(0)])
    _run_to_finish(cm, tm, pm, turn2)
    assert len(tower.calls) == 1  # the image sits in the reused prefix

    turn3 = _mm_pending(sched, 3, [[1, 2], None, [3, 4, 5, 6, 7], None, [8]],
                        [_image(0), _image(1)])
    _run_to_finish(cm, tm, pm, turn3)
    assert tower.calls[-1] == [turn3.mm.images[1].digest]  # just the new image


def test_embed_failure_drops_only_that_request():
    """The vision tower failing at admission fails that request with an error the scheduler
    sends; it neither crashes the loop nor blocks the requests queued behind it."""
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    cm, tm, pm, tower = _mm_managers()
    sched = _mm_sched()
    tower.fail = "image encoding failed: boom"
    bad = _mm_pending(sched, 1, [[1, 2], None, [3]], [_image(0)])
    good = PendingReq(2, torch.arange(5, dtype=torch.int32), SamplingParams(max_tokens=4))
    pm.pending_list = [bad, good]
    assert pm.schedule_next_batch(64) is None
    assert pm.failed == [(1, "image encoding failed: boom")]
    assert pm.pending_list == [good]
    batch = pm.schedule_next_batch(64)
    assert [r.uid for r in batch.reqs] == [2]
    assert cm.prefix_cache.size_info.protected_size == 0  # the failed admission unlocked


def test_multimodal_placeholder_mismatch_is_a_request_error():
    """A prompt whose placeholders disagree with its images is rejected at intake, before
    the forward's masked_scatter could fail inside the scheduler loop."""
    from types import SimpleNamespace

    pv, grid = _image(0)
    msg = SimpleNamespace(uid=1, input_ids=torch.tensor([1, IMG, IMG, IMG, IMG, IMG, 2],
                                                        dtype=torch.int32),
                          mm_embeds=None, pixel_values=pv, image_grid_thw=grid.view(1, -1))
    mm, error = _mm_sched()._build_mm(msg)
    assert mm is None and "5 image placeholder tokens" in error


def test_multimodal_key_survives_a_later_image():
    """Appending an image in a later turn must not change the earlier image's key, or no
    turn that adds an image could reuse the conversation so far."""
    sched = _mm_sched()
    one = _mm_pending(sched, 1, [[1, 2], None, [3]], [_image(0)])
    two = _mm_pending(sched, 2, [[1, 2], None, [3], None, [4]], [_image(0), _image(1)])
    n = one.input_len
    assert torch.equal(two.mm.key[:n], one.mm.key)
    assert (one.mm.key[2:8] < 0).all() and torch.equal(one.mm.key[:2], one.input_ids[:2])

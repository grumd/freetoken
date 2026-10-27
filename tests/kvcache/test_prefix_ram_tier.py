"""Prefix RAM tier (--cache-ram-gb): an evicted conversation and its GDN snapshot spill to host
RAM instead of being dropped, a later request with that prefix reloads them byte for byte, a
continuation adopts RAM-only nodes it recomputed, and a full RAM tier drops its oldest entries."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager

PAGES = 256


class _Host:
    """HostPrefixPool's interface over CPU tensors: ``payload`` stands in for the KV pages
    (one value per device page) and the pool's recurrent state for the snapshots."""

    def __init__(self, num_pages, num_snaps, payload, state):
        self.num_pages, self.num_snaps = num_pages, num_snaps
        self._free_pages, self._free_snaps = list(range(num_pages)), list(range(num_snaps))
        self.payload, self.state = payload, state
        self.pages = torch.full((num_pages,), -1, dtype=payload.dtype)
        self.snaps = torch.zeros((num_snaps, *state.shape[2:]), dtype=state.dtype)

    free_pages = property(lambda self: len(self._free_pages))
    free_snaps = property(lambda self: len(self._free_snaps))

    def alloc_pages(self, n):
        out, self._free_pages = self._free_pages[:n], self._free_pages[n:]
        return out

    def release_pages(self, pages):
        self._free_pages = sorted(self._free_pages + list(pages))

    def alloc_snap(self):
        return self._free_snaps.pop(0)

    def release_snap(self, s):
        self._free_snaps = sorted(self._free_snaps + [s])

    def save_pages(self, dev, host):
        self.pages[host] = self.payload[dev.long()]

    def load_pages(self, host, dev):
        self.payload[dev.long()] = self.pages[host]

    def save_snap(self, slot, h):
        self.snaps[h] = self.state[0, slot]

    def load_snap(self, h, slot):
        self.state[0, slot] = self.snaps[h]


def _setup(host_pages=64, host_snaps=4, state_slots=8):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    pool = LinearStatePool(group=g, num_slots=state_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)
    payload = torch.full((PAGES,), -1, dtype=torch.int64)
    host = _Host(host_pages, host_snaps, payload, pool.recurrent_states)
    cm = CacheManager(PAGES, 1, torch.zeros(4, 512, dtype=torch.int32), "hybrid_radix",
                      linear_state_pool=pool, host_pool=host)
    return cm, host, payload, pool


def _pend(ids):
    return SimpleNamespace(input_ids=torch.tensor(ids, dtype=torch.int32), input_len=len(ids), mm_embeds=None)


def _run(cm, payload, pool, ids, tag):
    """Admit ``ids``, 'prefill' what the tree lacks (stamping each new page with tag*1000+pos
    and the live GDN state with ``tag``), finish, and return the match it was admitted with."""
    from freetoken.core import Req, SamplingParams

    mr = cm.match_req(_pend(ids))
    h = mr.cuda_handle
    cm.lock(h)
    if pool.num_free_slots < 3:
        cm.ensure_mamba_slots(3)  # PrefillAdder's admission gate
    cm.note_admission(h)
    n = len(ids) - 1  # the last token is the decode input
    new = cm._allocate(n - h.cached_len)
    row = torch.cat([h.get_matched_indices(), new]).to(torch.int32)
    cm.page_table[0, :n] = row
    payload[new.long()] = torch.arange(h.cached_len, n) + tag * 1000
    req = Req(input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=0, cached_len=n,
              output_len=1, uid=tag, sampling_params=SamplingParams(), cache_handle=h)
    req.linear_slot_idx, req.mamba_ping_pong = pool.alloc(1)[0], tuple(pool.alloc(2))
    pool.recurrent_states[0, req.linear_slot_idx] = tag
    req.mamba_next_track_idx, req.mamba_last_track_seqlen = 1, None
    cm.cache_req(req, finished=True)
    cm.check_integrity()
    return mr


def _session(tag, length):
    return [tag * 100 + i for i in range(length)] + [7]


def test_a_conversation_whose_snapshot_was_evicted_comes_back_from_ram():
    cm, host, payload, pool = _setup()
    a = _session(1, 40)
    _run(cm, payload, pool, a, tag=1)
    # other conversations need GDN slots: A's snapshot (and KV) spill to RAM instead of dying
    for t in range(2, 7):
        _run(cm, payload, pool, _session(t, 8), tag=t)
    assert host.free_snaps < host.num_snaps
    spilled = [n for n in cm.prefix_cache._all_nodes() if n.value is None]
    assert spilled and all(n.host_pages is not None for n in spilled)

    mr = _run(cm, payload, pool, a + [9, 9], tag=99)  # A's next turn
    assert mr.cuda_handle.cached_len == 40
    pages = mr.cuda_handle.get_matched_indices().long()
    assert payload[pages].tolist() == [1000 + i for i in range(40)]  # the KV bytes came back
    assert float(pool.recurrent_states[0, mr.mamba_value].float().mean()) == 1.0  # and the state


def test_cold_spill_without_snapshot_pressure_and_reload():
    cm, host, payload, pool = _setup(state_slots=32)
    a = _session(1, 30)
    _run(cm, payload, pool, a, tag=1)
    for t in range(2, 6):
        _run(cm, payload, pool, _session(t, 6), tag=t)
    free_before = len(cm.free_slots)
    cm.keep_requests = 2
    cm._evict_cold()  # A is cold now: it spills
    assert len(cm.free_slots) > free_before
    assert host.free_pages < host.num_pages
    mr = _run(cm, payload, pool, a + [5], tag=50)
    assert mr.cuda_handle.cached_len == 30
    assert payload[mr.cuda_handle.get_matched_indices().long()].tolist() == [1000 + i for i in range(30)]


def test_a_full_ram_tier_drops_its_oldest_conversation():
    cm, host, payload, pool = _setup(host_pages=48, host_snaps=3, state_slots=32)
    cm.keep_requests = 1
    for t in (1, 2, 3, 4):
        _run(cm, payload, pool, _session(t, 20), tag=t)
        cm._evict_cold()  # spills every conversation but the newest
    # 3 x 20 spilled pages do not fit in 48: the oldest (1) was dropped to make room for 3
    assert host.free_pages == 8
    assert _run(cm, payload, pool, _session(1, 20) + [3], tag=10).cuda_handle.cached_len == 0
    mr = _run(cm, payload, pool, _session(3, 20) + [3], tag=11)
    assert mr.cuda_handle.cached_len == 20
    assert payload[mr.cuda_handle.get_matched_indices().long()].tolist() == [3000 + i for i in range(20)]


def test_a_continuation_adopts_ram_only_nodes_it_recomputed():
    cm, host, payload, pool = _setup(state_slots=32)
    a = _session(1, 30)
    _run(cm, payload, pool, a, tag=1)
    # split A's node at 16 with a sibling, then push both out to RAM
    _run(cm, payload, pool, a[:16] + [55, 56, 7], tag=2)
    cm.keep_requests = 1
    _run(cm, payload, pool, _session(3, 4), tag=3)
    cm._evict_cold()
    pc = cm.prefix_cache
    # a request that matches only the device part, then recomputes A's RAM-only tail itself
    for n in pc._all_nodes():
        n.host_mamba is not None and host.release_snap(n.host_mamba)
        n.host_mamba = None if n.value is None else n.host_mamba
    cm.check_integrity = lambda: None  # the hand edit above leaves snapshot-less RAM leaves
    mr = _run(cm, payload, pool, a + [8], tag=4)
    assert mr.cuda_handle.cached_len == 0
    adopted = [n for n in pc._all_nodes() if n.value is not None and n.host_pages is not None]
    assert adopted, "the recomputed span should back the RAM-only nodes again"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_host_pool_round_trips_qsa_pages_and_gdn_snapshots():
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache.host_prefix_pool import HostPrefixPool
    from freetoken.kvcache.qsa_pool import QSAKVCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    dev = torch.device("cuda", torch.cuda.current_device())
    kv = QSAKVCache(num_kv_heads=1, num_layers=8, head_dim=256, num_pages=65, page_size=64,
                    dtype=torch.bfloat16, device=dev, index_head_dim=128, num_index_layers=2,
                    index_ratio=4, num_req_slots=2, layer_ids=[1, 3, 5, 7])
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0, 2), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    lsp = LinearStatePool(group=g, num_slots=4, dtype=torch.bfloat16, device=dev, tp_size=1)
    views = kv.page_views()
    for v in views:
        v.copy_(torch.randn(v.shape, device=dev).to(v.dtype))
    lsp.conv_states.normal_()
    lsp.recurrent_states.normal_()
    host = HostPrefixPool(views, [lsp.conv_states, lsp.recurrent_states], 64 << 20, 64)
    assert host.num_pages > 8 and host.num_snaps >= 2
    src = torch.tensor([9, 3, 4, 40, 41])
    before = [v[src].clone() for v in views]
    hp = host.alloc_pages(5)
    host.save_pages(src, hp)
    snap = (lsp.conv_states[:, 2].clone(), lsp.recurrent_states[:, 2].clone())
    hs = host.alloc_snap()
    host.save_snap(2, hs)
    for v in views:
        v.zero_()
    lsp.conv_states.zero_()
    lsp.recurrent_states.zero_()
    dst = torch.tensor([60, 1, 2, 30, 50])
    host.load_pages(hp, dst)
    host.load_snap(hs, 3)
    for v, b in zip(views, before):
        assert torch.equal(v[dst], b)
    assert torch.equal(lsp.conv_states[:, 3], snap[0]) and torch.equal(lsp.recurrent_states[:, 3], snap[1])

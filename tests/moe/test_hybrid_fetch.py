"""Hybrid decode's bandwidth-matched fetch split.

Covers the two halves of --moe-hybrid-max-fetch auto: the profile reader that turns
`ft bench bw` kernel bandwidths into a fetch fraction, and the ensure kernel's
per-step integer split (GPU kernel vs CPU reference mirror, and the balance rule).
"""

import json
import os

import pytest
import torch

from freetoken.moe.bench_profile import default_profile_path, load_backend_recommendation, load_hybrid_fetch_fraction
from freetoken.moe.offload_cache import OffloadMoeCache

Q = 1 << 16


def _balanced_fetch(num_missing: int, frac_q16: int) -> int:
    """Reference split: F ~ frac * misses, rounded to whichever integer neighbor
    minimizes the slower overlapped side (fetch ~ F*(1-frac), CPU ~ (M-F)*frac)."""
    lo = (num_missing * frac_q16) >> 16
    cost = lambda f: max(f * (Q - frac_q16), (num_missing - f) * frac_q16)  # noqa: E731
    return min(num_missing, lo if cost(lo) <= cost(lo + 1) else lo + 1)


def test_balanced_fetch_tracks_fraction():
    # The split follows fetched : cpu = pcie : (cpu - pcie) up to integer rounding, and
    # never over/under-shoots by more than one expert.
    for frac in (0.1, 0.415, 0.454, 0.7, 1.0):
        q = round(frac * Q)
        for m in range(0, 65):
            f = _balanced_fetch(m, q)
            assert 0 <= f <= m
            assert abs(f - frac * m) <= 1.0
    # ceil would over-fetch here (the regression this rule fixed): 41.5% of 3 misses is
    # 1.24 -> fetching 2 makes the PCIe side ~1.6x slower than balance; keep it at 1.
    assert _balanced_fetch(3, round(0.415 * Q)) == 1
    assert _balanced_fetch(4, round(0.415 * Q)) == 2


def test_load_hybrid_fetch_fraction(tmp_path):
    prof = {
        "gpu": {"name": "FAKE GPU"},
        "dtype_kernels": {
            "bf16": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0},
            # overlapped (contended) pair wins over the standalone numbers when present
            "nvfp4_x": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0,
                        "cpu_moe_overlap_gbs": 90.0, "pcie_gather_overlap_gbs": 30.0},
        },
        "workloads": {
            "m": {"kernels": {"ds_fp4": {"cpu_moe_gbs": 80.0, "pcie_gather_gbs": 50.0}}}
        },
    }
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    # standalone fallback: full-contention assumption -> pcie / cpu
    assert load_hybrid_fetch_fraction("bf16", path=str(path)) == pytest.approx(0.4)
    # overlapped pair preferred: pcie_ov / (pcie_ov + cpu_ov)
    assert load_hybrid_fetch_fraction("nvfp4_x", path=str(path)) == pytest.approx(0.25)
    # per-model fallback when there is no per-dtype entry for the format
    assert load_hybrid_fetch_fraction("ds_fp4", path=str(path)) == pytest.approx(0.625)
    assert load_hybrid_fetch_fraction("nvfp4", path=str(path)) is None
    # a profile from different hardware is ignored
    assert load_hybrid_fetch_fraction("bf16", gpu_name="OTHER", path=str(path)) is None


def test_profile_lookup_prefers_the_gpu_uuid_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("FREETOKEN_BENCHBW_PATH", raising=False)
    uuid = "GPU-2f3a9b1c-0000-1111-2222-333344445555"

    def write(path, name, verdict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"gpu": {"name": name}, "dtypes": {"bf16": verdict}}, f)

    # legacy single file only: used when the name matches, ignored otherwise
    write(default_profile_path(), "FAKE GPU", "hybrid")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "hybrid"
    assert load_backend_recommendation("bf16", gpu_name="OTHER", gpu_uuid=uuid) is None
    # this card's own file wins over the legacy one
    write(default_profile_path(uuid), "FAKE GPU", "offload")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "offload"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_fraction_gpu_matches_cpu_reference():
    torch.manual_seed(0)
    num_experts, cache_size, top_k, frac = 32, 40, 8, 0.415

    def make():
        return OffloadMoeCache(
            num_layers=2, num_experts=num_experts, cache_size=cache_size,
            device=torch.device("cuda"), quant_format="bf16", decode_target="hybrid",
            hybrid_max_fetch=num_experts, hybrid_fetch_fraction=frac,
        )

    gpu, ref = make(), make()
    frac_q16 = round(frac * Q)
    for step in range(64):
        ids = torch.randperm(num_experts)[:top_k].to(torch.int32)
        g, c = ids.clone().cuda(), ids.clone()  # a CPU ids tensor drives the reference path
        gpu.ensure_experts_hybrid(0, g)
        ref.ensure_experts_hybrid(0, c)
        missing = int(gpu.num_missing_full.item())
        fetched = int(gpu.num_indices.item())
        assert missing == int(ref.num_missing_full.item())
        assert fetched == int(ref.num_indices.item()) == _balanced_fetch(missing, frac_q16)
        # slot rewrites (hit/fetched -> slot, overflow -> -1) and LRU state stay identical
        assert torch.equal(g.cpu(), c)
        assert torch.equal(gpu.slot_for_id.cpu(), ref.slot_for_id.cpu())
        assert torch.equal(gpu.id_of_slot.cpu(), ref.id_of_slot.cpu())
        assert (g >= 0).sum().item() == len(set(ids.tolist())) - (missing - fetched)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_fixed_cap_unchanged():
    # fraction 0 (no profile / explicit --moe-hybrid-max-fetch) keeps the fixed cap.
    cache = OffloadMoeCache(
        num_layers=1, num_experts=32, cache_size=40, device=torch.device("cuda"),
        quant_format="bf16", decode_target="hybrid", hybrid_max_fetch=1,
    )
    ids = torch.arange(8, dtype=torch.int32).cuda()
    cache.ensure_experts_hybrid(0, ids)
    assert int(cache.num_missing_full.item()) == 8
    assert int(cache.num_indices.item()) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_prefill_capped_fetch_takes_the_heaviest_misses():
    """Hybrid routed prefill: given per-expert token counts, the fetched share of the misses is
    the experts serving the most tokens (the CPU keeps the light ones); GPU kernel == reference."""
    torch.manual_seed(0)
    num_experts, cache_size = 64, 80

    def make(device):
        return OffloadMoeCache(num_layers=2, num_experts=num_experts, cache_size=cache_size,
                               device=torch.device(device), quant_format="bf16")

    gpu, ref = make("cuda"), make("cpu")
    for step in range(8):
        routed = torch.randperm(num_experts)[:40].sort().values.to(torch.int32).view(1, -1)
        counts = torch.randperm(200)[:40] + 1          # distinct, so the ranking is unambiguous
        g, c = routed.clone().cuda(), routed.clone()
        gpu.ensure_experts_capped(1, g, 0.3, counts.cuda())
        ref.ensure_experts_capped(1, c, 0.3, counts)
        assert torch.equal(g.cpu(), c)
        assert torch.equal(gpu.slot_for_id.cpu(), ref.slot_for_id.cpu())
        missing, fetched = int(gpu.num_missing_full.item()), int(gpu.num_indices.item())
        assert fetched == _balanced_fetch(missing, round(0.3 * Q))
        src = gpu.src_indices[:fetched].cpu().tolist()
        slot = dict(zip(routed.view(-1).tolist(), c.view(-1).tolist()))
        tokens = dict(zip(routed.view(-1).tolist(), counts.tolist()))
        misses = [e for e in slot if e in src or slot[e] < 0]
        assert sorted(src) == sorted(sorted(misses, key=lambda e: -tokens[e])[:fetched])


def _split_reference(slot_for_id, usage, id_of_slot, recency, credit, step, layer, ids, frac_q16, E):
    """Python model of ensure_experts_split for one layer: returns (slot per route, fetched experts)."""
    base = layer * E
    active = list(dict.fromkeys(ids))
    missing = [e for e in active if slot_for_id[base + e] < 0]
    credit[layer] += len(missing) * frac_q16
    n_fetch = min(len(missing), credit[layer] >> 16)
    credit[layer] -= n_fetch << 16
    for e in active:
        if slot_for_id[base + e] >= 0:
            usage[slot_for_id[base + e]] = step
    order = sorted(missing, key=lambda e: (-recency[base + e], e))[:n_fetch]
    owners = {slot_for_id[base + e] for e in active if slot_for_id[base + e] >= 0}
    for e in order:
        victim = min((s for s in range(len(usage)) if s not in owners), key=lambda s: (usage[s], s))
        owners.add(victim)
        if id_of_slot[victim] >= 0:
            slot_for_id[id_of_slot[victim]] = -1
        id_of_slot[victim] = base + e
        slot_for_id[base + e] = victim
        usage[victim] = step
    for e in active:
        recency[base + e] = step
    return [slot_for_id[base + e] for e in ids], order


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("frac", [0.0, 0.1, 0.5, 1.0])
def test_decode_split_matches_reference(frac):
    """CPU-assisted decode's ensure: fetched share by per-layer credit (exact in the long run),
    most recently routed misses first, LRU victims never among this step's routed slots, and the
    CPU's inputs (x, expert ids of the unfetched misses, weights) staged in pinned memory."""
    torch.manual_seed(1)
    E, S, K, H, L = 48, 64, 8, 256, 2
    cache = OffloadMoeCache(num_layers=L, num_experts=E, cache_size=S, device=torch.device("cuda"),
                            quant_format="bf16")
    cache.cpu_decode_fetch_q16 = q = round(frac * Q)
    io = {"x": torch.zeros(1, H, dtype=torch.bfloat16, pin_memory=True),
          "ids": torch.zeros(1, K, dtype=torch.int32, pin_memory=True),
          "w": torch.zeros(1, K, dtype=torch.float32, pin_memory=True)}
    slot_for_id, usage, id_of_slot = [-1] * (L * E), [0] * S, [-1] * S
    recency, credit = [-1] * (L * E), [0] * L
    total_missing = total_fetched = 0
    # a skewed routing so experts recur
    probs = torch.softmax(torch.randn(L, E) * 2, -1)
    for step in range(1, 301):
        layer = step % L
        ids = torch.multinomial(probs[layer], K).to(torch.int32)
        x = torch.randn(1, H, device="cuda").to(torch.bfloat16)
        w = torch.softmax(torch.randn(1, K, device="cuda"), -1)
        g = ids.view(1, K).cuda()
        cache.ensure_experts_split(layer, g, w, x, io)
        torch.cuda.synchronize()
        want, order = _split_reference(slot_for_id, usage, id_of_slot, recency, credit, step, layer,
                                       ids.tolist(), q, E)
        assert g.view(-1).tolist() == want
        n = int(cache.num_indices.item())
        assert cache.src_indices[:n].tolist() == order
        assert cache.id_of_slot.tolist() == id_of_slot
        on_cpu = [e if s < 0 else -1 for e, s in zip(ids.tolist(), want)]
        assert io["ids"].view(-1).tolist() == on_cpu
        assert torch.equal(io["x"], x.cpu()) and torch.equal(io["w"], w.cpu())
        total_missing += sum(1 for e in dict.fromkeys(ids.tolist()) if e in order or e in on_cpu)
        total_fetched += n
    assert cache.fetch_credit.tolist() == credit
    assert abs(total_fetched - frac * total_missing) < L

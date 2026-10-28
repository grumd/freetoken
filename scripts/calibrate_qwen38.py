#!/usr/bin/env python3
"""One-time machine calibration of CPU-assisted decode for serve-qwen38.sh (run through
calibrate-qwen38.sh, which starts and stops the server).

probe  (no server, GPUs idle): GPU / PCIe / CPU facts, pinned host->device bandwidth per GPU, and
       the CPU expert kernel's time per expert on one layer of the checkpoint's real NVFP4 experts.
run    two loads of serve-qwen38.sh: DECODE_FETCH=1 (plain decode, the reference), then CPU-assisted
       decode with FREETOKEN_CPU_DECODE_TUNABLE=1, where the internal /v1/admin/decode-tune
       endpoint switches the share and the CPU thread count between requests. Greedy streaming
       decodes at a short and a long context; picks the recommendation, confirms it and writes
       the env file and the report.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # torch ordinals = nvidia-smi indices
GRID = (1.0, 0.5, 0.25, 0.15, 0.1, 0.05)
TIE = 0.02      # results this close to the best count as the same (noise of a short run)
MARGIN = 0.05   # CPU help must beat DECODE_FETCH=1 by this much at every depth to be recommended
QUESTIONS = (
    "Using the material above, write a long and detailed technical walkthrough of how its main "
    "components fit together, with examples, trade-offs and possible improvements.",
    "Based on the text above, write an extensive tutorial for a new contributor: setup, the most "
    "important modules one by one, common pitfalls, and a list of exercises with solutions.",
)


# ----------------------------------------------------------------------------------------------
# probe


def _cpu_facts() -> dict:
    info = open("/proc/cpuinfo").read()
    model = next((ln.split(":", 1)[1].strip() for ln in info.splitlines() if ln.startswith("model name")), "?")
    flags = set(next((ln.split(":", 1)[1].split() for ln in info.splitlines() if ln.startswith("flags")), []))
    mem_kib = next(int(ln.split()[1]) for ln in open("/proc/meminfo") if ln.startswith("MemTotal:"))
    return {
        "model": model,
        "logical_cpus": os.cpu_count(),
        "avx2": "avx2" in flags and "fma" in flags,
        "avx512f": "avx512f" in flags,
        "avx512_bf16": "avx512_bf16" in flags,
        "avx512_vnni": "avx512_vnni" in flags,
        "avx_vnni": "avx_vnni" in flags,
        "ram_gib": round(mem_kib / 2**20, 1),
    }


def _gpu_facts(gpus: list[int]) -> list[dict]:
    fields = "index,name,memory.total,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    rows = {}
    for line in out.strip().splitlines():
        idx, name, mem, gen, width, gen_max, width_max = (v.strip() for v in line.split(","))
        rows[int(idx)] = {"index": int(idx), "name": name, "memory_mib": int(mem), "pcie_gen": gen,
                          "pcie_width": width, "pcie_gen_max": gen_max, "pcie_width_max": width_max}
    return [rows[g] for g in gpus]


def _h2d_gbps(devices: list[int], mib: int = 256, reps: int = 8) -> float:
    """Pinned host->device bandwidth, all ``devices`` copying at once; the aggregate GB/s."""
    import torch

    n = mib << 20
    src = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    dst = [torch.empty(n, dtype=torch.uint8, device=f"cuda:{d}") for d in devices]
    streams = [torch.cuda.Stream(device=f"cuda:{d}") for d in devices]
    for _ in range(2):
        for t, s in zip(dst, streams):
            with torch.cuda.stream(s):
                t.copy_(src, non_blocking=True)
    for d in devices:
        torch.cuda.synchronize(d)
    t0 = time.perf_counter()
    for _ in range(reps):
        for t, s in zip(dst, streams):
            with torch.cuda.stream(s):
                t.copy_(src, non_blocking=True)
    for d in devices:
        torch.cuda.synchronize(d)
    return n * reps * len(devices) / (time.perf_counter() - t0) / 1e9


def _read_ftw_layer(snap: Path, layer: int) -> tuple[dict, int]:
    """One layer's expert banks from an FTW checkpoint, as host tensors; and bytes per expert."""
    import torch

    index = json.load(open(snap / "freetoken_weight.json"))
    shards = index["shards"]
    banks, per_expert = {}, 0
    for t in index["tensors"]:
        if t["kind"] != "experts_bank" or not t["name"].endswith(f"#L{layer:05d}"):
            continue
        name = t["name"].split("#", 1)[0]
        shard = next(s for s in shards if s["global_off"] <= t["global_off"] < s["global_off"] + s["nbytes"])
        buf = torch.empty(t["nbytes"], dtype=torch.uint8)
        view, done = memoryview(buf.numpy()), 0
        with open(snap / shard["file"], "rb", buffering=0) as f:
            f.seek(t["global_off"] - shard["global_off"])
            while done < len(view):
                n = f.readinto(view[done:])
                if not n:
                    raise OSError(f"{shard['file']}: short read in {t['name']}")
                done += n
        dtype = getattr(torch, t["dtype"])
        banks[name] = buf.view(dtype).view(t["shape"])
        per_expert += t["nbytes"] // t["shape"][0]
    return banks, per_expert


def _cpu_bench(snap: Path, gpus: list[int], thread_counts: list[int]) -> dict:
    """The CPU decode kernel (fp32 activations, fp32 output) on layer 0's real experts: time per
    task for 1, 2 and 8 routed experts at each thread count, all 512 experts in rotation so the
    reads come from RAM, not the L3."""
    import types

    import torch
    from freetoken.moe.cpu_executor import CpuMoeExecutor, physical_core_cpus

    banks, per_expert = _read_ftw_layer(snap, 0)
    E = next(iter(banks.values())).shape[0]
    cache = types.SimpleNamespace(quant_format="nvfp4", num_layers=1, num_experts=E,
                                  bank_sources={k: [v] for k, v in banks.items()})
    cores = physical_core_cpus()
    top_k = 8
    ex = CpuMoeExecutor(cache, top_k=top_k, activation="silu", apply_router_weight_on_input=False,
                        num_threads=len(cores), max_tokens=1, device=torch.device("cpu"), cpus=cores,
                        out_fp32=True, w4a8=False)
    io = ex.stage(0, 1)
    task = ex._task_for(0, 1)
    io["x"].copy_(torch.randn(1, ex.H).bfloat16())
    io["w"].fill_(1.0 / top_k)
    gen = torch.Generator().manual_seed(0)
    order = torch.randperm(E, generator=gen)
    result = {"isa": ex.isa, "bytes_per_expert": per_expert, "threads": {}}
    pos = 0
    for n in thread_counts:
        if n > ex.max_threads:
            continue
        ex.set_active_threads(n)
        row = {}
        for k in (1, 2, 8):
            ids = torch.full((1, top_k), -1, dtype=torch.int32)
            times = []
            for rep in range(max(48, 384 // k)):
                ids[0, :k] = order[(pos + torch.arange(k)) % E]
                pos += k
                io["ids"].copy_(ids)
                t0 = time.perf_counter()
                ex._ext.run_task(task)
                times.append(time.perf_counter() - t0)
            times.sort()
            row[k] = times[len(times) // 2] * 1e6  # median us per task
        us = row[8] / 8
        result["threads"][n] = {"us_per_task": row, "us_per_expert": us, "gbps": per_expert / us / 1e3}
    del ex
    return result


def probe(args) -> None:
    snap = Path(args.snap)
    gpus = [int(g) for g in args.gpus.split(",")]
    meta = json.load(open(snap / "freetoken_weight.json"))
    hw = {"date": datetime.datetime.now().isoformat(timespec="minutes"), "host": platform.node(),
          "gpus_arg": args.gpus, "snap": str(snap), "quant_format": meta.get("quant_format", ""),
          "cpu": _cpu_facts(), "gpus": _gpu_facts(gpus)}
    from freetoken.moe.cpu_executor import physical_core_cpus

    cores = physical_core_cpus()
    hw["cpu"]["physical_cores"] = len(cores)
    per = max(1, len(cores) // len(gpus))
    hw["default_threads"] = per - 1 if len(gpus) == 1 and per > 2 else per
    hw["max_threads"] = per
    import torch

    for i, g in enumerate(hw["gpus"]):
        g["h2d_gbps"] = round(_h2d_gbps([gpus[i]]), 2)
    if len(gpus) > 1:
        hw["h2d_all_gbps"] = round(_h2d_gbps(gpus), 2)
    hw["gpus"] = [{**g, **{k: v for k, v in _gpu_facts([g["index"]])[0].items() if k.startswith("pcie_")}}
                  for g in hw["gpus"]]
    torch.cuda.empty_cache()
    if hw["quant_format"] == "nvfp4":
        counts = sorted({1, 2, max(1, per // 2), hw["default_threads"], per})
        hw["cpu_kernel"] = _cpu_bench(snap, gpus, counts)
        # a TP rank holds half of every expert's rows and streams half the bytes
        shard = hw["cpu_kernel"]["bytes_per_expert"] / len(gpus)
        hw["pcie_us_per_expert"] = round(shard / (min(g["h2d_gbps"] for g in hw["gpus"]) * 1e3), 1)
    json.dump(hw, open(args.out, "w"), indent=1)
    print(render_hw(hw))


def render_hw(hw: dict) -> str:
    c = hw["cpu"]
    isa = ", ".join(n for n, k in (("AVX2", "avx2"), ("AVX-512", "avx512f"), ("AVX-512 BF16", "avx512_bf16"),
                                   ("AVX-512 VNNI", "avx512_vnni"), ("AVX-VNNI", "avx_vnni")) if c[k]) or "none"
    lines = [f"CPU: {c['model']}, {c['physical_cores']} cores / {c['logical_cpus']} threads, {c['ram_gib']} GiB RAM",
             f"     ISA: {isa}"]
    for g in hw["gpus"]:
        # an idle link drops to gen1, so the generation shown is the one the system allows
        lines.append(f"GPU {g['index']}: {g['name']}, {g['memory_mib'] / 1024:.0f} GiB, PCIe gen{g['pcie_gen_max']} "
                     f"x{g['pcie_width']}, pinned H2D {g['h2d_gbps']:.1f} GB/s")
    if "h2d_all_gbps" in hw:
        lines.append(f"       all GPUs copying at once: {hw['h2d_all_gbps']:.1f} GB/s total")
    k = hw.get("cpu_kernel")
    if k:
        mb = k["bytes_per_expert"] / 1e6
        lines.append(f"CPU expert kernel ({k['isa']}, layer 0 of the checkpoint, {mb:.2f} MB per expert):")
        lines.append("  threads   us/expert   GB/s   us per layer with 1 / 2 / 8 experts")
        for n, r in k["threads"].items():
            t = r["us_per_task"]
            t1, t2, t8 = (t.get(x, t.get(str(x))) for x in (1, 2, 8))
            lines.append(f"  {int(n):>7}   {r['us_per_expert']:9.1f}   {r['gbps']:4.1f}   "
                         f"{t1:.0f} / {t2:.0f} / {t8:.0f}")
        lines.append(f"PCIe fetch of one expert{' shard' if len(hw['gpus']) > 1 else ''}: "
                     f"~{hw['pcie_us_per_expert']:.0f} us at the measured H2D rate")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# run


class Server:
    def __init__(self, port: int):
        self.base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(self.base + "/v1/models", timeout=30) as r:
            self.model = json.load(r)["data"][0]["id"]

    def _post(self, path: str, body: dict, timeout: float = 600):
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=timeout)

    def tune(self, fetch: float | None = None, threads: int | None = None) -> dict:
        try:
            with self._post("/v1/admin/decode-tune", {"decode_fetch": fetch, "threads": threads}) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise SystemExit(f"decode-tune refused ({e.code}): {e.read().decode(errors='replace')}")

    def max_context(self) -> int:
        with urllib.request.urlopen(self.base + "/v1/models", timeout=30) as r:
            return int(json.load(r)["data"][0].get("max_model_len") or 0)

    def decode(self, prompt: str, max_tokens: int, skip: int) -> dict:
        """Greedy streaming decode; tok/s over the tokens after the first ``skip``."""
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0, "stream": True, "ignore_eos": True,
                "stream_options": {"include_usage": True}}
        stamps, usage, t_start = [], {}, time.perf_counter()
        with self._post("/v1/chat/completions", body, timeout=3600) as resp:
            for line in resp:
                line = line.strip()
                if not line.startswith(b"data:") or line == b"data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices", []):
                    d = ch.get("delta", {})
                    if d.get("content") or d.get("reasoning_content"):
                        stamps.append(time.perf_counter())
        n = usage.get("completion_tokens", len(stamps))
        if len(stamps) <= skip + 1:
            raise SystemExit(f"decode streamed only {len(stamps)} chunks")
        # one chunk per token normally; scale if the server batched some
        per_chunk = n / len(stamps)
        tps = (len(stamps) - 1 - skip) * per_chunk / (stamps[-1] - stamps[skip])
        return {"tps": tps, "tokens": n, "prompt_tokens": usage.get("prompt_tokens", 0),
                "ttft": stamps[0] - t_start}


def build_context(repo: Path, tokenizer, target: int) -> str:
    """Deterministic filler of about ``target`` tokens: this checkout's docs, then its source."""
    files = sorted(glob.glob(str(repo / "docs" / "**" / "*.md"), recursive=True))
    files += sorted(glob.glob(str(repo / "python" / "freetoken" / "**" / "*.py"), recursive=True))
    parts, total = [], 0
    for f in files:
        text = Path(f).read_text(errors="replace")
        n = len(tokenizer.encode(text).ids)
        if total + n > target:
            ids = tokenizer.encode(text).ids[: max(0, target - total)]
            parts.append(f"=== {Path(f).relative_to(repo)} ===\n" + tokenizer.decode(ids))
            break
        parts.append(f"=== {Path(f).relative_to(repo)} ===\n{text}")
        total += n
        if total >= target:
            break
    return "\n\n".join(parts)


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def choose_fetch(table: dict[float, dict[str, float]], depths: list[str]) -> tuple[float, str]:
    """``table[P][depth]`` = decode tok/s. Highest P whose mean speed-up over P=1 is within TIE of
    the best and that loses nowhere; P=1 unless that beats P=1 by MARGIN at every depth."""
    base = table[1.0]
    speedup = {p: [table[p][d] / base[d] for d in depths] for p in table}
    score = {p: geomean(s) for p, s in speedup.items() if min(s) >= 1 - TIE}
    best = max(score.values())
    pick = max(p for p in score if score[p] >= best * (1 - TIE))
    if pick < 1.0 and min(speedup[pick]) < 1 + MARGIN:
        worst = min(speedup[pick])
        return 1.0, (f"the best share ({pick:g}) is only {(worst - 1) * 100:+.1f}% against DECODE_FETCH=1 "
                     f"at its weakest depth (needs +{MARGIN:.0%}): CPU help does not pay on this machine")
    if pick == 1.0:
        return 1.0, "no share beats DECODE_FETCH=1: CPU help does not pay on this machine"
    best_p = max(score, key=score.get)
    why = (f"{pick:g} is {(score[pick] - 1) * 100:+.0f}% over DECODE_FETCH=1 (geometric mean over depths)")
    if best_p != pick:
        why += (f"; {best_p:g} measured {(score[best_p] / score[pick] - 1) * 100:+.1f}% more, within "
                f"the {TIE:.0%} noise, and the higher share keeps the CPU less busy")
    return pick, why


def choose_threads(table: dict[int, dict[str, float]], depths: list[str]) -> int:
    """Fewest threads within TIE of the fastest (mean over depths)."""
    score = {n: geomean([table[n][d] for d in depths]) for n in table}
    best = max(score.values())
    return min(n for n in table if score[n] >= best * (1 - TIE))


class Serve:
    """serve-qwen38.sh in the background, on the calibration port; stopped with the script's TERM
    trap, which also clears its systemd scope."""

    def __init__(self, args, tag: str, env: dict[str, str]):
        self.log = Path(args.work) / f"serve-{tag}.log"
        full = {**os.environ, "SNAP": args.snap, "GPUS": args.gpus, "PORT": str(args.port),
                "HOST": "127.0.0.1", "CALIBRATION": "/dev/null", **env}
        print(f"loading the model ({tag}, log: {self.log})", flush=True)
        self.avail0 = _mem_available_gib()
        t0 = time.perf_counter()
        with open(self.log, "w") as log:
            self.proc = subprocess.Popen(["bash", str(Path(__file__).resolve().parent / "serve-qwen38.sh")],
                                         env=full, stdout=log, stderr=subprocess.STDOUT)
        deadline = t0 + args.load_timeout
        while True:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2) as r:
                    if json.load(r).get("status") == "ok":
                        break
            except (OSError, ValueError):
                pass
            if self.proc.poll() is not None or time.perf_counter() > deadline:
                self.stop()
                tail = "".join(self.log.read_text(errors="replace").splitlines(True)[-20:])
                raise SystemExit(f"the server did not come up; last lines of {self.log}:\n{tail}")
            time.sleep(3)
        print(f"loaded in {time.perf_counter() - t0:.0f} s", flush=True)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        # the pinned expert banks take a while to go back to the kernel after the exit, and the
        # next load refuses to start until they have
        deadline = time.perf_counter() + 180
        while _mem_available_gib() < self.avail0 - 4 and time.perf_counter() < deadline:
            time.sleep(2)


def _mem_available_gib() -> float:
    return next(int(ln.split()[1]) for ln in open("/proc/meminfo") if ln.startswith("MemAvailable:")) / 2**20


def run(args) -> None:
    """Load 1: DECODE_FETCH=1 (plain decode, the reference). Load 2: CPU-assisted decode,
    runtime-tunable, for the share and thread sweep. The decode path is fixed per load."""
    from tokenizers import Tokenizer

    t_begin = time.perf_counter()
    hw = json.load(open(args.hw))
    tok = Tokenizer.from_file(str(Path(hw["snap"]) / "tokenizer.json"))
    repo = Path(__file__).resolve().parent.parent
    out_tokens = args.skip + args.tokens
    log: list[str] = []
    prompts: dict[str, list[str]] = {}
    ctx_tokens: dict[str, int] = {}

    def note(msg):
        print(msg, flush=True)
        log.append(msg)

    def measure(srv, name, tag, fetch, threads):
        if fetch is not None:
            srv.tune(fetch, threads)
        rs = [srv.decode(p, out_tokens, args.skip) for p in prompts[name]]
        tps = sum(r["tps"] for r in rs) / len(rs)
        label = "1 (off)" if fetch is None else f"{fetch:g}"
        note(f"  {tag:<8} {name:<5} fetch {label:<7} threads {threads or '-':<2}  {tps:6.2f} tok/s  "
             f"({', '.join(f'{r['tps']:.1f}' for r in rs)})")
        return tps

    def enter(srv, name):
        # prefill once, untimed; later requests hit the cached prefix. A request for the other
        # context evicts it, so each phase keeps one context's runs together.
        t0 = time.perf_counter()
        r = srv.decode(prompts[name][0], 8 + args.skip, 0)
        ctx_tokens[name] = r["prompt_tokens"]
        note(f"{name} context: {r['prompt_tokens']} tokens (prefill + warm-up {time.perf_counter() - t0:.0f} s)")

    # load 1: the reference
    serve = Serve(args, "plain", {"DECODE_FETCH": "1"})
    try:
        srv = Server(args.port)
        ctx_max = srv.max_context()
        depths = [("short", args.short)]
        if args.long:
            long_ctx = min(args.long, int(ctx_max * 0.9) - out_tokens - 64) if ctx_max else args.long
            if long_ctx > 4 * args.short:
                depths.append(("long", long_ctx))
        names = [n for n, _ in depths]
        for name, target in depths:
            context = build_context(repo, tok, target)
            prompts[name] = [f"{context}\n\n{q}" for q in QUESTIONS]
        base_runs = {}
        for name in names:
            enter(srv, name)
            base_runs[name] = [measure(srv, name, "off", None, None) for _ in range(3)]
    finally:
        serve.stop()
    base = {n: sum(v) / len(v) for n, v in base_runs.items()}

    # load 2: CPU help, share and threads switched at runtime
    env = {"DECODE_FETCH": f"{GRID[1]:g}", "FREETOKEN_CPU_DECODE_TUNABLE": "1"}
    if args.threads:
        env["CPU_DECODE_THREADS"] = str(args.threads)
    serve = Serve(args, "cpu-assisted", env)
    try:
        srv = Server(args.port)
        state = srv.tune()
        if state.get("status") != "ok":
            raise SystemExit(f"the server is not tunable: {state}")
        max_threads, threads0 = state["max_threads"], state["threads"]
        screen = {1.0: base}
        for name in names:
            enter(srv, name)
            for p in GRID[1:]:
                screen.setdefault(p, {})[name] = measure(srv, name, "screen", p, threads0)
        fetch, why = choose_fetch(screen, names)
        note(f"screen pick: DECODE_FETCH={fetch:g} - {why}")

        # Threads at the long context only (the smallest expert cache, the most CPU work), then
        # the confirmation of the pick and the fastest share, each context once, the cached one
        # first: one more long prefill at most.
        threads, thread_table, confirm = threads0, {}, {"base": base, "shares": {}}
        if fetch < 1:
            fastest = max((p for p in screen if p < 1), key=lambda p: geomean([screen[p][n] / base[n] for n in names]))
            cands = sorted({fetch, fastest}, reverse=True)
            samples = {c: {} for c in cands}
            for name in reversed(names):
                if name != names[-1]:
                    enter(srv, name)
                if name == names[-1] and not args.threads:
                    for n in sorted({threads0, max_threads, max(1, threads0 // 2)}):
                        thread_table[n] = measure(srv, name, "threads", fetch, n)
                    threads = choose_threads({n: {name: v} for n, v in thread_table.items()}, [name])
                    note(f"threads pick: {threads} (at the {name} context)")
                for c in cands:
                    runs = [measure(srv, name, "confirm", c, threads) for _ in range(2)]
                    if threads == threads0:
                        runs.append(screen[c][name])
                    elif c == fetch and name == names[-1]:
                        runs.append(thread_table[threads])
                    samples[c][name] = sum(runs) / len(runs)
            confirm["shares"] = samples
            final, why2 = choose_fetch({1.0: base, **samples}, names)
            why = f"the confirmation runs undid it: {why2}" if final == 1.0 else f"confirmation runs: {why2}"
            fetch = final
    finally:
        serve.stop()

    res = {"hw": hw, "depths": {n: ctx_tokens[n] for n in names}, "screen": screen,
           "threads_table": thread_table, "confirm": confirm, "decode_fetch": fetch,
           "threads": threads if fetch < 1 else hw["default_threads"], "why": why,
           "screen_threads": threads0, "sweep_s": round(time.perf_counter() - t_begin), "log": log,
           "out_tokens": args.tokens, "skip": args.skip}
    json.dump(res, open(args.out, "w"), indent=1)
    report = render_report(res, args.total_s)
    print("\n" + report)
    write_outputs(res, report, Path(args.env))


def render_report(res: dict, total_s: float | None = None) -> str:
    hw = res["hw"]
    names = list(res["depths"])
    head = "   ".join(f"{n} ({res['depths'][n] / 1000:.0f}k)".rjust(13) for n in names)
    lines = [f"FreeToken CPU-assisted decode calibration, {hw['date']} on {hw['host']}, GPUS={hw['gpus_arg']}",
             "", render_hw(hw), "",
             f"Decode tok/s ({res['out_tokens']} greedy tokens after {res['skip']} warm-up, mean of "
             f"{len(QUESTIONS)} prompts), CPU threads {res['screen_threads']}; DECODE_FETCH=1 is a "
             "separate load (plain decode), mean of 3:",
             f"  DECODE_FETCH   {head}   vs 1"]
    base = res["screen"][1.0]
    for p, row in sorted(res["screen"].items(), key=lambda kv: -float(kv[0])):
        sp = geomean([row[n] / base[n] for n in names])
        lines.append(f"  {float(p):>12g}   " + "   ".join(f"{row[n]:13.1f}" for n in names)
                     + f"   {(sp - 1) * 100:+5.1f}%")
    if res["threads_table"]:
        lines.append(f"CPU threads at the screen's pick, {names[-1]} context:")
        for n, v in sorted(res["threads_table"].items(), key=lambda kv: int(kv[0])):
            lines.append(f"  {int(n):>12}   {v:13.1f}")
    if res["confirm"].get("shares"):
        c = res["confirm"]
        lines.append(f"Confirmation at {res['threads']} threads (DECODE_FETCH=1 and each share: mean of 3 runs):")
        lines.append(f"  {1:>12}   " + "   ".join(f"{c['base'][n]:13.1f}" for n in names))
        for p, row in sorted(c["shares"].items(), key=lambda kv: -float(kv[0])):
            mark = "  <- pick" if float(p) == res["decode_fetch"] else ""
            lines.append(f"  {float(p):>12g}   " + "   ".join(f"{row[n]:13.1f}" for n in names) + mark)
    lines += ["", f"Recommendation: DECODE_FETCH={res['decode_fetch']:g} CPU_DECODE_THREADS={res['threads']}",
              f"  {res['why']}"]
    t = f"two loads and the sweep {res['sweep_s'] / 60:.1f} min"
    if total_s:
        t += f", {(total_s + res['sweep_s']) / 60:.1f} min in all with the probe"
    lines.append(f"  ({t})")
    return "\n".join(lines)


def write_outputs(res: dict, report: str, env_path: Path) -> None:
    hw = res["hw"]
    env_path.parent.mkdir(parents=True, exist_ok=True)
    gpus = "; ".join(f"{g['name']} PCIe gen{g['pcie_gen_max']} x{g['pcie_width']} {g['h2d_gbps']:.1f} GB/s"
                     for g in hw["gpus"])
    k = hw.get("cpu_kernel", {})
    cpu = f"{hw['cpu']['model']}, {hw['cpu']['physical_cores']} cores"
    if k:
        t = k["threads"].get(str(res["threads"])) or k["threads"].get(res["threads"])
        if t:
            cpu += f", {t['us_per_expert']:.0f} us/expert at {res['threads']} threads ({k['isa']})"
    names = list(res["depths"])
    c = res["confirm"]
    speeds = " / ".join(f"{c['base'][n]:.1f}" for n in names)
    pick = c.get("shares", {}).get(res["decode_fetch"])
    if pick:
        speeds = f"DECODE_FETCH=1 {speeds}, picked {' / '.join(f'{pick[n]:.1f}' for n in names)}"
    depths = " / ".join(f"{res['depths'][n] // 1000}k" for n in names)
    env_path.write_text(
        f"# FreeToken CPU-assisted decode calibration (scripts/calibrate-qwen38.sh), {hw['date']}\n"
        f"# GPUS={hw['gpus_arg']}: {gpus}\n"
        f"# CPU: {cpu}\n"
        f"# decode tok/s at {depths} context: {speeds}\n"
        f"# {res['why']}\n"
        f"DECODE_FETCH={res['decode_fetch']:g}\n"
        f"CPU_DECODE_THREADS={res['threads']}\n"
    )
    env_path.with_suffix(".txt").write_text(report + "\n")
    print(f"\nwrote {env_path} (and the report next to it, {env_path.with_suffix('.txt').name})")


def write_skip(args) -> None:
    """No sweep: the feature cannot run here (INT4 checkpoint, no SIMD CPU kernel)."""
    hw = json.load(open(args.hw))
    res = {"hw": hw, "depths": {}, "screen": {}, "threads_table": {}, "confirm": {"base": {}},
           "decode_fetch": 1.0, "threads": hw.get("default_threads", 1), "why": args.why,
           "sweep_s": 0, "out_tokens": 0, "skip": 0}
    report = "\n".join([f"FreeToken CPU-assisted decode calibration, {hw['date']} on {hw['host']}, "
                        f"GPUS={hw['gpus_arg']}", "", render_hw(hw), "",
                        "Recommendation: DECODE_FETCH=1 (off)", f"  {args.why}"])
    print("\n" + report)
    write_outputs(res, report, Path(args.env))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--snap", required=True)
    p.add_argument("--gpus", required=True)
    p.add_argument("--out", required=True)
    s = sub.add_parser("run")
    s.add_argument("--hw", required=True)
    s.add_argument("--snap", required=True)
    s.add_argument("--gpus", required=True)
    s.add_argument("--port", type=int, required=True)
    s.add_argument("--work", required=True, help="scratch dir for the server logs")
    s.add_argument("--env", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--short", type=int, default=4096)
    s.add_argument("--long", type=int, default=100000)
    s.add_argument("--tokens", type=int, default=192, help="measured decode tokens per prompt")
    s.add_argument("--skip", type=int, default=48, help="warm-up tokens per prompt, not measured")
    s.add_argument("--threads", type=int, default=0, help="fix the CPU thread count (no thread sweep)")
    s.add_argument("--load-timeout", type=float, default=900)
    s.add_argument("--total-s", type=float, default=None)
    k = sub.add_parser("skip")
    k.add_argument("--hw", required=True)
    k.add_argument("--env", required=True)
    k.add_argument("--why", required=True)
    args = ap.parse_args()
    {"probe": probe, "run": run, "skip": write_skip}[args.cmd](args)


if __name__ == "__main__":
    main()

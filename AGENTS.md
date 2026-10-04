# Instructions for AI coding agents

Read [CONTRIBUTING.md](CONTRIBUTING.md) first. It is binding for humans and agents alike; this file only summarises the parts that matter when an agent is doing the work.

## AI policy

AI-assisted code is welcome. Submitting code the contributor does not understand is not. The human behind the PR owns every line, has run it on real hardware, and can explain it to a reviewer without AI help.

Agents must not:

- Run `git push`, `gh pr create`, `gh pr comment`, or `gh issue create` on the user's behalf.
- Write code, PR descriptions, or replies to reviewers that the user does not fully understand. The user must be able to explain and defend every line without AI help.
- Report tests or benchmarks as run when they were not.

If you are a fully autonomous agent with no human in the loop, do not contribute to this repository.

## Repository layout

The main subsystems:

```
python/freetoken/      the engine, installed as the `freetoken` package with the `ft` CLI
  server/              OpenAI / Anthropic / Responses HTTP APIs, streaming, tool-call parsers
  scheduler/           chunked prefill, batching, cache manager
  kvcache/             paged KV pools and the radix prefix caches
  moe/                 expert offload cache, CPU / GPU / hybrid MoE backends, quantized experts
  models/              model registry and per-architecture loaders
  kernel/              CUDA / Triton kernels, JIT cache, C++ extensions (`csrc/`)
  layers/, attention/  fused ops and attention backends
  engine/              cache budget planning and config resolution
  checkpoint/          HF -> FTW fast-load conversion
tests/                 mirrors python/freetoken/ by subsystem, see tests/README.md
benchmarks/            end-to-end and micro benchmarks, see benchmarks/README.md
docs/                  install, quickstart, CLI and model docs
freetoken-kernel-cache/ companion wheel of prebuilt kernels, see its README
scripts/               wheel build and release scripts
```

## Development

Linux x86_64 with an NVIDIA GPU. Use `uv`, not bare `pip`:

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
uv run pytest tests/ -m "not slow"
```

CUDA kernels are JIT-compiled with `nvcc` on first use unless the prebuilt `freetoken-kernel-cache` wheel is installed. The C++ extensions under `python/freetoken/kernel/csrc/` are built by `setup.py`; after changing them run `python setup.py build_ext --inplace`.

Put a new test in the `tests/` directory that mirrors the module it protects, and extend an existing file before creating a new one. Bug fixes come with a test that fails before and passes after. Performance changes come with A/B numbers against `main`.

## Issues and PRs

- Search existing issues and PRs before starting. Items on the [Roadmap](https://github.com/FlashML-org/FreeToken/issues/79) are discussed with maintainers before implementation; features not on it start as an issue.
- When helping the user draft an issue, follow the matching template in `.github/ISSUE_TEMPLATE/` (engine bug, model checkpoint, feature request) and fill in every required field: hardware, driver, FreeToken version, checkpoint ID, exact command, and the full log.
- One change per PR, linked to its issue, with the hardware, checkpoint ID and exact command it was tested with.

## Code comments

Comments explain a non-obvious "why", never restate the code. Write the code first, then add a comment only where a reader would otherwise be confused. Keep them to one or two lines. Configuration files get no comments. Use ASCII: `-` not em-dash, `->` not arrows.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/), one line, imperative, lowercase, no trailing period:

```
fix(kvcache): size the SWA radix pool for chunked prefill
```

PRs are squash-merged, so the PR title follows the same format. The subject line is usually enough; add a body only when the change needs a why that the diff does not show, and keep it to a few lines. Only commit when the user asks. If the user wants attribution, use `Assisted-by: <agent name>`, not `Co-authored-by`.

## Caveats to revisit for new models

Found reviewing `perf(prefill)` (9017dfd, 2026-10-04) on Qwen3.8-Flash-Next; re-check them before enabling these paths on a new model (e.g. Qwen4).

- **The W4A16 CPU prefill kernel hardcodes the expert MLP.** `kernel/csrc/cpu_moe/wna16_cpu.h` computes `silu(gate) * up` and applies the router weight after `down`, with no `alpha` / `limit`. It also assumes AutoGPTQ zero points stored minus one, group size 128, gate and up concatenated (not interleaved) on the output axis, `hidden % 128 == 0` and `inter % 64 == 0`. `Engine._maybe_init_wna16_cpu_prefill` checks only the bank roles, so with `FREETOKEN_WNA16_CPU_PREFILL=1` a model that differs gets wrong routed-expert outputs silently. Gate it on the layer's `activation`, `apply_router_weight_on_input` and `limit` first. `tests/moe/test_wna16_cpu_prefill.py` allows 2e-2 of the max while the kernel lands at ~1.7e-3 rel-RMS; tighten it when touching the kernel.
- **CPU prefill is not bit-reproducible.** Which misses the CPU computes depends on the VRAM expert-cache state, and the CPU and GPU partials round differently, so the same prompt can give slightly different logits between runs (two identical runs differed from the first row); GPU-only prefill is bit-identical. Against bf16 full-precision reference logits (5656 tokens, `--max-prefill-length 1024`): GPU-only 0.145 mean KLD, every routed expert on the CPU 0.125, four CPU/GPU split runs 0.135-0.138 (mean 0.137). Numerically equivalent configs still move the score a few percent (GPU-only one-shot 0.140 vs 1024-token chunks 0.145), so compare several runs rather than a single score, and A/B unrelated changes with `FREETOKEN_WNA16_CPU_PREFILL=0`.
- **Keep the hybrid merge to one rounding.** `OffloadMoELayer._prefill_cpu_hybrid` adds the GPU routes and the CPU's fp32 partial in fp32 and rounds once. It used to add two bf16-rounded partials: on real Qwen3.8 prefill activations that was further from exact W4 math than GPU-only in every sample (4.29e-3 vs 4.11e-3 rel-RMS; one rounding gives 4.06e-3), and five such runs averaged 0.156 mean KLD against the reference (0.140-0.169) where one rounding gives 0.137. The GPU half carries ~95% of the output, so rounding it separately costs nearly a full extra rounding. `test_hybrid_merge_rounds_once` pins this.
- **Decode-time GDN snapshots depend on scheduling order.** `CacheManager.snapshot_decode_states` flips `mamba_next_track_idx` without checking for a pending prefill track (`mamba_last_track_seqlen`). That is safe only because both scheduler loops commit (`cache_req`) a prefill batch before launching the next batch. If a decode step can ever run before that commit, the commit donates the decode snapshot's slot under the prefill track's shorter length, and later prefix hits restore the wrong GDN state. Boundary snapshots are off under MTP and skip multimodal requests.

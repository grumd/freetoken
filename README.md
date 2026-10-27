# FreeToken fork: Qwen3.8-Flash-Next on consumer GPUs

A fork of [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) (Apache-2.0), an edge-native MoE serving engine. It builds on [anon761/freetoken](https://github.com/anon761/freetoken), which added DeepSeek-V4.1, vision, MTP/DSpark and TP. This repo tunes the offload path for **Qwen3.8-Flash-Next** in two checkpoints:

- Intel's INT4 [W4A16 AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)
- NVIDIA's [FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)

Experts live in host RAM, and the full 262k context runs on 2x 20 GB GPUs. The expert cache in VRAM grows into whatever the live context does not use.

| 2x RTX 3080 20GB, Ryzen 7 9800X3D, 96 GB DDR5 | INT4 | FP4 |
|---|---|---|
| Decode (GPUs with P2P) | ~79 tok/s | ~89 tok/s |
| Cold TTFT, 150 / 1850 tok | 0.32 / 1.50 s | 0.37 / 1.45 s |
| Host RAM for experts | 58.6 GiB | 63.9 GiB |

## Quick start

```bash
git clone https://github.com/grumd/freetoken && cd freetoken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"

hf download Intel/Qwen3.8-Flash-Next-W4A16-AutoRound --local-dir ~/models/q38-int4-hf
ft checkpoint --model ~/models/q38-int4-hf --out ~/models/ftw/q38-int4   # fast-load FTW
SNAP=~/models/ftw/q38-int4 scripts/serve-qwen38.sh                       # :8080, chat UI at /
```

[`scripts/README.md`](scripts/README.md) covers:

- requirements: AVX-512 BF16 CPU, 96 GB RAM, NVMe, CUDA 13 + `nvcc`
- the FP4 checkpoint
- FTW conversion options
- every knob
- single-GPU settings and troubleshooting

General install notes: [`docs/install.md`](docs/install.md).

## What this repo changes

Compared with anon761/freetoken. Env vars are runtime switches; `scripts/serve-qwen38.sh` sets the defaults.

**Prefill**
- **Routed-only prefill:** chunks up to `--moe-routed-prefill-tokens` load only their routed experts instead of streaming all 24,576 (`e88b3ff`).
- **Hybrid CPU prefill:** AVX-512 BF16 kernels compute VRAM-cache misses straight from the pinned host banks while the GPU fetches the most-routed ones. Works for W4A16 and NVFP4. `FREETOKEN_WNA16_CPU_PREFILL`, `..._FETCH`, `..._THREADS` (`9017dfd`, `c19c6b5`).
- **W4A16 tile-dequant prefill kernel:** exact-dequant mode with fp32 gate/up, `FREETOKEN_WNA16_PREFILL_PRECISE` (`17c97b7`, `82026cf`).
- **NVFP4 kernels:** tile-dequant prefill and fused-epilogue decode, tuned for sm_86. `FREETOKEN_NVFP4_TILE` (`8c40fb0`).
- **Decode-time GDN state snapshots:** taken every 64 tokens, so an agent's next turn reuses the previous reply. `FREETOKEN_DECODE_SNAPSHOT_INTERVAL` (`9017dfd`).
- **Hit-D2D on by default:** longer chunks copy VRAM-cached experts device-side and stream only the misses, pp16384 -14% (`34204eb`). A probe race that randomly disabled it at startup is fixed (`9d188f9`).

**Decode**
- Fused-epilogue W4A16 expert MLP (`e824145`).
- Triton streaming GEMV for single-token bf16 linears, plus one launch for the router and shared-expert front (`d715465`, `5234afe`).
- Two-rank all-reduce through shared pinned host memory, so no P2P is needed (`45c2556`).
- With P2P between the GPUs, an LL all-reduce and all-gather that push straight into the peer's VRAM instead, +4.9% decode. `FREETOKEN_P2P_ALLREDUCE` (`3290db0`, `6b5846c`).
- Hyper-connection mix fused into one kernel and, over P2P, split across both GPUs. `FREETOKEN_HC_FUSE`, `FREETOKEN_HC_SHARD` (`9d8d75c`).
- NVFP4 decode overlaps the expert-miss copy with the routes already in VRAM, and scores the router first so the shared expert runs during the copy. `FREETOKEN_COPY_OVERLAP`, `FREETOKEN_EARLY_ROUTER` (`2878d5c`, `9c4a0f2`).
- Together: FP4 decode 72.9 -> 78.4 tok/s, greedy output token-identical.
- FP4-Local's MXFP8 dense linears are re-encoded losslessly instead of served as bf16. `FREETOKEN_DENSE_MXFP8` (`744ff21`).

**Memory**
- **Elastic VRAM:** KV pages and expert-cache slots share one pool of VRAM at runtime. The cache takes whatever the live context does not need, and gives slots back, coldest first, as the context grows. Decode +16% (FP4 77.8 -> 90.2 tok/s, INT4 67.6 -> 78.7), output token-identical; the full 262k context and several parallel requests (800k tokens of KV in total) still fit. `--elastic-kv`, on by default at TP=2 in `serve-qwen38.sh` (`148342e`).
- **Prefix RAM tier:** evicted conversations and their GDN state spill to pinned host RAM and come back exactly instead of being prefilled again (13.6k tokens in 27 ms instead of 6.3 s). For hosts with 128 GB of RAM: `CACHE_RAM_GB=16` / `--cache-ram-gb 16` (`148342e`).
- **Balanced W4A16 TP split:** 320/320 per expert instead of 256/384, also when loading from FTW. `FREETOKEN_WNA16_BALANCED_TP` (`17c97b7`, `3d9bc64`).
- **Vision tower on the CPU:** `--vision-on-cpu` (`e88b3ff`).
- **Token embedding in pinned host memory** (`65199ba`).
- **FTW loader fix:** host-resident weights are copied out of the FTW read window instead of pinning it, saving 3 GiB (`1d64a5b`).
- **Low-RAM hosts:** `FREETOKEN_FTW_READ_CONCURRENCY` lowers the load peak (`ec6cb73`).

**Checkpoints**
- **Loading NVIDIA FP4-Local:** NVFP4-CSF compressed scales, MXFP8 attention/GDN, NVFP4 vision MLP, shards under `tensors/` (`fda980f`).
- **NVFP4 expert globals in fp32:** FP4-Local's globals underflow fp16 (`81b7b8f`).
- **FTW PLE handling:** an NVFP4 PLE table is repacked into one-read rows (`9b2a293`). `--ple none` keeps the table in the source checkpoint (`3d9bc64`).

**Server and tooling**
- Single-file chat page at `/` (`3663bdf`).
- Decode profiler, routing log, decode and prefill logit dumps (`4ad6fca`).
- `scripts/serve-qwen38.sh` and `scripts/memguard.sh`.

## Inherited from anon761/freetoken

Compared with upstream `3d919e9`. Full list in [`CHANGELOG.md`](CHANGELOG.md).

- DeepSeek-V4.1-Flash (`dsv41`): mHC, CSA2 sparse attention, Lightning Indexer, Engram memory, DSpark draft.
- Qwen3.8-Flash-Next vision tower and `image_url` input. PLE table dtype via `ft checkpoint --ple`.
- MTP speculative decoding (embedded, external or INT4 head) with CUDA-graphed rounds.
- TP-sharded loading, per-GPU `--memory-ratio`, fp8 all-reduce.
- Quantization dialect layer: compressed-tensors fp8/NVFP4, AutoRound W4A16 experts, `--dense-fp8`.
- FTW: TP slicing, carried side tables (PLE/Engram/MTP).

## For agents and contributors

- Read [`AGENTS.md`](AGENTS.md) first. Its "Caveats to revisit for new models" section lists what the CPU prefill kernels hardcode.
- Tests: `uv run pytest tests/ -m "not slow"`. About 19 tests (DeepSeek-V4/Engram, rotary, GLM, QSA chunked prefill, scheduler accounting) already failed before the FP4 commits. Some tests are order-dependent, so compare whole-suite runs.
- CPU prefill extensions JIT-build with `-march=native`. After editing `kernel/csrc/`, run `python setup.py build_ext --inplace`.
- `CPU_PREFILL=1` is not bit-reproducible. A/B kernels with `CPU_PREFILL=0`, and judge accuracy per layer, not by one KLD run.
- An FTW is tied to the expert-bank layout. Re-run `ft checkpoint` after changing a quant format.

## Citation & Acknowledgment

FreeToken (upstream). Please cite the original paper:

```bibtex
@article{yang2026freetoken,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026}
}
```

FreeToken was deeply inspired by [mini-sglang](https://github.com/sgl-project/mini-sglang), and
learned the design and reused code from [SGLang](https://github.com/sgl-project/sglang),
[vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[LightLLM](https://github.com/ModelTC/lightllm) and [llama.cpp](https://github.com/ggml-org/llama.cpp).

Licensed under Apache-2.0 ([`LICENSE`](LICENSE)).

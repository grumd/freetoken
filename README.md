# FreeToken fork: Qwen3.8-Flash-Next on consumer GPUs

A fork of [FlashML-org/FreeToken](https://github.com/FlashML-org/FreeToken) (Apache-2.0), by way of [anon761/freetoken](https://github.com/anon761/freetoken), tuned to serve **Qwen3.8-Flash-Next** with its full 262k context on two 20 GB GPUs. The experts live in host RAM; whatever VRAM the live context does not need becomes an expert cache.

| 2x RTX 3080 20GB, Ryzen 7 9800X3D, 96 GB DDR5 | [INT4](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) | [FP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) |
|---|---|---|
| Decode | ~79 tok/s | ~89 tok/s |
| Cold TTFT, 150 / 1850 tok | 0.32 / 1.50 s | 0.37 / 1.45 s |
| Host RAM for experts | 58.6 GiB | 63.9 GiB |
| Download | 169 GB | 96 GB |

Decode was measured with P2P between the GPUs (patched driver); without it expect roughly 8% less.

## Quick start

Linux, CUDA 13 with `nvcc`, 96 GB of RAM and an NVMe SSD; an AVX-512 BF16 CPU speeds up prefill ([details](scripts/README.md#requirements)).

```bash
git clone https://github.com/grumd/freetoken && cd freetoken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"

hf download Intel/Qwen3.8-Flash-Next-W4A16-AutoRound --local-dir ~/models/q38-int4-hf
ft checkpoint --model ~/models/q38-int4-hf --out ~/models/ftw/q38-int4   # one-time fast-load conversion
SNAP=~/models/ftw/q38-int4 scripts/serve-qwen38.sh                       # API on :8080, chat page at /
```

For FP4, download `nvidia/Qwen3.8-Flash-Next-FP4-Local` instead; the same script serves both. The API is OpenAI- and Anthropic-compatible at `http://127.0.0.1:8080/v1`.

The defaults are the fastest settings measured. Set something else only if:

- you have one GPU: `GPUS=0` (the context drops to 128k)
- you have 128 GB of RAM: `CACHE_RAM_GB=16` keeps older conversations in RAM, so going back to one skips its prefill
- you need another address: `HOST=0.0.0.0 PORT=8000`

The remaining knobs are in [`scripts/README.md`](scripts/README.md).

## What this fork adds

- Experts in pinned host RAM behind a VRAM expert cache that grows and shrinks with the live context
- Hybrid prefill: short chunks load only their routed experts, and the CPU computes cache misses while the GPU fetches the rest
- W4A16 and NVFP4 kernels tuned for RTX 30-series, and a two-GPU all-reduce over P2P or shared host memory
- NVIDIA's FP4-Local checkpoint, the vision tower on the CPU, and a chat page at `/`
- An optional host-RAM tier for old conversations' prefixes

Each change with its commit and measured effect: [`docs/fork-changes.md`](docs/fork-changes.md).

## For contributors

- Read [`AGENTS.md`](AGENTS.md) first; its "Caveats to revisit for new models" lists what the CPU prefill kernels hardcode.
- Tests: `uv run pytest tests/ -m "not slow"`. About 19 tests (DeepSeek-V4/Engram, rotary, GLM, QSA chunked prefill, scheduler accounting) already failed before the FP4 commits, and some are order-dependent, so compare whole-suite runs.
- A/B with `CPU_PREFILL=0`: CPU prefill is not bit-reproducible.
- After editing `kernel/csrc/`, run `python setup.py build_ext --inplace`. Re-run `ft checkpoint` after changing a quant format.

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

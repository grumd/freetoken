# FreeToken fork for Qwen3.8-Flash-Next on 1-2 GPUs

Optimized for **Qwen3.8-Flash-Next** if you have 96GB+ RAM and 1-2 GPUs.

Recommended to use with Nvidia's [QAD FP4 quant](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local). But it also supports Intel's [INT4 W4A16](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)

MoE experts are pinned to host RAM. There's no setting for context length. As a session's used context grows, it dynamically allocates more KV cache. All the remaining free VRAM is used for MoE expert cache to speed up decode. 

## Benchmarks

FP4 with default settings: RTX 3080s on PCIe 4.0 x8, 96 GB of DDR5. For 16GB tests I just limited my 3080 20GB to max 16GB used. Both 3080s have unlocked ReBAR and P2P.

### **One GPU at PCIe 4.0 x8**

| GPU | Prefill batch | Decode | 16k prefill | Max context (F16) |
|---|---|---|---|---|
| 16 GB | **8192** | 27 tok/s | **1,520 tok/s** | 147k |
| 16 GB | 4096 | 27 tok/s | 790 tok/s | 186k |
| 16 GB | 2048 | 27 tok/s | 880 tok/s | **205k** |
| 20 GB | **8192** | 34 tok/s | **1,580 tok/s** | **262k** |
| 20 GB | 4096 | 34 tok/s | 830 tok/s | 262k |

### **Two GPUs at PCIe 4.0 x8**

| GPU | Prefill batch | Decode | 16k prefill | Max context (F16) |
|---|---|---|---|---|
| 2x 16 GB | 8192 | 82 tok/s | 2,760 tok/s | 262k |
| 2x 20 GB | 8192 | 93 tok/s | 3,030 tok/s | 262k |

- **RAM:** the experts stay pinned in host RAM: 64 GiB for FP4, 59 GiB for INT4. You will need 96 GB or more RAM to use this.
- **INT4** decodes about 11% slower and downloads 169 GB instead of 96 GB.
- **Two GPUs** were measured with P2P between them (patched driver); without P2P, expect about 8% less decode.
- **PCIe 4.0 x8:** all tests done on PCIe 4.0 x8, even single-GPU tests. If you have 4.0 x16 or even PCIe 5.0 x16, you should have much better decode and prefill.

## Quick start

Linux, CUDA 13 with `nvcc`, 96 GB of RAM; an AVX-512 BF16 CPU speeds up prefill ([details](scripts/README.md#requirements)).

```bash
git clone https://github.com/grumd/freetoken && cd freetoken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"

hf download nvidia/Qwen3.8-Flash-Next-FP4-Local --local-dir ~/models/q38-fp4-hf
ft checkpoint --model ~/models/q38-fp4-hf --out ~/models/ftw/q38-fp4   # one-time fast-load conversion
SNAP=~/models/ftw/q38-fp4 scripts/serve-qwen38.sh                      # API on :8080, chat page at /
```

For INT4, download `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` instead.

The default settings should already be optimized, and the context size and expert cache size automatically match your free VRAM.

- you have one GPU: `GPUS=0`
- you need more than 147k tokens on one 16 GB GPU: `PREFILL_CHUNK=4096` or 2048 (see benchmarks above)
- you have 128 GB of RAM: `CACHE_RAM_GB=16` keeps older chat sessions in RAM, so going back to an old chat can skip prefilling from scratch

See [`scripts/README.md`](scripts/README.md) for more details about options.

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

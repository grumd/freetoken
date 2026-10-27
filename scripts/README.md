# scripts

## `serve-qwen38.sh`: Qwen3.8-Flash-Next on consumer GPUs

Serves Qwen3.8-Flash-Next from an FTW checkpoint: experts pinned in host RAM behind a VRAM expert cache that takes whatever the live context does not need, the PLE n-gram table read from disk, the vision tower on the CPU, and hybrid CPU/GPU prefill. Up to 262144 tokens of context, one request at a time; the per-setup numbers are in the [main README](../README.md#what-to-expect).

The same script serves both checkpoints. It reads `quant_format` from the FTW index to pick the model name.

| | INT4 | FP4 |
|---|---|---|
| Source | [`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) | [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) |
| Download | 169 GB | 96 GB |
| FTW on disk | 169 GB (PLE table bf16) | 106 GB (PLE table NVFP4) |
| Pinned expert banks | 58.6 GiB | 63.9 GiB |
| Served name | `q38-int4` | `q38-fp4` |
| Decode, 2x RTX 3080 with P2P | ~79 tok/s | ~89 tok/s |
| Cold TTFT 150 / 750 / 1850 / 3000 tok | 0.32 / 1.02 / 1.50 / 2.08 s | 0.37 / 0.91 / 1.45 / 2.38 s |

Measured on 2x RTX 3080 20GB (PCIe 4.0 x8 each), a Ryzen 7 9800X3D, 96 GB DDR5 and an NVMe SSD, with the defaults below. The GPUs had peer access (P2P) through a patched driver; without it expect roughly 8% less decode, since the P2P all-reduce (+4.9%) and the sharded hyper-connection mix (+3%) need it.

### Requirements

- **OS / toolchain:** Linux x86_64, NVIDIA driver r580+ (CUDA 13), CUDA 13 toolkit with `nvcc` on PATH (kernels JIT-compile on first use), `g++` (the CPU prefill extension builds with `-march=native`).
- **GPU:** one or two with 16 GB or more; tuned on 2x 20 GB. One 16 GB GPU holds 149k tokens of context, 20 GB the full 262k. VRAM use:
  - dense weights: 8.2 GiB unsplit, 5.5 GiB per GPU at TP=2
  - KV: 25 KiB/token (6.4 GiB at 262k)
  - expert cache: whatever is left.
- **RAM:** 96 GB class. The script refuses to start under 75 GiB `MemAvailable`. Pinned memory cannot swap. With 128 GB, see `CACHE_RAM_GB` under Settings.
- **CPU:** AVX-512 BF16 (Zen 4/5, Sapphire Rapids or newer) for `CPU_PREFILL=1`. Without it the server logs `FREETOKEN_WNA16_CPU_PREFILL ignored` and prefills on the GPU only.
- **Disk:** NVMe. PLE rows are read on every token.
- **Optional:** a systemd user session for the RAM-capped scope (`CGROUP=0` skips it).

### Setup

```bash
# 1. install (repo root)
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"

# 2. download
hf download Intel/Qwen3.8-Flash-Next-W4A16-AutoRound --local-dir ~/models/q38-int4-hf
hf download nvidia/Qwen3.8-Flash-Next-FP4-Local      --local-dir ~/models/q38-fp4-hf

# 3. convert to FTW (needs one GPU for the repack, pick it with --gpu)
ft checkpoint --model ~/models/q38-int4-hf --out ~/models/ftw/q38-int4
ft checkpoint --model ~/models/q38-fp4-hf  --out ~/models/ftw/q38-fp4

# 4. serve
SNAP=~/models/ftw/q38-int4 scripts/serve-qwen38.sh
```

The API is OpenAI/Anthropic-compatible at `http://127.0.0.1:8080/v1`, with a chat page at `/`. The script prints `model loaded` once `/health` reports ok. An FTW loads in about 1 min. Loading from the HF snapshot builds the expert banks serially and takes about 8 min.

### FTW conversion notes

- **Default `--ple bf16`:** the FTW is self-contained, so the HF download can be deleted afterwards.
- **`--ple none` (INT4 only):** skips copying the 95 GB PLE table. The server then reads it from the source snapshot, which must stay in place.
- **`--ple fp8` (INT4):** halves the PLE table. Lossy.
- **FP4:** its NVFP4 PLE table is always repacked into `model-plenvfp4-*.safetensors` rows, so `--ple none` does not work for it.
- **Re-convert** after pulling changes to the expert-bank layout (quant format or kernel commits). An old FTW will not load.

### Settings

The defaults are the fastest settings measured on this hardware; usually only `SNAP` is needed.

| Var | Default | When to change it |
|---|---|---|
| `SNAP` | required | FTW checkpoint dir |
| `GPUS` | `0,1` | one GPU: `GPUS=0`; context and expert cache size themselves to its VRAM |
| `CACHE_RAM_GB` | `0` | 128 GB of RAM: `16` keeps about 520k tokens of older conversations in RAM; going back to one reloads it (a 13.6k-token one in 27 ms instead of 6.3 s). Leave it off on 96 GB |
| `HOST`, `PORT` | `127.0.0.1`, `8080` | listen address |
| `EXTRA` | | extra `ft serve` flags; arguments after the script name are passed too |

<details><summary>All knobs (tuning and A/B runs)</summary>

| Var | Default | Meaning |
|---|---|---|
| `SNAP` | required | FTW checkpoint dir |
| `GPUS` | `0,1` | nvidia-smi indices; the count sets the TP size |
| `MAXLEN` | `262144` | longest request; lowered to what fits in VRAM, and reserved as KV when `ELASTIC=0` (`131072` on one GPU) |
| `ELASTIC` | `1` | KV and the expert cache share VRAM at runtime (`--elastic-kv`); replaces `MEM_RATIO`. Decode +16%, output identical |
| `KEEP_FREE_MB` | `128` | elastic: VRAM left free per GPU, also at the prefill peak |
| `PREFILL_RESERVE_GB` | auto | elastic: VRAM per GPU kept for prefill activations (auto: 2 GiB per 8192-token chunk) |
| `CACHE_RAM_GB` | `0` | whole GiB of pinned host RAM (all GPUs) for evicted prompt prefixes, reloaded instead of prefilled again (`--cache-ram-gb`); raises `MIN_AVAIL_GIB` and `MEM_MAX` by as much |
| `MEM_RATIO` | `0.86` (`0.84` on one GPU) | `ELASTIC=0`: VRAM share for weights + expert cache + KV; the rest is activation headroom |
| `CPU_PREFILL` | `1` | compute about 65% of prefill cache misses on the CPU; `0` = GPU only, bit-reproducible |
| `CPU_PREFILL_FETCH` | `0.35` | share of misses fetched over PCIe instead |
| `ROUTED_PF` | `3072` (`1280` without CPU prefill) | prefill chunks up to this many tokens load only their routed experts |
| `PRECISE_PREFILL` | `1` | W4A16 exact-dequant prefill (no speed cost) |
| `HIT_D2D` | `1` | prefill chunks above `ROUTED_PF` copy VRAM-cached experts device-side and stream only the misses over PCIe |
| `BALANCED` | `1` | W4A16 at TP=2: even 320/320 expert split |
| `DENSE_FP8` | `0` | fp8 dense linears: faster decode, lossy |
| `MTP` | `off` | `--mtp` mode (`off`, `on`, `auto`, `file`) |
| `VISION_THREADS` | `8` | CPU threads for the ViT |
| `HOST`, `PORT`, `NAME` | `127.0.0.1`, `8080`, from FTW | listen address, served model name |
| `MIN_AVAIL_GIB` | `75` + `CACHE_RAM_GB` | start-up RAM check |
| `CGROUP`, `MEM_MAX` | `1`, `78G` + `CACHE_RAM_GB` | systemd scope with a hard RAM cap and no swap |
| `MEMGUARD`, `FLOOR` | `1`, `8` | `memguard.sh` kills the server under `FLOOR` GiB free (log: `/tmp/ft-memguard.log`) |
| `EXTRA` | | extra `ft serve` flags; arguments after the script name are passed too |
| `FT` | `.venv/bin/ft` | `ft` binary to run |

</details>

**Other hardware:**

- **Bigger cards:** nothing to set; elastic VRAM gives the extra memory to the context and the expert cache.
- **Faster PCIe or RAM:** `CPU_PREFILL_FETCH` and `ROUTED_PF` were tuned for PCIe 4.0 x8 per GPU and ~72 GB/s of DDR5. On a faster link, fetch more over PCIe and lower `ROUTED_PF`.

### Troubleshooting

- **`REFUSING TO START`:** free RAM first. Lowering `MIN_AVAIL_GIB` can freeze the host.
- **CUDA OOM on the first decode after a long prompt:** with `ELASTIC=1` raise `PREFILL_RESERVE_GB` (or `KEEP_FREE_MB`); with `ELASTIC=0` lower `MEM_RATIO` by 0.01-0.02.
- **Load peak too high:** `FREETOKEN_FTW_READ_CONCURRENCY=1` (about -5 GiB).
- **Runs differ with `CPU_PREFILL=1`:** expected. Which misses the CPU computes depends on the cache state. Compare builds with `CPU_PREFILL=0`.
- **Per-prefill CPU/GPU split timing:** `FREETOKEN_WNA16_CPU_PREFILL_STATS=1`.

## Other scripts

| Script | Purpose |
|---|---|
| `memguard.sh` | RAM watchdog used by `serve-qwen38.sh` (`PAT`, `FLOOR`, `LOG`) |
| `run-tests.sh` | fresh-venv install + unit tests + serve/chat smoke test (comments in German) |
| `mtp_bench.py` | MTP speculative-decoding throughput and acceptance against a running server |
| `dspark_greedy_identity.py`, `dspark_verify_diff.py` | DSpark (DeepSeek-V4.1) greedy-identity and verify-vs-decode harnesses |
| `build-release-wheels.sh`, `publish-wheels.sh`, `ci/` | runtime + kernel-cache wheel builds and releases |

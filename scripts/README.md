# scripts

| Script | Purpose |
|---|---|
| `serve.sh` | serve Qwen3.8-Flash-Next (INT4 or FP4) on one or two GPUs |
| `calibrate.sh`, `calibrate.py` | measure the best CPU-assisted decode flags for this machine (FP4) |
| `build-release-wheels.sh`, `publish-wheels.sh`, `ci/` | upstream wheel builds and releases, used by `.github/workflows` |

## `serve.sh`

Serves Qwen3.8-Flash-Next from an FTW checkpoint: experts pinned in host RAM behind a VRAM expert cache that takes whatever the live context does not need, the PLE n-gram table read from disk, the vision tower on the CPU, and hybrid CPU/GPU prefill. Up to 262144 tokens of context, one request at a time; the per-setup numbers are in the [main README](../README.md#what-to-expect).

```bash
MODEL=~/models/ftw/q38-fp4 scripts/serve.sh                          # every GPU, up to two
MODEL=~/models/ftw/q38-fp4 GPUS=1 scripts/serve.sh                   # the second GPU only
MODEL=~/models/ftw/q38-fp4 scripts/serve.sh --max-prefill-length 4096   # any ft serve flag
```

The API is OpenAI/Anthropic-compatible at `http://127.0.0.1:8080/v1`, with a chat page at `/`. The script prints `model loaded` once `/health` reports ok. An FTW loads in about 1 min.

What the script does, top to bottom:

1. **Settings** from the environment (table below).
2. **The `ft serve` command line:** the tuned defaults. Flags you pass after the script name come later and override them.
3. **RAM safety:** pinned memory cannot swap, so running out of RAM freezes the host instead of reaching the OOM killer. The script refuses to start under `REQUIRED_FREE_RAM_GB`, runs the server in a systemd user scope capped at `RAM_LIMIT_GB` with no swap (plain `setsid` without a systemd user session), and kills it if free RAM drops below `MIN_FREE_RAM_GB`.
4. **Shutdown:** on TERM/INT it stops `ft`, then kills whatever is left of the server (a TP worker that outlives `ft` keeps a GPU and its pinned experts), within llama-manager's 10 s deadline.

| | INT4 | FP4 |
|---|---|---|
| Source | [`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) | [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) |
| Download | 169 GB | 96 GB |
| FTW on disk | 169 GB (PLE table bf16) | 106 GB (PLE table NVFP4) |
| Pinned expert banks | 58.6 GiB | 63.9 GiB |
| Decode, 2x RTX 3080 with P2P | ~79 tok/s | ~89 tok/s |
| Cold TTFT 150 / 750 / 1850 / 3000 tok | 0.32 / 1.02 / 1.50 / 2.08 s | 0.37 / 0.91 / 1.45 / 2.38 s |

Measured on 2x RTX 3080 20GB (PCIe 4.0 x8 each), a Ryzen 7 9800X3D, 96 GB DDR5 and an NVMe SSD, with the defaults. The GPUs had peer access (P2P) through a patched driver; without it expect roughly 8% less decode.

### Settings

| Var | Default | Meaning |
|---|---|---|
| `MODEL` | required | FTW checkpoint dir |
| `MODEL_NAME` | `basename $MODEL` | served model name |
| `HOST`, `PORT` | `127.0.0.1`, `8080` | listen address |
| `GPUS` | every GPU `nvidia-smi` lists, at most two | nvidia-smi indices; the count sets the TP size. Context and expert cache size themselves to the VRAM |
| `PREFIX_CACHE_RAM_GB` | `0` | whole GiB of pinned host RAM for evicted prompt prefixes (`--cache-ram-gb`). With 128 GB of RAM, `16` keeps about 520k tokens of older conversations; going back to one reloads it (13.6k tokens in 27 ms instead of 6.3 s). Raises `REQUIRED_FREE_RAM_GB` and `RAM_LIMIT_GB` by as much. Leave it off on 96 GB |
| `REQUIRED_FREE_RAM_GB` | `75` + `PREFIX_CACHE_RAM_GB` | refuse to start with less free RAM (`MemAvailable`) than this. Lowering it can freeze the host |
| `RAM_LIMIT_GB` | `78` + `PREFIX_CACHE_RAM_GB` | hard RAM cap of the server (its systemd scope, no swap) |
| `MIN_FREE_RAM_GB` | `2` | kill the server when free RAM (`MemAvailable`) drops below this; guards the rest of the desktop |
| `FT_BIN` | `.venv/bin/ft`, else `ft` on PATH | `ft` binary |

### Defaults it passes to `ft serve`

| Flag | Why |
|---|---|
| `--moe-strategy offload --elastic-kv` | experts in host RAM; KV and the expert cache share VRAM at runtime, so the cache gets whatever the live context does not need (decode +16%, output identical) and the context shrinks to what fits |
| `--max-running-requests 1` | batching loses: 3 streams at once decode slower in total than 1 |
| `--max-prefill-length 8192` | tokens per prefill chunk. Elastic keeps 256 MiB + 230 KiB per chunk token free for activations; `4096` gives one 16 GB GPU 186k instead of 147k context, at about half the long-prompt prefill speed |
| `--moe-routed-prefill-tokens 3072` | chunks up to this size load only their routed experts instead of streaming every expert over PCIe (~2.4 s per chunk at x8 gen4). Re-tune on a faster link; `1280` with `FREETOKEN_WNA16_CPU_PREFILL=0` |
| `--moe-prefill-hit-d2d` | longer chunks copy VRAM-cached experts device-side and stream only the misses (pp16384 -14%, output identical) |
| `--vision-on-cpu --vision-threads 8` | the ViT would take 0.84 GiB of VRAM on every GPU; 16 threads encode a full-size image in 3.5 s, 4 in 6.8 s |
| `--mtp off` | MTP is 10-24% slower decode here in every config |
| `--enable-cache-report` | `usage.prompt_tokens_details.cached_tokens`, so clients can tell a prefix-cache hit from prefill |
| `--moe-decode-fetch 0.1` | FP4 on one GPU only: decode fetches 10% of its expert-cache misses over PCIe and the CPU computes the rest (28 -> 53 tok/s at 4k, 20 -> 46 at 261k). Two GPUs miss too rarely for it to pay. `calibrate.sh` finds the best value for your machine; `1` = plain decode |

Other useful flags: `--moe-cpu-decode-threads` (CPU threads per GPU for `--moe-decode-fetch` below 1; default all physical cores but one on one GPU), `--prefill-reserve-gb` / `--vram-keep-free-mb` (CUDA OOM after a long prompt), `--dense-fp8` (faster decode, lossy), `--moe-collect-stats` (status line with the decode miss split and CPU time).

### Engine switches

These have no `ft serve` flag. The script sets them unless they are already in the environment.

| Var | Default | Meaning |
|---|---|---|
| `FREETOKEN_WNA16_CPU_PREFILL` | `1` | routed prefill computes about 65% of its cache misses on the CPU (AVX-512 BF16; without it the server logs `FREETOKEN_WNA16_CPU_PREFILL ignored`). `0` = GPU only |
| `FREETOKEN_WNA16_CPU_PREFILL_FETCH` | `0.35` | share of those misses fetched over PCIe instead |
| `FREETOKEN_WNA16_PREFILL_PRECISE` | `1` | INT4: exact-dequant prefill, no speed cost |
| `FREETOKEN_WNA16_BALANCED_TP` | `1` | INT4 at TP=2: even 320/320 expert split |

For bit-reproducible A/B runs use `FREETOKEN_WNA16_CPU_PREFILL=0` and `--moe-decode-fetch 1`: which misses the CPU computes depends on the cache state.

### Requirements

- **OS / toolchain:** Linux x86_64, NVIDIA driver r580+ (CUDA 13), CUDA 13 toolkit with `nvcc` on PATH (kernels JIT-compile on first use), `g++` (the CPU prefill extension builds with `-march=native`).
- **GPU:** one or two with 16 GB or more. One 16 GB GPU holds 147k tokens of context, 20 GB the full 262k. Dense weights take 8.2 GiB unsplit (5.5 GiB per GPU at TP=2), KV 25 KiB/token (6.4 GiB at 262k), the expert cache the rest.
- **RAM:** 96 GB class.
- **CPU:** AVX-512 BF16 (Zen 4/5, Sapphire Rapids or newer) for CPU prefill. CPU-assisted decode follows RAM bandwidth (~50 GB/s of expert reads on dual-channel DDR5).
- **Disk:** NVMe. PLE rows are read on every token.

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
MODEL=~/models/ftw/q38-int4 scripts/serve.sh
```

FTW conversion notes:

- **Default `--ple bf16`:** the FTW is self-contained, so the HF download can be deleted afterwards.
- **`--ple none` (INT4 only):** skips copying the 95 GB PLE table. The server then reads it from the source snapshot, which must stay in place.
- **`--ple fp8` (INT4):** halves the PLE table. Lossy.
- **FP4:** its NVFP4 PLE table is always repacked into `model-plenvfp4-*.safetensors` rows, so `--ple none` does not work for it.
- **Re-convert** after pulling changes to the expert-bank layout (quant format or kernel commits). An old FTW will not load.

### Troubleshooting

- **`REFUSING TO START`:** free RAM first.
- **CUDA OOM on the first decode after a long prompt:** raise `--prefill-reserve-gb` or `--vram-keep-free-mb`.
- **Load peak too high:** `FREETOKEN_FTW_READ_CONCURRENCY=1` (about -5 GiB).
- **Per-prefill CPU/GPU split timing:** `FREETOKEN_WNA16_CPU_PREFILL_STATS=1`.

## `calibrate.sh`: CPU-assisted decode (FP4)

How much of decode's expert-cache misses the CPU should compute depends on the CPU, the RAM, the GPUs and the PCIe link. `calibrate.sh` measures this machine once, with the `MODEL` and `GPUS` you serve with, and prints the flags to pass to `serve.sh`:

```bash
MODEL=~/models/ftw/q38-fp4 GPUS=0 scripts/calibrate.sh
...
Recommendation: scripts/serve.sh --moe-decode-fetch 0.15 --moe-cpu-decode-threads 7

MODEL=~/models/ftw/q38-fp4 GPUS=0 scripts/serve.sh --moe-decode-fetch 0.15 --moe-cpu-decode-threads 7
```

It needs idle GPUs and the same free RAM as the server. On 2x RTX 3080 it takes about 12 min with one GPU and 6 min with two. Calibrate again after changing the CPU, RAM, GPUs, PCIe slots or the BIOS memory settings.

1. **Probe, without the server (~5 s):** pinned host->device bandwidth per GPU and of all at once, the CPU's ISA, and the CPU expert kernel's time per expert on one layer of real experts. With an INT4 checkpoint or a CPU without AVX2 it stops here and recommends `--moe-decode-fetch 1`.
2. **Reference load:** `serve.sh` on `CALIBRATION_PORT` with `--moe-decode-fetch 1`. Three greedy streaming decodes each at about 4k and 100k tokens of context.
3. **Sweep load:** the server restarts with CPU help and `FREETOKEN_CPU_DECODE_TUNABLE=1`, under which a loopback-only endpoint changes the share (0.5, 0.25, 0.15, 0.1, 0.05) and the thread count between requests. The decode path is fixed per load (the CUDA graphs are captured at start), which is why plain decode gets its own load.
4. **Pick:** the highest share within 2% of the fastest among those that lose nowhere, the fewest threads within 2% of the fastest, and `--moe-decode-fetch 1` unless the pick beats plain decode by 5% at every context length. The pick and the fastest share then run twice more and the choice is made again from the means of three runs.

It prints a report (hardware, the speed of every share tried, the choice and why) ending with the recommended flags. The server logs and the raw numbers stay in the scratch dir it names (`/tmp/ft-calibrate.*`).

| Var | Default | Meaning |
|---|---|---|
| `MODEL`, `GPUS`, `FT_BIN` | as for `serve.sh` | other `serve.sh` variables pass through |
| `LONG_CONTEXT_TOKENS` | `100000` | long-context length; `0` skips it (about half the time on one GPU) |
| `SHORT_CONTEXT_TOKENS` | `4096` | short-context length |
| `MEASURED_TOKENS`, `WARMUP_TOKENS` | `192`, `48` | measured decode tokens per prompt, after this many warm-up tokens |
| `FIXED_CPU_THREADS` | | fix the CPU thread count instead of sweeping it |
| `CALIBRATION_PORT` | `18080` | the calibration server's port |
| `LOAD_TIMEOUT_S` | `900` | seconds to wait for each model load |

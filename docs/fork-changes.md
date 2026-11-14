# What this fork changes

Every change against [anon761/freetoken](https://github.com/anon761/freetoken), with its commit and measured effect. Lossless changes have no switch; lossy options are `ft serve` flags, off by default. The settings that remain are listed under [Configuration](#configuration).

**Prefill**
- **Routed-only prefill:** chunks up to `--moe-routed-prefill-tokens` load only their routed experts instead of streaming all 24,576 (`e88b3ff`).
- **Hybrid CPU prefill:** AVX-512 BF16 kernels compute VRAM-cache misses straight from the pinned host banks while the GPU fetches the most-routed ones. Works for W4A16 and NVFP4. `FREETOKEN_WNA16_CPU_PREFILL`, `..._FETCH`, `..._THREADS` (`9017dfd`, `c19c6b5`).
- **W4A16 tile-dequant prefill kernel:** exact-dequant mode with fp32 gate/up, `FREETOKEN_WNA16_PREFILL_PRECISE` (`17c97b7`, `82026cf`).
- **NVFP4 kernels:** tile-dequant prefill and fused-epilogue decode, tuned for sm_86; the inline-dequant kernels stay as the fallback for other shapes (`8c40fb0`).
- **Decode-time GDN state snapshots:** taken every 64 tokens, so an agent's next turn reuses the previous reply. `FREETOKEN_DECODE_SNAPSHOT_INTERVAL` (`9017dfd`).
- **Image prompts:** a prompt with images is chunked like any other (it used to crash the server past one 8192-token chunk), and its prefix is cached, keyed on each image's pixels. The vision tower runs only for images past the cache hit. 1080p frame in a 20k-token conversation: the next turn 0.57 s with no vision-tower run instead of a full re-prefill plus 12 s of ViT per image (`ccd4a03`).
- **Hit-D2D on by default:** longer chunks copy VRAM-cached experts device-side and stream only the misses, pp16384 -14% (`34204eb`). A probe race that randomly disabled it at startup is fixed (`9d188f9`).
- **16384-token chunks on a borrowed reserve:** a chunk past `--moe-routed-prefill-tokens` streams every expert over PCIe once, so prefill was copy-bound (per 8192-token chunk and layer, one RTX 3080: 110 ms of expert copy against 53 ms of compute). `serve.sh` now prefills in 16384-token chunks. The standing activation reserve still covers 8192-token chunks (`--prefill-reserve-tokens 8192`); a longer prompt retires 1.8 GiB of expert slots while it prefills and gets them back before its first decode step, as long as the KV pages it needs leave room, so the context limit is the same as before (a fixed 16384 reserve caps one RTX 3080 at 238k). Near the limit, or with an explicit `--prefill-reserve-gb`, the chunks stay at 8192. FP4, n-gram on, time to first token: one RTX 3080 16k 12.1 -> 6.7 s cold, 10.7 -> 6.1 s warm, 64k 47.5 -> 28.1 s cold; two 16k 7.8 -> 5.3 s cold, 5.9 -> 4.9 s warm, 64k 30.4 -> 21.0 s cold. Decode unchanged. Rounding-level only: against 8192-token chunks the 16384 logits of a 48k prompt match bit for bit up to the last 8192 chunk, mean KL 0.003 and 99.1% top-1 agreement (4096 vs 8192: 0.039 and 94.9%) (`f559bfb`).
- **Aligned MXFP8 row slices:** the large-M MXFP8 path dequantizes the weight in 13107-row slices at K=2560, an odd width that sends cuBLAS to an `align1` kernel at half speed. The GDN in_proj and the QSA qkv projection now run their first slice over 13112 rows and drop the extra outputs (1024 <= M <= 16384): 15.5 -> 9.3 ms and 12.8 -> 6.2 ms per 8192 tokens. Same math on every GPU: byte-identical on sm_86 (exhaustive M sweep on an RTX 3080, and byte-equal prefill logits); on other GPUs cuBLAS may pick another algorithm for the padded shape, which can change only the summation order (rounding). Always on (`f559bfb`).
- **PLE rows read off the critical path:** a prefill chunk's PLE rows (16 random 4 KiB NVMe reads per token, ~0.3 s per 8192 tokens, ~0.5 s on two GPUs) were read before its forward launched. Now a worker thread reads them while layers 0-1 run (the PLE layer, layer 2, waits on a stream flag), and reads the next chunk of a long prompt while the current one computes. On top of the 16384-token chunks: 16k 6.7 -> 6.2 s cold, 6.1 -> 5.8 s warm and 64k 28.1 -> 24.4 s on one RTX 3080; 5.3 -> 4.9 s, 4.9 -> 4.6 s and 21.0 -> 17.5 s on two. Byte-identical prefill logits; always on where the GPU has stream memops, the same condition as the decode flag sync (`f559bfb`).

**Decode**
- Fused-epilogue W4A16 expert MLP (`e824145`).
- Triton streaming GEMV for single-token bf16 linears, plus one launch for the router and shared-expert front (`d715465`, `5234afe`).
- Two-rank all-reduce through shared pinned host memory, so no P2P is needed (`45c2556`).
- With P2P between the GPUs, an LL all-reduce and all-gather that push straight into the peer's VRAM instead, +4.9% decode. `FREETOKEN_P2P_ALLREDUCE` (`3290db0`, `6b5846c`).
- Hyper-connection mix fused into one kernel and, over P2P, split across both GPUs, bit-identical (`9d8d75c`).
- NVFP4 decode overlaps the expert-miss copy with the routes already in VRAM, and scores the router first so the shared expert runs during the copy, bit-identical (`2878d5c`, `9c4a0f2`).
- Together: FP4 decode 72.9 -> 78.4 tok/s, greedy output token-identical.
- FP4-Local's MXFP8 dense linears are re-encoded losslessly instead of served as bf16 (`744ff21`).
- **More MXFP8 dense groups (opt-in, off by default):** `--dense-mxfp8-groups lm_head,hc,shared|all` also serves the lm_head, the hyper-connection mixes and the shared experts as MXFP8. FP4-Local ships the shared experts in MXFP8, so `shared` is lossless; `lm_head` and `hc` are rounded. Small survivors are repacked after load so the freed bf16 is really released. One RTX 3080, FP4: `lm_head` 5.75 -> 5.17 GiB resident, +228 expert slots, decode 54.7 -> 57.6 tok/s, KLD within run-to-run noise (0.095 vs 0.094-0.098); `all` 4.37 GiB, 60.3 tok/s, KLD 0.102. Recommended: `lm_head` (`3d52107`, `10f046d`).
- **Prompt-lookup speculative decoding:** `--ngram-spec` drafts the continuation of an earlier repeat of the request's own context (>= 3 tokens, up to 7) and checks it in one verify window; no draft model. A cost policy drafts only where a round beats a plain step; at TP=2 rank 0's decisions ride the per-iteration message broadcast. On in `serve.sh`. FP4, a code edit that copies a 351-line file back: one RTX 3080 52.2 -> 84.0 tok/s (+61%), two 81.9 -> 126.7 (+55%); prose +1% on one GPU, -2% on two (the verify graphs' VRAM). Also fixes rollback in the shared MTP verify path (PLE conv state, QSA pending ring, verify batches on the CPU prefill path, the PLE verify buffer captured at bs=1) (`f642314`).
- **CPU-assisted decode (FP4):** decode fetches only a share of its expert-cache misses over PCIe; the CPU computes the rest straight from the pinned banks (~57 us and ~50 GB/s per expert on 7 Zen 5 cores) while the GPU runs the hits, and its fp32 partial joins the route sum before the one rounding. Every routed expert still runs, on one side or the other. One RTX 3080 at PCIe 4.0 x8: decode 28 -> 53 tok/s at 4k context, 26 -> 50 at 131k, 20 -> 46 at 261k; prefill unchanged. Per MoE layer the fp32 CPU path is as close to exact W4 math as the GPU kernel (2.19e-3 rel-RMS both); int8 activations (W4A8) were no faster and 3.6x further off, so only fp32 is kept. Output is not token-identical to the GPU path: teacher-forced KLD vs it is 0.035-0.040, the same as a GPU-only change of summation order (0.029). On two GPUs only ~4% of routes miss and it does not pay. `--moe-decode-fetch` (the fetched share; 1 = off, 0.1 in `serve.sh` on one GPU), `--moe-cpu-decode-threads` (`fca53ea`).
- **CPU-assisted decode (INT4):** the same split for W4A16 experts. The CPU kernel works on 64 output columns of the AutoGPTQ banks at a time and subtracts each zero point before the multiply: (code - zero) goes through a bf16 table exactly and `vdpbf16ps` multiplies it with pairs of bf16 activations into fp32 sums, which avoids the cancellation of folding the zero point out afterwards. 49.9 us and 51 GB/s per expert on 7 Zen 5 cores (a PCIe x8 fetch: 191 us). One RTX 3080: 44 tok/s at 4k context and 42 at 100k (`--moe-decode-fetch 0.15`), against about 25 plain. An exact fp64 variant was correctly rounded in 100% of per-layer outputs on real experts (this kernel 99.69%, the GPU kernel 99.78%) but decoded 4-5% slower with no measurable KLD difference (jyohukuchan reference, teacher-forced decode: 0.1014 vs 0.1039, paired CI -0.0008..+0.0059), so only the bf16 kernel is kept. AVX-512 BF16 only. The INT4 GPU decode kernel also gained FP4's copy overlap and CPU-route skipping; with `--moe-decode-fetch 1` its output is unchanged.
- **CPU-assisted decode calibration:** `scripts/calibrate.sh` measures a machine once and prints the `serve.sh` flags to use (`--moe-decode-fetch`, `--moe-cpu-decode-threads`). It probes pinned H2D per GPU and the CPU expert kernel on real experts. It then decodes at 4k and 100k context on two loads: plain decode as the reference, then CPU help with the fetched share (0.5 to 0.05) and the CPU thread count switched between requests through a loopback-only `/v1/admin/decode-tune`, which only a `FREETOKEN_CPU_DECODE_TUNABLE=1` server accepts. The share now lives in a device tensor the captured graphs read. 2x RTX 3080 at PCIe 4.0 x8, 9800X3D: one GPU picks 0.15 with 7 threads (35 -> 52 tok/s at 4k, 31 -> 49 at 100k) in 12 min; two GPUs pick 1 (every share 1-11% slower) in 6 min. Runtime switching between the split and plain path was dropped: from the second re-capture of the decode graphs at TP=2, the P2P all-reduce gives deterministic wrong output (eager decode stays correct; root cause not found).

**Memory**
- **Elastic VRAM:** KV pages and expert-cache slots share one pool of VRAM at runtime. The cache takes whatever the live context does not need, and gives slots back, coldest first, as the context grows. Decode +16% (FP4 77.8 -> 90.2 tok/s, INT4 67.6 -> 78.7), output token-identical; the full 262k context and several parallel requests (800k tokens of KV in total) still fit (`148342e`). Since the 2026-11-14 cleanup it is on by default wherever it is supported (`--elastic-kv` unset = auto): `--moe-strategy offload` with GPU decode and an MHA or QSA KV pool. There upstream's static sizing flags (`--memory-ratio`, `--gpu-memory-ratio`, `--kv-reserve-tokens`, `--moe-cache-size`/`-rate`/`-auto`, `--num-pages`/`--num-tokens`) need `--no-elastic-kv`, and the server refuses to start without it; `--elastic-kv` where elastic VRAM is not supported is an error. Every other setup uses the static split as upstream does.
- **Prefix RAM tier:** evicted conversations and their GDN state spill to pinned host RAM and come back exactly instead of being prefilled again (13.6k tokens in 27 ms instead of 6.3 s). For hosts with 128 GB of RAM: `PREFIX_CACHE_RAM_GB=16` / `--cache-ram-gb 16` (`148342e`).
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
- `scripts/serve.sh`, with a RAM check, a capped systemd scope and a RAM watchdog.

## Configuration

The fork's rules: one code path for every supported setup; behavior from measured hardware or a cost model rather than fixed thresholds; no switch for a lossless change; lossy options as documented `ft serve` flags, off by default. Environment variables are left for debugging, diagnostics and builds.

**`ft serve` flags the fork adds** ([`docs/cli.md`](cli.md)): `--[no-]elastic-kv` (default on where supported) and its tuning (`--vram-keep-free-mb`, `--expert-cache-min-slots`, `--elastic-kv-min-tokens`, `--elastic-kv-step-tokens`, `--elastic-keep-requests`, `--elastic-lend-after`, `--prefill-reserve-gb`, `--prefill-reserve-tokens`), `--cache-ram-gb`, `--moe-routed-prefill-tokens`, `--moe-decode-fetch`, `--moe-cpu-decode-threads`, `--ngram-spec` (with `--ngram-spec-max-draft`, `--ngram-spec-min-match`, `--ngram-spec-policy`), `--vision-on-cpu`, `--vision-threads`. Lossy, off by default: `--dense-mxfp8-groups` (and upstream's `--dense-fp8`).

**Engine switches still read from the environment** (set by `serve.sh`, pending a decision on whether they become defaults or flags):

| Var | `serve.sh` | Meaning |
|---|---|---|
| `FREETOKEN_WNA16_CPU_PREFILL` | `1` | hybrid CPU prefill, W4A16 and NVFP4 despite the name; not bit-reproducible |
| `FREETOKEN_WNA16_CPU_PREFILL_FETCH` | `0.35` | share of the CPU prefill's misses fetched over PCIe instead |
| `FREETOKEN_WNA16_CPU_PREFILL_THREADS` | | CPU prefill threads per GPU (default: the GPU's share of the physical cores) |
| `FREETOKEN_WNA16_PREFILL_PRECISE` | `1` | INT4: exact-dequant tile prefill, fp32 gate/up |
| `FREETOKEN_WNA16_BALANCED_TP` | `1` | INT4 at TP=2: 320/320 expert split |

**Troubleshooting:** `FREETOKEN_P2P_ALLREDUCE=0` (NCCL instead of the P2P all-reduce), `FREETOKEN_HOST_ALLREDUCE=0` (NCCL instead of the host all-reduce), `FREETOKEN_DECODE_SNAPSHOT_INTERVAL` (decode tokens between GDN snapshots, default 64, `0` off), `FREETOKEN_FTW_READ_CONCURRENCY` (lower load peak on low-RAM hosts).

**Diagnostics:** `FREETOKEN_DUMP_PREFILL_LOGITS`, `FREETOKEN_DUMP_DECODE_LOGITS` (logit dumps for KLD), `FREETOKEN_CPU_DECODE_PREFILL=1` (prefill chunks as small as a decode step take CPU-assisted decode, for teacher-forced dumps), `FREETOKEN_ROUTE_LOG`, `FREETOKEN_PROFILE_PREFILL_DIR` / `_MIN`, `FREETOKEN_PROFILE_DECODE_DIR` / `_SKIP` / `_STEPS`, `FREETOKEN_PREFIX_DEBUG`, `FREETOKEN_WNA16_CPU_PREFILL_STATS`, `FREETOKEN_NGRAM_SPEC_DEBUG`, `FREETOKEN_NGRAM_SPEC_MODE_FILE`, `FREETOKEN_NGRAM_SPEC_OFF_FILE`, `FREETOKEN_CPU_DECODE_TUNABLE` (calibration only).

**Removed in the 2026-11-14 cleanup** (they only turned off a default path, picked a superseded kernel, or retuned a measured constant): `FREETOKEN_MXFP8_PAD_SLICES`, `FREETOKEN_PLE_ASYNC_PREFILL`, `FREETOKEN_HC_FUSE`, `FREETOKEN_HC_SHARD`, `FREETOKEN_COPY_OVERLAP`, `FREETOKEN_EARLY_ROUTER`, `FREETOKEN_DECODE_FRONT_FUSE`, `FREETOKEN_TRITON_GEMV`, `FREETOKEN_EMBED_ON_HOST`, `FREETOKEN_NGRAM_SPEC_GRAPH`, `FREETOKEN_NVFP4_TILE`, `FREETOKEN_NVFP4_NK_TILE`, `FREETOKEN_NVFP4_GEMV_V2`, `FREETOKEN_WNA16_DECODE_V2`, `FREETOKEN_WNA16_PREFILL_LEGACY`, `FREETOKEN_DENSE_MXFP8`, `FREETOKEN_CPU_MOE_SPIN_US`, `FREETOKEN_PLE_QUEUE_DEPTH`, and `--moe-cpu-decode-act` (int8 was no faster and 3.6x less accurate).

## Inherited from anon761/freetoken

Compared with upstream `3d919e9`. Full list in [`CHANGELOG.md`](../CHANGELOG.md).

- DeepSeek-V4.1-Flash (`dsv41`): mHC, CSA2 sparse attention, Lightning Indexer, Engram memory, DSpark draft.
- Qwen3.8-Flash-Next vision tower and `image_url` input. PLE table dtype via `ft checkpoint --ple`.
- MTP speculative decoding (embedded, external or INT4 head) with CUDA-graphed rounds.
- TP-sharded loading, per-GPU `--memory-ratio`, fp8 all-reduce.
- Quantization dialect layer: compressed-tensors fp8/NVFP4, AutoRound W4A16 experts, `--dense-fp8`.
- FTW: TP slicing, carried side tables (PLE/Engram/MTP).

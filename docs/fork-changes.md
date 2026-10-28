# What this fork changes

Every change against [anon761/freetoken](https://github.com/anon761/freetoken), with its commit and measured effect. The env vars are runtime switches for A/B runs; the defaults, and [`scripts/serve-qwen38.sh`](../scripts/serve-qwen38.sh), already take the fast path.

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
- **CPU-assisted decode (FP4):** decode fetches only a share of its expert-cache misses over PCIe; the CPU computes the rest straight from the pinned banks (~57 us and ~50 GB/s per expert on 7 Zen 5 cores) while the GPU runs the hits, and its fp32 partial joins the route sum before the one rounding. Every routed expert still runs, on one side or the other. One RTX 3080 at PCIe 4.0 x8: decode 28 -> 53 tok/s at 4k context, 26 -> 50 at 131k, 20 -> 46 at 261k; prefill unchanged. Per MoE layer the fp32 CPU path is as close to exact W4 math as the GPU kernel (2.19e-3 rel-RMS both); int8 activations (W4A8) are no faster and 3.6x further off. Output is not token-identical to the GPU path: teacher-forced KLD vs it is 0.035-0.040, the same as a GPU-only change of summation order (0.029). On two GPUs only ~4% of routes miss and it does not pay. `--moe-decode-fetch` / `DECODE_FETCH` (the fetched share; 1 = off, 0.1 by default on one GPU), `--moe-cpu-decode-act fp32|int8`, `--moe-cpu-decode-threads` (`fca53ea`).

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

Compared with upstream `3d919e9`. Full list in [`CHANGELOG.md`](../CHANGELOG.md).

- DeepSeek-V4.1-Flash (`dsv41`): mHC, CSA2 sparse attention, Lightning Indexer, Engram memory, DSpark draft.
- Qwen3.8-Flash-Next vision tower and `image_url` input. PLE table dtype via `ft checkpoint --ple`.
- MTP speculative decoding (embedded, external or INT4 head) with CUDA-graphed rounds.
- TP-sharded loading, per-GPU `--memory-ratio`, fp8 all-reduce.
- Quantization dialect layer: compressed-tensors fp8/NVFP4, AutoRound W4A16 experts, `--dense-fp8`.
- FTW: TP slicing, carried side tables (PLE/Engram/MTP).

#!/bin/bash
# Serve Qwen3.8-Flash-Next (Intel INT4 W4A16 AutoRound or NVIDIA FP4-Local) from an FTW
# checkpoint: experts in pinned host RAM behind a VRAM LRU expert cache, PLE n-gram table read
# from disk, vision tower on the CPU, full 262144 context with bf16 KV. Tuned on 2x RTX 3080
# 20GB (PCIe 4.0 x8 each). Every knob is env-overridable; see scripts/README.md.
set -u
HERE=$(dirname "$(readlink -f "$0")")
REPO=$(dirname "$HERE")

: "${SNAP:?set SNAP to the FTW checkpoint dir (see scripts/README.md)}"
: "${FT:=$REPO/.venv/bin/ft}"
[ -x "$FT" ] || FT=$(command -v ft) || { echo "ft not found: install the repo or set FT" >&2; exit 1; }

fmt=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("quant_format", ""))' \
  "$SNAP/freetoken_weight.json" 2>/dev/null)
case "$fmt" in
  w4a16) : "${NAME:=q38-int4}" ;;
  nvfp4) : "${NAME:=q38-fp4}" ;;
  *) : "${NAME:=$(basename "$SNAP")}" ;;
esac

if [ -z "${GPUS:-}" ]; then        # nvidia-smi indices, entry i = TP rank i; TP size = count
  GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | head -2 | paste -sd,)
  : "${GPUS:=0}"
  echo "serve-qwen38: GPUS=$GPUS (detected; set GPUS to choose)" >&2
fi
IFS=, read -ra gpu_list <<<"$GPUS"
: "${ELASTIC:=1}"               # 1 = KV and the expert cache share VRAM at runtime: the cache gets
                                # whatever the live context does not need (decode +16%, output
                                # identical) and MAXLEN shrinks to what VRAM can hold; replaces MEM_RATIO
if [ "$ELASTIC" = "0" ] && [ "${#gpu_list[@]}" -lt 2 ]; then
  # one 20 GB GPU cannot hold the full 262k KV beside the 1024-slot expert-cache minimum and needs
  # more activation headroom than a TP=2 rank
  : "${MAXLEN:=131072}" "${MEM_RATIO:=0.84}"
fi
: "${MAXLEN:=262144}"           # longest request (prompt + output)
: "${PREFILL_CHUNK:=8192}"      # tokens per prefill chunk; elastic keeps 256 MiB + 230 KiB per token
                                # free for its activations (8192: 2.0 GiB, 4096: 1.2, 2048: 0.7). Halving
                                # it roughly halves long-prompt prefill speed; it buys context on 16 GB.
: "${PORT:=8080}"
: "${HOST:=127.0.0.1}"
: "${MTP:=off}"
: "${VISION_THREADS:=8}"        # host ViT intra-op threads (16 -> 3.5 s, 4 -> 6.8 s per full-size image)
: "${DENSE_FP8:=0}"             # 1 = fp8 dense linears (faster decode, lossy)
: "${MEM_RATIO:=0.86}"          # ELASTIC=0: VRAM share for weights + expert cache + KV; the rest is
                                # activation headroom. 0.87+ OOMed on 20 GB cards on the first decode
                                # after a full 262k prefill.
: "${CPU_PREFILL:=1}"           # 1 = routed prefill computes most VRAM-cache misses on the CPU
                                # (AVX-512 BF16, 4 physical cores per rank) while the GPU fetches the
                                # rest; 0 = GPU only (bit-reproducible with DECODE_FETCH=1, use both for A/B runs)
: "${CPU_PREFILL_FETCH:=0.35}"  # share of the misses fetched over PCIe (the most-routed ones)
: "${PRECISE_PREFILL:=1}"       # W4A16: apply group scales after the dot, gate/up in fp32 (no cost)
# CPU-assisted decode: explicit env > this machine's calibration for this GPU set
# (calibrate-qwen38.sh) > the defaults below
: "${CALIBRATION:=${XDG_CONFIG_HOME:-$HOME/.config}/freetoken/calibration-gpus-$(echo "$GPUS" | tr -c '0-9\n' '-').env}"
decode_src=env
if [ "$fmt" = nvfp4 ] && [ -z "${DECODE_FETCH:-}" ] && [ -r "$CALIBRATION" ]; then
  cal_fetch=$(sed -n 's/^DECODE_FETCH=\([0-9.]*\)$/\1/p' "$CALIBRATION" | tail -1)
  cal_threads=$(sed -n 's/^CPU_DECODE_THREADS=\([0-9]*\)$/\1/p' "$CALIBRATION" | tail -1)
  if [ -n "$cal_fetch" ]; then
    DECODE_FETCH=$cal_fetch decode_src="calibration $CALIBRATION"
    [ -n "$cal_threads" ] && : "${CPU_DECODE_THREADS:=$cal_threads}"
  fi
fi
[ -z "${DECODE_FETCH:-}" ] && decode_src=default
# FP4 on one GPU: decode computes most expert-cache misses on the CPU (28 -> 53 tok/s at 4k, 20 -> 46
# at 261k); two GPUs miss too rarely for it to pay (93 -> 89 tok/s at 0.5)
[ "${#gpu_list[@]}" -lt 2 ] && [ "$fmt" = nvfp4 ] && : "${DECODE_FETCH:=0.1}"
: "${DECODE_FETCH:=1}"          # FP4: share of decode's expert-cache misses fetched over PCIe into the cache;
                                # the CPU computes the rest from the pinned banks (1 = all over PCIe, 0 = all
                                # on the CPU, and the cache then only changes during prefill)
: "${CPU_DECODE_THREADS:=}"     # DECODE_FETCH < 1: CPU threads per GPU; empty = all physical cores but one
                                # on one GPU, the GPU's share of them with several
: "${CPU_DECODE_ACT:=fp32}"     # DECODE_FETCH < 1: CPU expert activations, fp32 (exact) or int8 (W4A8, no
                                # faster here: the CPU GEMV is memory-bound either way)
if [ "$CPU_PREFILL" = "1" ]; then : "${ROUTED_PF:=3072}"; else : "${ROUTED_PF:=1280}"; fi
                                # prefill chunks <= ROUTED_PF tokens load only their routed experts
                                # instead of streaming every expert over PCIe (~2.4 s per chunk at
                                # x8 gen4); re-tune on a faster link
: "${HIT_D2D:=1}"               # 1 = longer chunks copy VRAM-cached experts device-side and stream only
                                # the misses (pp16384 -14% warm, output identical)
: "${BALANCED:=1}"              # W4A16 at TP=2: even 320/320 expert split (vs 256/384)
: "${KEEP_FREE_MB:=128}"        # elastic: VRAM left free per GPU, also at the prefill peak
: "${PREFILL_RESERVE_GB:=}"     # elastic: prefill activation reserve per GPU; empty = auto
: "${CACHE_RAM_GB:=0}"          # whole GiB of pinned host RAM (all GPUs) for evicted prompt prefixes:
                                # a later turn reloads them instead of prefilling again. 0 = off.
                                # Needs RAM beyond the 96 GB class; raises MIN_AVAIL_GIB and MEM_MAX.
: "${MIN_AVAIL_GIB:=$((75 + CACHE_RAM_GB))}"  # refuse to start below this much MemAvailable
: "${CGROUP:=1}"                # 1 = run in a systemd --user scope with a hard RAM cap, no swap
: "${MEM_MAX:=$((78 + CACHE_RAM_GB))G}"
: "${MEMGUARD:=1}"              # 1 = kill the server when MemAvailable drops under FLOOR GiB
: "${FLOOR:=2}"                # GiB; the server itself is capped by MEM_MAX, so this guards the rest of the desktop
: "${EXTRA:=}"

# Pinned memory cannot swap, so an overshoot freezes the host instead of reaching the OOM killer.
avail_gib=$(awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo)
if [ "$avail_gib" -lt "$MIN_AVAIL_GIB" ]; then
  echo "REFUSING TO START: ${avail_gib} GiB available, need ${MIN_AVAIL_GIB}." >&2
  exit 1
fi
venv_bin=$(dirname "$(readlink -f "$FT")")
[ "$MEMGUARD" = "1" ] && FLOOR=$FLOOR LOG="${MEMGUARD_LOG:-/tmp/ft-memguard.log}" PAT="^$venv_bin/" \
  "$HERE/memguard.sh" >/dev/null 2>&1 &

args=(
  --model "$SNAP"
  --served-model-name "$NAME"
  --tp-size "${#gpu_list[@]}" --gpu "$GPUS"
  --moe-strategy offload
  --max-seq-len-override "$MAXLEN"
  --max-running-requests 1
  --mtp "$MTP"
  --vision-on-cpu --vision-threads "$VISION_THREADS"
  --moe-routed-prefill-tokens "$ROUTED_PF"
  --max-prefill-length "$PREFILL_CHUNK"
  --enable-cache-report
  --host "$HOST" --port "$PORT"
)
if [ "$ELASTIC" = "1" ]; then
  args+=(--elastic-kv --vram-keep-free-mb "$KEEP_FREE_MB")
  [ -n "$PREFILL_RESERVE_GB" ] && args+=(--prefill-reserve-gb "$PREFILL_RESERVE_GB")
else
  args+=(--kv-reserve-tokens "$MAXLEN" --memory-ratio "$MEM_RATIO")
fi
[ "$CACHE_RAM_GB" != "0" ] && args+=(--cache-ram-gb "$CACHE_RAM_GB")
[ "$DENSE_FP8" = "1" ] && args+=(--dense-fp8)
[ "$HIT_D2D" = "1" ] && args+=(--moe-prefill-hit-d2d)
[ -n "$EXTRA" ] && args+=($EXTRA)

export FREETOKEN_WNA16_BALANCED_TP="$BALANCED"
export FREETOKEN_WNA16_CPU_PREFILL="$CPU_PREFILL"
export FREETOKEN_WNA16_CPU_PREFILL_FETCH="$CPU_PREFILL_FETCH"
export FREETOKEN_WNA16_PREFILL_PRECISE="$PRECISE_PREFILL"
export FREETOKEN_MOE_DECODE_FETCH="$DECODE_FETCH"
export FREETOKEN_CPU_DECODE_ACT="$CPU_DECODE_ACT"
[ -n "$CPU_DECODE_THREADS" ] && export FREETOKEN_CPU_DECODE_THREADS="$CPU_DECODE_THREADS"
if [ "$fmt" = nvfp4 ]; then
  echo "serve-qwen38: DECODE_FETCH=$DECODE_FETCH CPU_DECODE_THREADS=${CPU_DECODE_THREADS:-auto} ($decode_src)" >&2
fi

unit="ft-serve-$$"
if [ "$CGROUP" = "1" ] && command -v systemd-run >/dev/null; then
  # --scope execs ft in place, so $! is the server's main process
  systemd-run --user --scope --quiet --unit="$unit" -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0 \
    "$FT" serve "${args[@]}" "$@" &
else
  CGROUP=0
  setsid "$FT" serve "${args[@]}" "$@" &
fi
ft_pid=$!

# ft exits on TERM in ~1 s, but a TP worker that outlives it keeps a GPU and its pinned experts
cleanup() {
  kill -TERM "$ft_pid" 2>/dev/null
  for _ in $(seq 14); do kill -0 "$ft_pid" 2>/dev/null || break; sleep 0.5; done
  if [ "$CGROUP" = "1" ]; then
    systemctl --user kill --signal=SIGKILL "$unit.scope" 2>/dev/null
  else
    kill -KILL -- "-$ft_pid" 2>/dev/null
  fi
}
trap 'cleanup; exit 143' TERM INT HUP

# /v1/models answers as soon as the frontend is up; /health says "ok" only once the model is loaded
(
  until curl -sf -m 2 "http://127.0.0.1:$PORT/health" | grep -q '"status": *"ok"'; do
    kill -0 "$ft_pid" 2>/dev/null || exit 0
    sleep 3
  done
  echo "FreeToken: model loaded on port $PORT, chat UI at http://$HOST:$PORT/"
) &

wait "$ft_pid"

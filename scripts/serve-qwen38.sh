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

: "${GPUS:=0,1}"                # nvidia-smi indices, entry i = TP rank i; TP size = count
: "${MAXLEN:=262144}"
: "${PORT:=8080}"
: "${HOST:=127.0.0.1}"
: "${MTP:=off}"
: "${VISION_THREADS:=8}"        # host ViT intra-op threads (16 -> 3.5 s, 4 -> 6.8 s per full-size image)
: "${DENSE_FP8:=0}"             # 1 = fp8 dense linears (faster decode, lossy)
: "${MEM_RATIO:=0.86}"          # VRAM share for weights + expert cache + KV; the rest is activation
                                # headroom. 0.87+ OOMed on 20 GB cards on the first decode after a
                                # full 262k prefill. One unsplit GPU needs more headroom: start at 0.84.
: "${CPU_PREFILL:=1}"           # 1 = routed prefill computes most VRAM-cache misses on the CPU
                                # (AVX-512 BF16, 4 physical cores per rank) while the GPU fetches the
                                # rest; 0 = GPU only (bit-reproducible, use it for A/B runs)
: "${CPU_PREFILL_FETCH:=0.35}"  # share of the misses fetched over PCIe (the most-routed ones)
: "${PRECISE_PREFILL:=1}"       # W4A16: apply group scales after the dot, gate/up in fp32 (no cost)
if [ "$CPU_PREFILL" = "1" ]; then : "${ROUTED_PF:=3072}"; else : "${ROUTED_PF:=1280}"; fi
                                # prefill chunks <= ROUTED_PF tokens load only their routed experts
                                # instead of streaming every expert over PCIe (~2.4 s per chunk at
                                # x8 gen4); re-tune on a faster link
: "${HIT_D2D:=1}"               # 1 = longer chunks copy VRAM-cached experts device-side and stream only
                                # the misses (pp16384 -14% warm, output identical)
: "${BALANCED:=1}"              # W4A16 at TP=2: even 320/320 expert split (vs 256/384)
: "${MIN_AVAIL_GIB:=75}"        # refuse to start below this much MemAvailable
: "${CGROUP:=1}"                # 1 = run in a systemd --user scope with a hard RAM cap, no swap
: "${MEM_MAX:=78G}"
: "${MEMGUARD:=1}"              # 1 = kill the server when MemAvailable drops under FLOOR GiB
: "${FLOOR:=8}"
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

IFS=, read -ra gpu_list <<<"$GPUS"
args=(
  --model "$SNAP"
  --served-model-name "$NAME"
  --tp-size "${#gpu_list[@]}" --gpu "$GPUS"
  --moe-strategy offload
  --max-seq-len-override "$MAXLEN"
  --kv-reserve-tokens "$MAXLEN"
  --max-running-requests 1
  --memory-ratio "$MEM_RATIO"
  --mtp "$MTP"
  --vision-on-cpu --vision-threads "$VISION_THREADS"
  --moe-routed-prefill-tokens "$ROUTED_PF"
  --enable-cache-report
  --host "$HOST" --port "$PORT"
)
[ "$DENSE_FP8" = "1" ] && args+=(--dense-fp8)
[ "$HIT_D2D" = "1" ] && args+=(--moe-prefill-hit-d2d)
[ -n "$EXTRA" ] && args+=($EXTRA)

export FREETOKEN_WNA16_BALANCED_TP="$BALANCED"
export FREETOKEN_WNA16_CPU_PREFILL="$CPU_PREFILL"
export FREETOKEN_WNA16_CPU_PREFILL_FETCH="$CPU_PREFILL_FETCH"
export FREETOKEN_WNA16_PREFILL_PRECISE="$PRECISE_PREFILL"

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

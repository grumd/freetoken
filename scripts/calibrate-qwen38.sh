#!/bin/bash
# One-time calibration of CPU-assisted decode (DECODE_FETCH, CPU_DECODE_THREADS) for this machine
# and GPU setup. Run it once with the SNAP / GPUS you serve with; serve-qwen38.sh then reads the
# env file it writes. Measures PCIe and the CPU expert kernel, then loads the model twice through
# serve-qwen38.sh (plain decode, then CPU-assisted with the settings switched at runtime).
# See scripts/README.md.
set -u
HERE=$(dirname "$(readlink -f "$0")")
REPO=$(dirname "$HERE")

: "${SNAP:?set SNAP to the FTW checkpoint dir, as for serve-qwen38.sh}"
: "${GPUS:=0,1}"
: "${FT:=$REPO/.venv/bin/ft}"
[ -x "$FT" ] || FT=$(command -v ft) || { echo "ft not found: install the repo or set FT" >&2; exit 1; }
: "${PY:=$(dirname "$(readlink -f "$FT")")/python}"
: "${CAL_PORT:=18080}"          # the calibration servers' port (loopback only)
: "${SHORT_CTX:=4096}"          # context tokens of the short-context runs
: "${LONG_CTX:=100000}"         # context tokens of the long-context runs (capped to what fits); 0 = skip
: "${CAL_TOKENS:=192}"          # measured decode tokens per prompt, after CAL_SKIP warm-up tokens
: "${CAL_SKIP:=48}"
: "${THREADS:=}"                # fix the CPU decode thread count instead of sweeping it
: "${CAL_DIR:=${XDG_CONFIG_HOME:-$HOME/.config}/freetoken}"
: "${LOAD_TIMEOUT:=900}"        # seconds to wait for each model load

key=$(echo "$GPUS" | tr -c '0-9\n' '-')
ENV_OUT="$CAL_DIR/calibration-gpus-$key.env"
work=$(mktemp -d "${TMPDIR:-/tmp}/ft-calibrate.XXXXXX")
export CUDA_DEVICE_ORDER=PCI_BUS_ID
t_start=$(date +%s)

# a desktop compositor holds a few MiB; anything with real VRAM would skew the numbers
if nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '$1 >= 256 {f=1} END {exit !f}'; then
  echo "GPU compute processes are running; stop any server first (calibration needs idle GPUs):" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2
  exit 1
fi

echo "== hardware probe (GPUS=$GPUS) =="
"$PY" "$HERE/calibrate_qwen38.py" probe --snap "$SNAP" --gpus "$GPUS" --out "$work/hw.json" || exit 1

fmt=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["quant_format"])' "$work/hw.json")
isa=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("cpu_kernel", {}).get("isa", ""))' "$work/hw.json")
if [ "$fmt" != nvfp4 ]; then
  "$PY" "$HERE/calibrate_qwen38.py" skip --hw "$work/hw.json" --env "$ENV_OUT" \
    --why "the checkpoint is $fmt; CPU-assisted decode serves NVFP4 (FP4) experts only"
  exit 0
fi
case "$isa" in
  scalar*)
    "$PY" "$HERE/calibrate_qwen38.py" skip --hw "$work/hw.json" --env "$ENV_OUT" \
      --why "the CPU has no AVX2/AVX-512, so the CPU expert kernel runs scalar code (far slower than a PCIe fetch)"
    exit 0 ;;
esac

echo
echo "== decode sweep: two loads, DECODE_FETCH=1 then CPU-assisted (logs in $work) =="
"$PY" "$HERE/calibrate_qwen38.py" run --hw "$work/hw.json" --snap "$SNAP" --gpus "$GPUS" \
  --port "$CAL_PORT" --work "$work" --env "$ENV_OUT" --out "$work/result.json" \
  --short "$SHORT_CTX" --long "$LONG_CTX" --tokens "$CAL_TOKENS" --skip "$CAL_SKIP" \
  ${THREADS:+--threads "$THREADS"} --load-timeout "$LOAD_TIMEOUT" --total-s "$(( $(date +%s) - t_start ))"
rc=$?
[ "$rc" = 0 ] && cp "$work/result.json" "${ENV_OUT%.env}.json"
echo "calibration took $(( $(date +%s) - t_start )) s; scratch files in $work"
exit "$rc"

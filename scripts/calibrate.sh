#!/bin/bash
# Measure the best CPU-assisted decode settings (FP4) for this machine and GPU setup, and print
# the serve.sh flags to use them. Run it with the MODEL / GPUS you serve with.
#
#   MODEL=~/models/ftw/q38-fp4 scripts/calibrate.sh
#
# Probes PCIe and the CPU expert kernel, then loads the model twice through serve.sh: plain
# decode, then CPU-assisted with the settings switched at runtime. See scripts/README.md.
set -u
HERE=$(dirname "$(readlink -f "$0")")

# ---- settings ----------------------------------------------------------------------------------

: "${MODEL:?set MODEL to the FTW checkpoint dir, as for serve.sh}"
: "${GPUS:=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | head -2 | paste -sd,)}"
: "${GPUS:=0}"
: "${CALIBRATION_PORT:=18080}"
: "${SHORT_CONTEXT_TOKENS:=4096}"
: "${LONG_CONTEXT_TOKENS:=100000}"
: "${MEASURED_TOKENS:=192}"
: "${WARMUP_TOKENS:=48}"
: "${FIXED_CPU_THREADS:=}"
: "${LOAD_TIMEOUT_S:=900}"
: "${FT_BIN:=$(dirname "$HERE")/.venv/bin/ft}"
[ -x "$FT_BIN" ] || FT_BIN=$(command -v ft) || { echo "ft not found: install the repo or set FT_BIN" >&2; exit 1; }
PY=$(dirname "$(readlink -f "$FT_BIN")")/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID

work=$(mktemp -d "${TMPDIR:-/tmp}/ft-calibrate.XXXXXX")
t_start=$(date +%s)

# a desktop compositor holds a few MiB; anything with real VRAM would skew the numbers
if nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '$1 >= 256 {f=1} END {exit !f}'; then
  echo "GPU compute processes are running; stop any server first (calibration needs idle GPUs):" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2
  exit 1
fi

# ---- probe -------------------------------------------------------------------------------------

echo "== hardware probe (GPUS=$GPUS) =="
"$PY" "$HERE/calibrate.py" probe --snap "$MODEL" --gpus "$GPUS" --out "$work/hw.json" || exit 1

fmt=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["quant_format"])' "$work/hw.json")
isa=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("cpu_kernel", {}).get("isa", ""))' "$work/hw.json")
if [ "$fmt" != nvfp4 ]; then
  exec "$PY" "$HERE/calibrate.py" skip --hw "$work/hw.json" \
    --why "the checkpoint is $fmt; CPU-assisted decode serves NVFP4 (FP4) experts only"
fi
case "$isa" in
  scalar*)
    exec "$PY" "$HERE/calibrate.py" skip --hw "$work/hw.json" \
      --why "the CPU has no AVX2/AVX-512, so the CPU expert kernel runs scalar code (far slower than a PCIe fetch)" ;;
esac

# ---- decode sweep ------------------------------------------------------------------------------

echo
echo "== decode sweep: two loads, plain then CPU-assisted (logs in $work) =="
"$PY" "$HERE/calibrate.py" run --hw "$work/hw.json" --snap "$MODEL" --gpus "$GPUS" \
  --port "$CALIBRATION_PORT" --work "$work" --out "$work/result.json" \
  --short "$SHORT_CONTEXT_TOKENS" --long "$LONG_CONTEXT_TOKENS" --tokens "$MEASURED_TOKENS" --skip "$WARMUP_TOKENS" \
  ${FIXED_CPU_THREADS:+--threads "$FIXED_CPU_THREADS"} --load-timeout "$LOAD_TIMEOUT_S" --total-s "$(( $(date +%s) - t_start ))"
rc=$?
echo "calibration took $(( $(date +%s) - t_start )) s; logs and raw numbers in $work"
exit "$rc"

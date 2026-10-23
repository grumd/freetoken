#!/bin/bash
# RAM watchdog for serve-qwen38.sh: SIGKILL every process whose command line matches PAT once
# MemAvailable drops under FLOOR GiB. Pinned pages do not swap (and zram swap is DRAM anyway),
# so an overshoot freezes the host instead of reaching the OOM killer. Exits when the server does.
set -u
: "${PAT:?set PAT to a regex matching the server processes, e.g. ^/path/to/.venv/bin/}"
FLOOR=${FLOOR:-8}
LOG=${LOG:-/tmp/ft-memguard.log}

echo "$(date +%H:%M:%S) memguard: floor ${FLOOR} GiB, pattern '$PAT'" >> "$LOG"
while true; do
  avail=$(awk '/^MemAvailable:/{printf "%.2f", $2/1048576}' /proc/meminfo)
  if awk "BEGIN{exit !($avail < $FLOOR)}"; then
    echo "$(date +%H:%M:%S) ABORT avail=${avail} GiB -- killing FreeToken" >> "$LOG"
    pkill -9 -f "$PAT"
    exit 1
  fi
  pgrep -f "$PAT" >/dev/null || { [ "${ARMED:-0}" = 1 ] && exit 0; }
  pgrep -f "$PAT" >/dev/null && ARMED=1
  sleep 0.5
done

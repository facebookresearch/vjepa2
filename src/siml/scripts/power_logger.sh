#!/usr/bin/env bash
OUT="${1:-gpu_power_log.csv}"
echo "timestamp_ms,power_W" > "$OUT"
while true; do
  T=$(date +%s%3N)
  P=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -n1)
  echo "$T,$P" >> "$OUT"
  sleep 1
done

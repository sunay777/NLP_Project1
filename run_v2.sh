#!/bin/bash
# One-command runner for the v2 experiments (see RUN_V2.md).
#   ./run_v2.sh setup      create .venv, install pinned packages, run the smoke test
#   ./run_v2.sh bench      time this Mac and print the expected run time
#   ./run_v2.sh start      start the full queue in the background (safe to re-run: it resumes)
#   ./run_v2.sh start 1    start only stage 1 (the core results)
#   ./run_v2.sh status     progress + ETA
#   ./run_v2.sh stop       stop the queue (finished runs are kept)
#   ./run_v2.sh pack       zip the results to send back (~/Desktop/v2_results_*.tar.gz)
#   ./run_v2.sh packjson   small zip with only the numbers (JSON + logs, no weights)
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
LOG=results/v2/_queue/queue.log
case "${1:-help}" in
  setup)
    SYS=$(command -v python3.12 || command -v python3.13 || command -v python3.11 || command -v python3)
    "$SYS" -c 'import sys; assert sys.version_info >= (3, 10), "need Python >= 3.10, have " + sys.version' 
    [ -d .venv ] || "$SYS" -m venv .venv
    $PY -m pip install --upgrade pip -q
    $PY -m pip install -r requirements.txt -q
    $PY -c 'import torch, numpy, matplotlib; print("torch", torch.__version__, "| numpy", numpy.__version__, "| matplotlib", matplotlib.__version__)'
    $PY -m experiments.smoke_test ;;
  bench)
    $PY -m experiments.run_queue --benchmark ;;
  start)
    mkdir -p results/v2/_queue
    if pgrep -f "experiments.run_queue" >/dev/null; then echo "queue already running"; exit 0; fi
    STAGES=${2:+--stages ${@:2}}
    nohup $PY -m experiments.run_queue $STAGES >> "$LOG" 2>&1 &
    echo "queue started (pid $!). Log: $LOG   Progress: ./run_v2.sh status" ;;
  status)
    $PY -m experiments.run_queue --status; echo "--- last log lines:"; tail -n 5 "$LOG" 2>/dev/null || true ;;
  stop)
    pkill -INT -f "experiments.run_queue" && echo "stopping..." || echo "queue not running" ;;
  pack)
    OUT=~/Desktop/v2_results_$(hostname -s)_$(date +%Y%m%d_%H%M).tar.gz
    tar -czf "$OUT" results/v2 && ls -lh "$OUT" ;;
  packjson)
    OUT=~/Desktop/v2_numbers_$(hostname -s)_$(date +%Y%m%d_%H%M).tar.gz
    find results/v2 \( -name "*.json" -o -name "*.log" \) -print0 | tar -czf "$OUT" --null -T - && ls -lh "$OUT" ;;
  *)
    sed -n '2,12p' "$0" ;;
esac

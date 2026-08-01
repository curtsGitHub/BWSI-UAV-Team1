#!/usr/bin/env bash
# fly_and_log.sh -- run a flight program, capture ALL console output (including
# tracebacks and crashes) to a timestamped file inside the repo, then git commit
# it and best-effort push once the program has exited. Git never runs mid-flight.
#
# Usage, from the module directory on the Pi:
#   bash ../../diagnostics/fly_and_log.sh              # runs: python3 main.py
#   bash ../../diagnostics/fly_and_log.sh python3 main.py -s   # or any command
#
# The push needs git credentials on the Pi (same ones `git pull` uses). If the
# push fails (no network/creds), the log is still committed locally -- run
# `git push` later and nothing is lost.
set -u

REPO="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "[FLY] not inside a git repo"; exit 1; }
LOGDIR="$REPO/flight_logs"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOGDIR/${STAMP}_flight.log"

if [ $# -eq 0 ]; then set -- python3 main.py; fi
echo "[FLY] running: $* (log: $LOG)"
"$@" 2>&1 | tee "$LOG"
EXIT=${PIPESTATUS[0]}

VER="$(grep -m1 '^\[VER\]' "$LOG" || echo 'no [VER] line -- stale code?')"
echo "[FLY] program exited with code $EXIT ($VER)"

git -C "$REPO" add -- "$LOG"
if git -C "$REPO" commit -m "flight log $STAMP (exit $EXIT; $VER)" -- "$LOG"; then
    echo "[FLY] committed $(basename "$LOG")"
else
    echo "[FLY] git commit failed -- log is still on disk at $LOG"
    exit "$EXIT"
fi

if git -C "$REPO" push; then
    echo "[FLY] pushed -- log is on the remote"
else
    echo "[FLY] PUSH FAILED (network/credentials?) -- committed locally; run 'git push' when you can"
fi
exit "$EXIT"

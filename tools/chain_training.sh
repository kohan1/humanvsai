#!/usr/bin/env bash
# Wait for the Snake run to finish, then start Watermelon automatically.
#
# Waits on the LOG rather than on a process id: the pipeline spawns and reaps
# many python processes (SubprocVecEnv workers), so "are any python processes
# alive" is not a reliable finished-signal, and a pid captured at launch dies
# long before the pipeline does.
#
# Snake's pipeline prints exactly one of these when it stops:
#   "=== PIPELINE COMPLETE ==="   success
#   "PIPELINE ABORTED"           a gate rejected it
#   "PIPELINE FAILED"            a stage crashed
#
# Watermelon starts only on success. If Snake aborted or failed, chaining would
# just queue a second run behind a problem that needs looking at.

set -o pipefail
# This script lives in tools/, so step up to the project root.
cd "$(dirname "$0")/.."

SNAKE_LOG="snake/training/v2_pipeline.log"
WATERMELON_DIR="watermelon/training"
CHAIN_LOG="tools/chain_training.log"
POLL=60

log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$CHAIN_LOG"; }

log "chain started; waiting for snake to finish (polling every ${POLL}s)"

while true; do
    if [ -f "$SNAKE_LOG" ]; then
        if grep -q "=== PIPELINE COMPLETE ===" "$SNAKE_LOG"; then
            log "snake finished successfully"
            break
        fi
        if grep -qE "PIPELINE ABORTED|PIPELINE FAILED" "$SNAKE_LOG"; then
            log "snake did NOT finish cleanly — not starting watermelon"
            grep -E "PIPELINE ABORTED|PIPELINE FAILED" "$SNAKE_LOG" | tail -3 | tee -a "$CHAIN_LOG"
            log "chain stopping. Fix snake, then run watermelon manually."
            exit 1
        fi
    fi
    sleep "$POLL"
done

# Report how snake actually did, so the chain log is a complete record.
log "--- snake result ---"
grep -E "^BC:|^Final:|PPO improved|made the policy WORSE" "$SNAKE_LOG" | tail -4 | tee -a "$CHAIN_LOG"

log "starting watermelon pipeline"
cd "$WATERMELON_DIR" || { log "no $WATERMELON_DIR"; exit 1; }

bash run_pipeline.sh > pipeline.log 2>&1
STATUS=$?

cd - > /dev/null
if [ $STATUS -eq 0 ]; then
    log "watermelon pipeline finished (exit 0)"
    grep -E "^BC:|^Final:|PPO improved|made the policy WORSE" "$WATERMELON_DIR/pipeline.log" | tail -4 | tee -a "$CHAIN_LOG"
else
    log "watermelon pipeline exited $STATUS — see $WATERMELON_DIR/pipeline.log"
    tail -5 "$WATERMELON_DIR/pipeline.log" | tee -a "$CHAIN_LOG"
fi

log "chain complete"

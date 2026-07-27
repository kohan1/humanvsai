#!/usr/bin/env bash
# One-shot health check for an unattended training run.
#
#     bash tools/check_training.sh
#
# Prints a verdict on the last line, one of:
#
#   OK        <game>   progressing normally
#   STALLED   <game>   process alive but the step counter has not moved
#   DEAD      <game>   no training process and the run never reached its target
#   FINISHED  <game>   reached its target and saved a final model
#   IDLE               nothing is training and nothing was expected to be
#
# WHY A SCRIPT AND NOT A PROMPT
# The periodic check has to reach the same conclusion every time, hours apart,
# with no memory of the previous check. Progress is judged by comparing against
# a stored step count in .check_state, so "not moving" is measured rather than
# eyeballed.
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
STATE="$ROOT/tools/.check_state"

# The active run is whichever game has a log that is still being appended to.
# Finished runs are moved into logs/, so a top-level *.log is by definition the
# current one.
GAME=""
LOG=""
for g in snake watermelon; do
    for f in "$g"/training/*.log; do
        [ -e "$f" ] || continue
        GAME="$g"; LOG="$f"
    done
done

if [ -z "$LOG" ]; then
    echo "no in-progress log found under snake/training or watermelon/training"
    echo "IDLE"
    exit 0
fi

echo "log:      $LOG"

# Is a trainer actually running? Match the parent python process, not the
# multiprocessing workers, which linger briefly after a parent dies.
PARENT=$(powershell.exe -NoProfile -Command \
    "(Get-CimInstance Win32_Process -Filter \"Name like '%python%' and CommandLine like '%train.py%'\" | Where-Object { \$_.CommandLine -notmatch 'spawn_main' } | Measure-Object).Count" 2>/dev/null | tr -d '\r\n ')
PARENT=${PARENT:-0}

STEPS=$(grep total_timesteps "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+" || echo 0)
STEPS=${STEPS:-0}
TARGET=$(grep -oE "/ *[0-9,]+ *\[" "$LOG" 2>/dev/null | tail -1 | tr -d '/[ ,' || echo "")
BEST=$(grep "\[best\]" "$LOG" 2>/dev/null | tail -1 | sed 's/^ *//' || echo "none yet")
SAVED=$(grep -c "Saved final model" "$LOG" 2>/dev/null; true)
SAVED=$(printf "%s" "$SAVED" | head -1)
SAVED=${SAVED:-0}

echo "process:  $PARENT trainer(s) running"
echo "steps:    $STEPS"
echo "best:     $BEST"

PREV=0
[ -f "$STATE" ] && PREV=$(cut -d' ' -f2 "$STATE" 2>/dev/null || echo 0)
PREV=${PREV:-0}
echo "$GAME $STEPS $(date +%s)" > "$STATE"

if [ "$SAVED" -gt 0 ] && [ "$PARENT" -eq 0 ]; then
    echo "FINISHED $GAME"
elif [ "$PARENT" -eq 0 ]; then
    echo "DEAD $GAME"
elif [ "$STEPS" -le "$PREV" ] && [ "$PREV" -gt 0 ]; then
    # Alive but no progress since the previous check. Evaluations pause the
    # step counter, so this is only meaningful across checks minutes apart.
    echo "no movement since last check (was $PREV)"
    echo "STALLED $GAME"
else
    echo "OK $GAME"
fi

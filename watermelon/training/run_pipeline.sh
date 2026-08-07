#!/usr/bin/env bash
# Watermelon training pipeline, with gates.
#
# Same structure as snake/training/run_v2_pipeline.sh, and for the same reason:
# the ungated version of that script spent hours fine-tuning a BC policy that
# had already evaluated at 0.00. Every stage here has to earn the next one.
#
# PYTHONIOENCODING=utf-8: torch/SB3 print glyphs cp1252 cannot encode.
# python -u: stdout is a pipe here, and buffering means no visible progress.
set -o pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8

# Recalibrated for the compressed diameter ladder. The old numbers (teacher
# ~800, random ~500, gate 550) are all obsolete: with the top-tier sink now
# reachable, the same heuristic scores ~4160 over ~370 drops and random scores
# ~3090. A gate of 550 would pass literally anything, including a policy that
# had learned nothing at all — a dead gate is worse than no gate, because it
# reads as protection.
#
# Random is unusually strong here, only 1.35x behind the teacher, because
# merges are easy to hit by accident once fruit are small. So the gate sits
# just above random rather than close to the teacher.
MIN_BC_SCORE=3200

score_of() {
    grep -oE "Average score: [0-9.]+" "$1" | tail -1 | grep -oE "[0-9.]+"
}

# Float comparison via awk, not bc — bc is not installed on this box, and
# `[ "$(… | bc -l)" = "1" ]` fails *open* when bc is missing, silently
# disabling the gate it implements.
lt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a < b) }'; }

# train.py takes FINAL_PATH from the environment so a run can be kept apart
# from the shipped model. Stages 4 and 5 hardcoded watermelon_final.zip, so
# setting FINAL_PATH used to make them evaluate a DIFFERENT, older model than
# the one just trained -- and report its score as this run's result.
FINAL_MODEL="${FINAL_PATH:-watermelon_final.zip}"

echo "=== [0/5] pre-flight ==="
python -u sanity_check.py || {
    echo "PIPELINE ABORTED: sanity checks failed. Fix the env before training."
    exit 1
}

echo
echo "=== [1/5] behavioural cloning ==="
python -u pretrain.py || { echo "PIPELINE FAILED: pretrain"; exit 1; }

echo
echo "=== [2/5] gate: is the BC policy actually any good? ==="
python -u evaluate.py watermelon_pretrained.zip 20 2>&1 | tee _bc_eval.txt
BC=$(score_of _bc_eval.txt)
echo "BC score: ${BC:-unknown} (gate: >= $MIN_BC_SCORE)"
if [ -z "$BC" ] || lt "$BC" "$MIN_BC_SCORE"; then
    echo
    echo "PIPELINE ABORTED: BC scored ${BC:-unknown}, below $MIN_BC_SCORE."
    echo "Check worst_class_recall in the epoch lines above — a value near 0.00"
    echo "means the policy collapsed onto a few columns. Not worth PPO time."
    exit 1
fi

echo
echo "=== [3/5] PPO training ==="
python -u train.py || { echo "PIPELINE FAILED: train"; exit 1; }

echo
echo "=== [4/5] final evaluation ==="
python -u evaluate.py "$FINAL_MODEL" 30 2>&1 | tee _final_eval.txt
FINAL=$(score_of _final_eval.txt)

echo
echo "=== [5/5] regression check ==="
echo "heuristic teacher: run 'python evaluate.py heuristic 20' to compare"
echo "BC:    ${BC:-?}"
echo "Final: ${FINAL:-?}"
if [ -n "$FINAL" ] && [ -n "$BC" ] && lt "$FINAL" "$BC"; then
    echo
    echo "WARNING: PPO made the policy WORSE than behavioural cloning."
    echo "watermelon_pretrained.zip is the better model. Do not export $FINAL_MODEL."
    echo "Suspect (in order): reward ordering, ent_coef on the resume branch,"
    echo "learning rate. Run sanity_check.py first."
else
    echo "PPO improved on BC."
fi

echo
echo "=== PIPELINE COMPLETE ==="

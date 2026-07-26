#!/usr/bin/env bash
# Snake training pipeline, with gates.
#
# The gates exist because the previous version ran BC -> eval -> PPO
# unconditionally. BC evaluated at 0.00 and it went on to spend hours training
# PPO on a policy that predicted the same action for every input. Every stage
# below now has to earn the next one.
#
# PYTHONIOENCODING=utf-8: torch/SB3 print glyphs cp1252 cannot encode.
# python -u: stdout is a pipe here, and buffering means no visible progress.
set -o pipefail
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8

# BC must reach this to be worth fine-tuning. The heuristic teacher scores
# ~50, so a BC clone worth keeping should be in the same postcode. Anything
# near zero means collapse, not imperfection.
MIN_BC_SCORE=25

score_of() {
    # Extracts "Average score: N" from evaluate.py output.
    grep -oE "Average score: [0-9.]+" "$1" | tail -1 | grep -oE "[0-9.]+"
}

# Float comparison via awk, not bc — bc is not installed on the Windows box,
# and `[ "$(... | bc -l)" = "1" ]` fails *open* when bc is missing, silently
# disabling the very gate it implements.
lt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a < b) }'; }

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
python -u evaluate.py snake_pretrained.zip 30 2>&1 | tee _bc_eval.txt
BC=$(score_of _bc_eval.txt)
echo "BC score: ${BC:-unknown} (gate: >= $MIN_BC_SCORE)"
if [ -z "$BC" ] || lt "$BC" "$MIN_BC_SCORE"; then
    echo
    echo "PIPELINE ABORTED: BC scored ${BC:-unknown}, below $MIN_BC_SCORE."
    echo "Check per-class recall in the epoch lines above — a class near 0.00"
    echo "means the policy collapsed onto one action. Not worth PPO time."
    exit 1
fi

echo
echo "=== [3/5] PPO training ==="
python -u train.py || { echo "PIPELINE FAILED: train"; exit 1; }

echo
echo "=== [4/5] final evaluation ==="
python -u evaluate.py snake_final.zip 50 2>&1 | tee _final_eval.txt
FINAL=$(score_of _final_eval.txt)

echo
echo "=== [5/5] regression check ==="
echo "BC:    ${BC:-?}"
echo "Final: ${FINAL:-?}"
if [ -n "$FINAL" ] && [ -n "$BC" ] && lt "$FINAL" "$BC"; then
    echo
    echo "WARNING: PPO made the policy WORSE than behavioural cloning."
    echo "snake_pretrained.zip is the better model. Do not export snake_final.zip."
    echo "Suspect (in order): reward ordering, ent_coef on the resume branch,"
    echo "learning rate. Run sanity_check.py first."
else
    echo "PPO improved on BC."
fi

echo
echo "=== PIPELINE COMPLETE ==="

#!/usr/bin/env bash
# Install a freshly trained model into the website — but only if it is better
# than the one currently shipped.
#
#     tools/install_model.sh snake
#     tools/install_model.sh watermelon
#     tools/install_model.sh snake --force     # install regardless of score
#
# The regression guard is the point. PPO has produced a WORSE model than the
# one already shipped several times on this project (Watermelon: BC 749.50 ->
# PPO 535.37), and without a check the install step would happily overwrite a
# good model with a bad one. The last shipped score is recorded in
# <game>/training/.shipped_score so the comparison survives across runs.
#
# PYTHONIOENCODING=utf-8: torch.onnx prints glyphs cp1252 cannot encode and
# would otherwise crash the export part-way through.
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8

GAME="${1:?usage: install_model.sh <snake|watermelon> [--force]}"
FORCE="${2:-}"

case "$GAME" in
    snake)      MODEL=snake_final.zip;      ONNX=snake_ai.onnx;      EPISODES=50 ;;
    watermelon) MODEL=watermelon_final.zip; ONNX=watermelon_ai.onnx; EPISODES=30 ;;
    *) echo "unknown game '$GAME'"; exit 1 ;;
esac

DIR="$GAME/training"
SHIPPED_FILE="$DIR/.shipped_score"

[ -f "$DIR/$MODEL" ] || { echo "no $DIR/$MODEL — nothing to install"; exit 1; }

lt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a < b) }'; }

echo "=== evaluating $MODEL ($EPISODES episodes) ==="
( cd "$DIR" && python -u evaluate.py "$MODEL" "$EPISODES" 2>&1 | grep -vE "Warning|warn" ) | tee /tmp/_eval_$GAME.txt
NEW=$(grep -oE "Average score: [0-9.]+" /tmp/_eval_$GAME.txt | tail -1 | grep -oE "[0-9.]+")
[ -n "$NEW" ] || { echo "could not read a score — aborting"; exit 1; }

OLD=$(cat "$SHIPPED_FILE" 2>/dev/null || echo "0")
echo
echo "new model:     $NEW"
echo "shipped model: $OLD"

if [ "$FORCE" != "--force" ] && lt "$NEW" "$OLD"; then
    echo
    echo "REFUSING TO INSTALL: $NEW is worse than the shipped $OLD."
    echo "The site keeps its current model. Pass --force to override."
    exit 2
fi

echo
echo "=== exporting to ONNX ==="
( cd "$DIR" && python -u export.py 2>&1 | tail -2 ) || { echo "export failed"; exit 1; }

cp "$DIR/$ONNX" "$GAME/$ONNX" || exit 1

# The critic ships as a separate file for the inspector's value readout, and it
# MUST travel with its policy. Leaving the old one behind would pair a new
# policy with the previous model's value head — plausible numbers from a
# network that never saw these weights.
CRITIC="${GAME}_critic.onnx"
if [ -f "$DIR/$CRITIC" ]; then
    cp "$DIR/$CRITIC" "$GAME/$CRITIC" || exit 1
    echo "  critic updated alongside the policy"
fi

echo
echo "=== embedding as base64 for file:// ==="
( cd "$GAME" && python -u embed_model.py ) || { echo "embed failed"; exit 1; }

echo
echo "=== verifying ==="
python - "$GAME" "$ONNX" <<'PYEOF' || exit 1
import base64, hashlib, pathlib, re, sys
import onnx

game, onnx_name = sys.argv[1], sys.argv[2]
root = pathlib.Path(game)

# The embedded copy must be byte-identical to the model we just exported.
js = (root / "model_data.js").read_text(encoding="utf-8")
m = re.search(r'const [A-Z_]+_MODEL_B64 = "([A-Za-z0-9+/=]+)";', js)
assert m, "no *_MODEL_B64 constant in model_data.js"
decoded = base64.b64decode(m.group(1))
raw = (root / onnx_name).read_bytes()
same = hashlib.sha256(decoded).hexdigest() == hashlib.sha256(raw).hexdigest()
print(f"  embed sha256 match : {same}")
assert same, "embedded base64 does not match the .onnx"

model = onnx.load(root / onnx_name)
onnx.checker.check_model(model)
shp = lambda t: [d.dim_param or d.dim_value for d in t.type.tensor_type.shape.dim]
inp, out = model.graph.input[0], model.graph.output[0]
print(f"  onnx               : {inp.name} {shp(inp)} -> {out.name} {shp(out)}")

# The JS encoder must still agree with the model's input width, or inference
# silently produces nonsense rather than failing.
expected = {"snake": 16 * 16 * 5 + 14, "watermelon": 22 * 30 * 4 + 12}[game]
actual = shp(inp)[1]
print(f"  encoder width      : js {expected} vs model {actual}")
assert expected == actual, "JS encoder and model input have drifted apart"

# And the OUTPUT width against the number of actions game.js offers. The input
# check above would happily pass a Watermelon model trained on 48 drop columns
# while the page still maps 24 — the observation is unchanged, so nothing
# complains, and the model's logits are silently reinterpreted as different
# placements. Keep these in step with the games' own tables; they are the same
# constants build_checkpoints.py checks.
expected_actions = {"snake": 3, "watermelon": 48, "tetris": 40}[game]
actual_actions = shp(out)[1]
print(f"  action count       : js {expected_actions} vs model {actual_actions}")
assert expected_actions == actual_actions, (
    f"game.js offers {expected_actions} actions but this model outputs "
    f"{actual_actions}. Update <game>/game.js in the SAME change that installs "
    f"this model, then update the table in install_model.sh.")
print("  all checks passed")
PYEOF

echo "$NEW" > "$SHIPPED_FILE"
echo
echo "INSTALLED $GAME at $NEW (was $OLD)"

# ── Publish to GitHub ────────────────────────────────────────────────────────
# A dated release, replaced in place as the day's cycles improve on each other.
#
# A release rather than a commit: these are tens of MB of binaries that change
# wholesale every cycle, so committing them would grow the repository history
# permanently for files that are never diffed. model_data.js is deliberately not
# attached — it is derived from the .onnx by embed_model.py, so shipping both
# would double the payload and invite the two drifting apart.
GH="/c/Program Files/GitHub CLI/gh.exe"
TAG="models-$(date +%Y-%m-%d)"
if [ -x "$GH" ]; then
    SNAKE_S=$(cat snake/training/.shipped_score 2>/dev/null || echo "?")
    WM_S=$(cat watermelon/training/.shipped_score 2>/dev/null || echo "?")
    if "$GH" release view "$TAG" --repo kohan1/humanvsai >/dev/null 2>&1; then
        "$GH" release upload "$TAG" "$GAME/$ONNX" --repo kohan1/humanvsai --clobber >/dev/null 2>&1 \
            && echo "uploaded $ONNX to release $TAG"
    else
        "$GH" release create "$TAG" "$GAME/$ONNX" --repo kohan1/humanvsai \
            --title "Models $TAG" \
            --notes "Snake $SNAKE_S / Watermelon $WM_S. ONNX only — regenerate model_data.js with embed_model.py after downloading." >/dev/null 2>&1 \
            && echo "created release $TAG with $ONNX"
    fi
    "$GH" release edit "$TAG" --repo kohan1/humanvsai \
        --title "Models $TAG: Snake $SNAKE_S, Watermelon $WM_S" >/dev/null 2>&1
else
    echo "(gh not found — skipped the GitHub upload)"
fi

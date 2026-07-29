#!/usr/bin/env bash
# Publish the playable site to the gh-pages branch.
#
#     tools/deploy_pages.sh
#
# WHY A SEPARATE BRANCH
# The site needs the model_data.js files (~85 MB) to work, but those are
# generated, change wholesale after every training cycle, and are gitignored on
# main for good reason. Force-pushing an orphan gh-pages branch each time keeps
# them off main's history entirely — the branch has exactly one commit and is
# replaced, not appended to.
#
# WHY THE .onnx RATHER THAN model_data.js
# model_data.js embeds the model as base64 in a <script> tag. That is the only
# thing that works under file://, but on a served site it is a render-blocking
# script in <head> — 45 MB for Snake, 30 MB for Watermelon — so the browser
# paints a blank white page until the whole thing has downloaded and parsed.
# That reads as "stuck on loading", and it bites every time this script runs,
# because a force-push gives every file a new blob and invalidates the cache.
#
# All three games now prefer fetch()ing the .onnx and fall back to the base64
# constant only when it is present. So we ship the .onnx and strip the
# model_data.js <script> tag from the deployed HTML: nothing blocks the render,
# the human board is playable while the model streams in, and the payload drops
# by a third (base64 inflation) — 85 MB total to 64 MB.
#
# The repo keeps model_data.js so opening game.html as a local file still works.
#
# Builds in a temp directory so the working tree is untouched — training writes
# into snake/training and watermelon/training while this runs.
#
#     tools/deploy_pages.sh --dry-run <dir>
#
# stages the site into <dir> and stops before pushing, so the exact bytes that
# would go live can be served and tested first. Worth using whenever the way
# the site loads its models changes — a broken loader is invisible on a warm
# cache and only shows up for real visitors.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
REPO="kohan1/humanvsai"

DRY_RUN=""
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    STAGE="${2:?usage: deploy_pages.sh --dry-run <dir>}"
    rm -rf "$STAGE"; mkdir -p "$STAGE"
else
    STAGE="$(mktemp -d)"
    trap 'rm -rf "$STAGE"' EXIT
fi

echo "staging the site in $STAGE"

# Entry pages, the shared mesh background, and the Inside page with its
# generated training data.
cp index.html select.html inside.html mesh.js backgrounds.js themes.css "$STAGE/"
cp inspector.css inspector.js results.js settings.js "$STAGE/"
mkdir -p "$STAGE/inside"
cp inside/data.js "$STAGE/inside/"

for game in snake tetris watermelon; do
    mkdir -p "$STAGE/$game"
    # Everything the browser actually requests. training/, tools/ and the .zip
    # checkpoints are deliberately excluded. model_data.js is excluded too —
    # see the note at the top; the .onnx is copied in below instead.
    for f in game.html game.js style.css image_data.js; do
        [ -f "$game/$f" ] && cp "$game/$f" "$STAGE/$game/"
    done
    for d in assets images lib; do
        [ -d "$game/$d" ] && cp -r "$game/$d" "$STAGE/$game/"
    done

    # Drop the model_data.js <script> tag. Without this the deployed page
    # requests a file that is not there; the fallback would still work, but it
    # would log a 404 on every load and invite someone to "fix" it by shipping
    # the base64 again.
    sed -i '/<script src="model_data\.js">/d' "$STAGE/$game/game.html"
done

# The models themselves, fetched at runtime by each game's loader. Tetris keeps
# its .onnx under training/; the other two sit beside their page.
cp snake/snake_ai.onnx           "$STAGE/snake/"
cp watermelon/watermelon_ai.onnx "$STAGE/watermelon/"
cp tetris/training/tetris_ai.onnx "$STAGE/tetris/"

# Critic models, for the inspector's value readout. Separate files, fetched
# only when someone opens the panel — folding the value head into the playing
# model measured 22.6 -> 45.3 MB on Watermelon, doubling what every visitor
# downloads for a panel that is closed by default.
#
# Tetris has none: the 1B-step checkpoint that produced its .onnx no longer
# exists, and exporting a critic from the surviving 260M checkpoint would show
# a different, weaker network's opinion beside the shipped policy. The page
# hides the readout when the fetch 404s.
for f in snake/snake_critic.onnx watermelon/watermelon_critic.onnx; do
    [ -f "$f" ] && cp "$f" "$STAGE/$(dirname "$f")/"
done

# A page that ships no model and no base64 would fail silently at runtime, so
# check here instead.
for f in snake/snake_ai.onnx tetris/tetris_ai.onnx watermelon/watermelon_ai.onnx; do
    [ -s "$STAGE/$f" ] || { echo "ABORT: $f missing or empty in the staged site"; exit 1; }
done
if grep -rq '<script src="model_data\.js"' "$STAGE"/*/game.html; then
    echo "ABORT: a deployed game.html still loads model_data.js as a script"
    exit 1
fi

# Pages runs Jekyll by default, which ignores files and folders beginning with
# an underscore and can mangle others. This turns that off.
touch "$STAGE/.nojekyll"

echo
echo "=== deploy contents ==="
( cd "$STAGE" && find . -type f | wc -l ) | xargs echo "files:"
du -sh "$STAGE" | awk '{print "size:  " $1}'
echo
echo "files over 1 MB (GitHub rejects anything over 100 MB):"
find "$STAGE" -type f -size +1M -printf "%6.1f MB  %P\n" 2>/dev/null | sort -rn || true

# Refuse to publish anything GitHub will reject outright.
if find "$STAGE" -type f -size +100M | grep -q .; then
    echo "ABORT: a file exceeds GitHub's 100 MB limit"
    exit 1
fi

if [ -n "$DRY_RUN" ]; then
    echo
    echo "dry run — nothing pushed. Serve and test it with:"
    echo "    python -m http.server 8322 --directory $STAGE"
    exit 0
fi

echo
echo "=== pushing to gh-pages ==="
cd "$STAGE"
git init -q
git checkout -q -b gh-pages
git add -A
git -c user.name="kohan1" -c user.email="309233267+kohan1@users.noreply.github.com" \
    commit -q -m "Deploy site $(date -u '+%Y-%m-%d %H:%M UTC')"
git remote add origin "https://github.com/$REPO.git"
git push -q --force origin gh-pages

echo "deployed to https://kohan1.github.io/humanvsai/"

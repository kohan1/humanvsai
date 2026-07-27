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
# WHY model_data.js RATHER THAN THE .onnx
# Only Tetris falls back to fetch()ing its .onnx; Snake and Watermelon require
# the base64 constant. Serving the raw .onnx would be ~19 MB smaller per visit
# but needs a code change in two games, and shipping a site whose AI silently
# fails is worse than shipping a larger one. Revisit by adding a fetch fallback
# to both, then deploying the .onnx instead.
#
# Builds in a temp directory so the working tree is untouched — training writes
# into snake/training and watermelon/training while this runs.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
REPO="kohan1/humanvsai"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "staging the site in $STAGE"

# Entry pages
cp index.html select.html "$STAGE/"

for game in snake tetris watermelon; do
    mkdir -p "$STAGE/$game"
    # Everything the browser actually requests. training/, tools/ and the .zip
    # checkpoints are deliberately excluded.
    for f in game.html game.js style.css model_data.js image_data.js; do
        [ -f "$game/$f" ] && cp "$game/$f" "$STAGE/$game/"
    done
    for d in assets images lib; do
        [ -d "$game/$d" ] && cp -r "$game/$d" "$STAGE/$game/"
    done
done

# Tetris keeps its .onnx under training/, and its loader falls back to
# fetch()ing it next to the page.
[ -f tetris/training/tetris_ai.onnx ] && cp tetris/training/tetris_ai.onnx "$STAGE/tetris/"

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

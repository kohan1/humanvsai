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
cp index.html select.html inside.html "$STAGE/"
# Site-wide JS/CSS lives in shared/ and is referenced as shared/x.js from the
# root pages and ../shared/x.js from the game pages, so the deployed tree has
# to keep that directory rather than flattening it.
mkdir -p "$STAGE/shared"
cp shared/*.js shared/*.css "$STAGE/shared/"
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

    # The checkpoint ladder — the earlier, weaker models the in-game switcher
    # offers. These ARE deployed, unlike the .zip checkpoints above, because the
    # browser fetches them by URL. They are only requested when a visitor picks
    # a rung, so they cost nothing on a normal load.
    [ -d "$game/checkpoints" ] && cp -r "$game/checkpoints" "$STAGE/$game/"

    # Belt and braces: the model_data.js <script> tag is no longer in the
    # source pages at all — shared/model-source.js injects it, and only under
    # file:// — so this matches nothing today. Kept because a re-added tag
    # would put a 30-46 MB render-blocking script back on every visit, and
    # that failure looks like "the site is slow" rather than like a mistake.
    sed -i '/<script src="model_data\.js">/d' "$STAGE/$game/game.html"
done

# Cache-busting. GitHub Pages lets browsers keep files for about ten minutes,
# so a visitor who arrived before a deploy could run the new game.js against a
# cached old shared/*.js — and these pages depend on the two staying in step.
# Every relative .js/.css reference in the staged HTML gets ?v=<commit>, making
# each deploy a fresh set of URLs. CDN URLs (anything with a ':') are untouched.
VER=$(git rev-parse --short HEAD 2>/dev/null || date +%s)
for html in "$STAGE"/*.html "$STAGE"/*/game.html; do
    sed -i -E "s#(src|href)=\"([^\":?]+\.(js|css))\"#\1=\"\2?v=$VER\"#g" "$html"
done
echo "  cache-busting     : ?v=$VER on every local script and stylesheet"

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
# model-source.js runs inference in this worker and falls back to the main
# thread without it — silently, so a missing file would only show up as the
# games getting janky again. Check for it instead.
[ -s "$STAGE/shared/ort-worker.js" ] || { echo "ABORT: shared/ort-worker.js missing in the staged site"; exit 1; }
if grep -rq '<script src="model_data\.js"' "$STAGE"/*/game.html; then
    echo "ABORT: a deployed game.html still loads model_data.js as a script"
    exit 1
fi

# Every rung the switcher offers must actually be served. The manifest is
# generated separately from the deploy, so the two can drift — and the failure
# mode is a button that downloads a 404 and leaves the player on the previous
# model with an error in the console nobody reads.
# python3 where it exists (macOS and most Linux ship no bare `python`),
# python on Windows, where python3 is often the Store stub.
PY="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python)"
case "$PY" in *WindowsApps*) PY="$(command -v python 2>/dev/null || echo python)" ;; esac
"$PY" - "$STAGE" <<'PYEOF' || exit 1
import json, pathlib, sys
stage = pathlib.Path(sys.argv[1])
txt = (stage / "shared" / "checkpoints.js").read_text(encoding="utf-8")
data = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
missing, n = [], 0
for game, spec in data.items():
    for rung in spec.get("rungs", []):
        n += 1
        p = stage / game / rung["file"]
        if not p.is_file() or p.stat().st_size == 0:
            missing.append(f"{game}/{rung['id']} -> {rung['file']}")
        elif p.stat().st_size != rung["bytes"]:
            missing.append(f"{game}/{rung['id']} size {p.stat().st_size} "
                           f"!= manifest {rung['bytes']}")
if missing:
    print("ABORT: checkpoint ladder does not match what is staged:")
    for m in missing:
        print("  " + m)
    # checkpoints.js is committed but <game>/checkpoints/*.onnx are gitignored
    # (large, regenerable), so a fresh clone has the manifest without the
    # models. That is the likely cause of this abort.
    print("\nRun:  python tools/build_checkpoints.py")
    print("(the .onnx rungs are gitignored — the manifest is not)")
    sys.exit(1)
print(f"  checkpoint ladder : {n} rungs, all present and the right size")
PYEOF

# Pages runs Jekyll by default, which ignores files and folders beginning with
# an underscore and can mangle others. This turns that off.
touch "$STAGE/.nojekyll"

echo
echo "=== deploy contents ==="
( cd "$STAGE" && find . -type f | wc -l ) | xargs echo "files:"
du -sh "$STAGE" | awk '{print "size:  " $1}'
echo
echo "files over 1 MB (GitHub rejects anything over 100 MB):"
# %s (size in bytes) then format in awk. NOT -printf "%6.1f MB" — find has no
# float directive, so %f was read as the basename with a width/precision of
# 6.1, which truncated every name to one character and printed "w MB", "t MB",
# "s MB" where the sizes should have been.
find "$STAGE" -type f -size +1M -printf "%s\t%P\n" 2>/dev/null \
    | sort -rn | awk -F'\t' '{printf "%7.1f MB  %s\n", $1 / 1048576, $2}' || true

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
PUSHED=$(git rev-parse HEAD)

echo "deployed to https://kohan1.github.io/humanvsai/"

# WAIT FOR THE PAGES DEPLOYMENT TO FINISH BEFORE RETURNING.
#
# Pushing to gh-pages starts a Pages deployment that takes ~90s. Pushing AGAIN
# while one is in flight cancels it — the build and report-build-status jobs
# still go green, and only the deploy job fails with "Deployment cancelled", so
# the script looks like it succeeded while the site silently keeps serving the
# previous build.
#
# That is exactly what happened on 2026-08-06: four deploys inside fifteen
# minutes produced four cancelled deployments, and the site served day-old
# content while every local check said the branch was correct. The API reports
# only "Page build failed" with no detail, so it cost a long time to find.
#
# Blocking here makes the failure mode impossible: the next deploy cannot start
# until this one is terminal.
# gh on PATH first (the Mac, Linux, or Windows with gh installed normally),
# then the Windows installer's default location, which Git Bash does not put
# on PATH. It used to look ONLY at the Windows path, so a deploy from the Mac
# printed "gh not found", skipped this wait, and left the exact cancel-by-
# redeploy trap described above wide open on the primary dev machine.
GH="$(command -v gh 2>/dev/null || true)"
[ -n "$GH" ] || GH="/c/Program Files/GitHub CLI/gh.exe"
if [ -x "$GH" ]; then
    printf "waiting for the Pages deployment"
    confirmed=
    for _ in $(seq 1 40); do
        sleep 15
        # Status AND commit. Straight after a push, builds/latest can still be
        # the PREVIOUS build, already "built", which used to end this wait
        # before the new deployment had even started.
        read -r status commit < <("$GH" api repos/"$REPO"/pages/builds/latest                                   --jq '"\(.status) \(.commit)"' 2>/dev/null || echo "")
        if [ -n "$PUSHED" ] && [ "$commit" != "$PUSHED" ]; then printf "."; continue; fi
        case "$status" in
            built)   echo " -> built"; confirmed=1; break ;;
            errored) echo " -> ERRORED"
                     echo "the branch was pushed but Pages refused to publish it."
                     echo "check: gh api repos/$REPO/pages/builds/latest"
                     exit 1 ;;
            *)       printf "." ;;
        esac
    done
    [ -n "$confirmed" ] || echo " -> still not confirmed after 10 minutes; check before deploying again."
else
    echo "(gh not found — cannot confirm the deployment; wait ~90s before deploying again)"
fi

# CLAUDE.md — human vs ai

Context file for Claude Code sessions on this project. **Read this before
making changes** — most of what follows is hard-won and several bugs listed
here were found and fixed once already. Don't reintroduce them.

> **Keep this file current.** When a session turns up something a future
> session would otherwise have to rediscover — a non-obvious bug and its root
> cause, a platform gotcha, a design decision and its reasoning, a measured
> benchmark — add it here as part of that work, not as a follow-up. Prefer
> *why* over *what*: the code already says what it does. This file is the
> handoff; if it's only in the chat, it's lost.

## Never deploy twice in quick succession — it cancels itself

Pushing to `gh-pages` starts a Pages deployment that takes about 90 seconds.
Pushing again while one is in flight CANCELS it, and the failure is close to
invisible:

- `tools/deploy_pages.sh` prints "deployed to ..." and exits 0.
- The `build` and `report-build-status` jobs both go green.
- Only the `deploy` job fails, with `Error: Deployment cancelled.`
- The site keeps serving the PREVIOUS build, so every page still works.
- `gh api repos/<repo>/pages/builds/latest` says only "Page build failed",
  with `duration: 0` and no further detail.

On 2026-08-06 four deploys inside fifteen minutes produced four cancelled
deployments. The branch content was verified correct the whole time — right
files, right sizes, `.nojekyll` present — which made it look like a GitHub-side
fault. It was not: it was self-inflicted, and the giveaway is that the
Actions run shows build succeeding and only deploy failing after ~12s.

`deploy_pages.sh` now blocks until the deployment reaches a terminal state and
exits non-zero if it errored, so a second deploy cannot start while one is
running. **If you ever need to check whether the live site is actually current,
compare `inside/data.js`'s `generated` timestamp against the local file** —
every page returning 200 proves nothing, because the old build still serves.

---

## After every training run: rebuild the Inside page

`inside.html` is the project's actual deliverable — it is the write-up, and its
whole claim is that it shows *every* run, including the failures. A run that is
not in it is a hole in that claim, and the page silently goes stale because
nothing about training touches it.

So this is not optional and not a follow-up. **Whenever a run finishes — or is
stopped, or is abandoned — do all four of these before moving on:**

1. **Archive the log.** `start_training.ps1` always writes
   `remote_training.log`, whatever the run is. Copy it to
   `<game>/training/logs/train_<name>.log`. The builder deliberately skips
   `remote_training.log` itself, because once a run is archived the two are the
   same curve and the page listed it twice.
2. **Add a `RUN_NOTES` entry** in `tools/build_training_data.py`, keyed
   `"<game>/train_<name>.log"`. Keyed by game AND filename: both Snake and
   Watermelon have had a `train_100m.log`, and keying on the filename alone
   attached Snake's score to Watermelon's run.
3. **Record what actually happened**, including `outcome="rejected"` and why.
   Only put a number in `evalScore` that was really measured with
   `evaluate.py` — a run with no entry is emitted with its real curve and no
   outcome label, which is the honest default. Never guess one.
4. **Rebuild and deploy:**

   ```bash
   python tools/build_training_data.py
   bash tools/deploy_pages.sh
   ```

An in-progress run can go on the page too — snapshot the log to `logs/` the
same way and mark it `outcome="unknown"`. Re-snapshot when it finishes.

If the run produced a model worth shipping, `tools/install_model.sh` is a
separate step with its own regression gate. Rebuilding the page is not
optional either way: a rejected run is exactly the kind the page exists to
show.

---

## Keep the best model, not the last one

`train.py` used to save only the FINAL model. A run that peaked mid-way and
drifted down threw the peak away — and that is exactly what happened three
times on Watermelon (614.40, 872.00, 893.03 all rejected by the install guard
for ending worse than what was already shipped).

`BestScoreCallback` in `watermelon/training/train.py` now evaluates every
`EVAL_FREQ` steps and keeps the best in `watermelon_best.zip`. Two details
that matter:

- **Fixed seeds.** It always evaluates seeds `0..n-1`. Scores swing from 433
  to 1426 between seeds, so without fixing them "best" would mostly select a
  lucky draw rather than a better policy.
- **It evaluates at step 0**, so the STARTING model sets the bar. Otherwise a
  resume that degrades early would save the first thing it measured and call
  it best.

**Its numbers are not comparable to `evaluate.py`'s.** The callback uses 15
seeds by default, `install_model.sh` uses 30. The same model measured 964.80
on seeds 0-14 and 936.70 on seeds 0-29. Use the callback's score only to
compare checkpoints *within* a run; always re-measure with `evaluate.py`
before deciding whether to ship.

This safety net is also what makes a more aggressive trust region reasonable:
a run that overshoots and regresses no longer costs anything.

---

## BestScoreCallback's numbers are IN-SAMPLE — never compare them to a
## 60-seed figure

Raising `EVAL_EPISODES` to 40 fixed the noise problem and introduced a subtler
one. The callback evaluates seeds 0-39 and saves whichever checkpoint scores
best **on those exact seeds**. Across 150+ evaluations that is a search for the
checkpoint most flattered by one seed subset, so its number is an in-sample
maximum, not an estimate of general performance.

Measured on the merge-map run, the same file:

    callback (seeds 0-39)   107.8 drops   1045 score
    60 fixed seeds          103.2 drops    979 score

About four drops of optimism. This produced two false conclusions in one
session: a "milestone" that had supposedly beaten the shipped model, and a
"+3.1 standard error" advantage over the previous run that reversed entirely
once both were measured on 60 seeds (104.6 vs 106.8).

Rules that follow:

- **Never compare a callback figure to a 60-seed figure.** They are different
  estimators and the callback's is biased upward.
- **Comparing two callback figures is also unsafe.** Both are in-sample and can
  be overfitted to seeds 0-39 by different amounts, which is exactly what
  happened here.
- **Any claim about which model is better needs an independent evaluation** on
  a seed set the callback never selected on. `install_model.sh` does this, and
  it is the only reason none of these ever shipped.

---

## BestScoreCallback's 15 episodes cannot rank Watermelon models

Measured over 136 evaluations across 33.75M steps of the 100M run:

    spread   751.5 - 1130.8   (379 points)
    stdev    70.8             (7.5% of the mean)

At that variance a 15-episode evaluation cannot distinguish two similar
policies, and `BestScoreCallback` will happily save a *lucky sample* as a new
best. It did exactly that here: it recorded "new best 1130.80" against a
baseline of 1123.47 — a gap of 7.3 points, about one tenth of a standard
deviation — and the model it saved then measured **1015.80 over 30 episodes**,
i.e. WORSE than the shipped 1032.43 it had been resumed from.

So a "new best" line in a Watermelon log means nothing on its own. Two
consequences:

1. **Never promote a model on the callback's number.** `install_model.sh` runs
   its own 30-episode evaluation and refuses a regression, and that check is
   the only reason the noisy selection has never actually shipped a worse
   model. Keep it.
2. **Raise `EVAL_EPISODES` before trusting mid-run selection.** 15 was chosen
   for cost (~1500 drops per check against 250k of training). The cost was the
   right thing to optimise; the episode count was not checked against the
   distribution's spread.

Watermelon's scores are skewed and long-tailed — a single lucky 1585-point game
moves a 15-episode mean by 40 points. Snake and Tetris were not re-measured
this way, but Tetris is *more* skewed still (mean 119k against a median 57k),
so treat its checkpoint numbers with the same suspicion.

---

## Resume from the run-end model, ship the best one

These are two different jobs and the same file cannot do both.

`BestScoreCallback` picks the checkpoint that scored highest on fifteen fixed
seeds. Part of that is real skill and part is selection noise — it is, by
construction, the luckiest point on a noisy curve. That is exactly what you
want to *ship*, and a bad place to *resume* from.

Measured, on Watermelon:

| resumed from | result |
|---|---|
| 936.70's run-end model | climbed to 1123.47 over 20M steps |
| 1123.47 best checkpoint | **0 of 39 checks beat it over 39M steps** |

The second run dropped ~20% within 4M steps and never recovered — first-half
mean 922, second-half mean 926, dead flat. Twelve hours of GPU for nothing.
Regression to the mean is the obvious explanation with hindsight: perturbing a
point selected for being unusually good moves it back toward ordinary.

**So:** keep `<game>_best.zip` for shipping, and set `<game>_final.zip` to the
run-end model before resuming.

---

## Potential-based shaping, not bonuses

Watermelon's reward was redesigned on 2026-07-29. The shaping is
`gamma * phi(next) - phi(now)` (Ng, Harada & Russell 1999), which provably
leaves the optimal policy unchanged — it can only change how fast the agent
gets there. A plain bonus does not have that property, and this project has
already lost runs to an entropy bonus quietly destroying a good policy.

`phi` rewards a low stack, a level surface, and big fruit kept at the bottom.

Two invariants, both now asserted in `sanity_check.py`:

- **phi(terminal) must be 0.** A non-zero terminal potential is a disguised
  bonus for dying in a particular position, and the guarantee is void.
- **The shaping must telescope to about zero over an episode.** Measured -0.71
  against a merge component of +83.45. If that number drifts, either a
  terminal state has a potential or `phi` is not reset between episodes.

Why it was worth doing at all: **49% of drops produce no merge**, so under the
old reward half of every game's decisions returned nothing. Those steps now
carry a signal spanning -0.34 to +0.03, which is the difference between
ranking a placement and guessing.

A potential-based reward change does NOT require retraining from scratch, unlike
an observation change: returns shift by under 1%, so the critic stays valid and
a resume is safe.

---

## Safe shaping cannot raise a ceiling — that is the point of it

Watermelon's reward was rebuilt on 2026-07-29 with potential-based shaping and
it DID NOT PAY OFF. Measured over 30 games from comparable starting points:

| reward | start | result | gain |
|---|---|---|---|
| old | 936.70's model | **1032.43** | +95.73 |
| redesigned | 959.93's model | 1014.47 | +54.54 |

The new reward gained less, from a higher start. The shaping worked exactly as
designed — the telescoping check passed, no early-stopping problem, 41 stops in
489 iterations — and the result was still worse.

The reason is the property it was chosen for. Potential-based shaping is
provably unable to change which policy is optimal; it can only change how fast
that policy is found. So it can never raise the ceiling, only the speed of
approach — and with 20M steps there was no speed problem to solve. Choosing it
for safety meant choosing something that could not deliver what was wanted.

**If the goal is a better ceiling, change the observation, the architecture or
the action space. If the goal is faster convergence to the ceiling you already
have, shape the reward.** Those are different problems and this run confirmed
they need different tools.

Watermelon's real limit was neither the reward nor, as this section previously
guessed, the 22x30 observation grid. It was the size of the fruit. See the next
section.

---

## Watermelon's ceiling was geometric, and no reward could have found it

**Diagnosed 2026-08-08, after five training runs failed to beat 108 drops.**

Four reward designs and two observations all landed between 104 and 108 drops.
That band was not a coincidence and not a tuning problem — it was a wall none of
those levers touched, because the wall was in the physics constants.

Area leaves the well exactly once: when two MAX_TIER fruit merge and vanish
(`_resolve_merges` declines to spawn a replacement, and game.js does the same
with `if (tier === MAX_TIER) return;`). With the old ladder ending at 290px,
that state was **unreachable in any legal position**:

- side by side they need 580px; the well is 448 wide
- stacked they need 580px; the well is 484 tall, loss line to floor
- diagonally, the walls cap their horizontal separation at 158px, which forces
  their centres 243px apart vertically and puts the upper fruit's top edge at
  y=-21 — 136px past the loss line at y=115

Force two into contact in a test and the merge handler fires correctly. That is
what made this so hard to see: **the code was right and the situation was
impossible.** Nothing errors, nothing warns, and every reward function is
optimising honestly against a game that cannot be won.

Two corollaries that also went unnoticed for months:

- **A watermelon was a tombstone.** Two tier-9 fruit *could* reach each other,
  so tier 10 was produced and then could never be consumed: 66,052px of dead
  area, 30% of the well, permanent. Making the game's namesake fruit was the
  worst thing that could happen to you.
- **The working set did not fit.** One fruit of each tier — the minimum
  inventory for a cascade, since every tier waits for a partner — came to 98%
  of the well.

The fix was the diameter ladder, not the agent: a geometric `[24, 30, 37, 45,
56, 69, 86, 106, 131, 162, 200]`. A top pair now needs 400 of 448px, the
one-of-each inventory drops to 42%, and every merge shrinks the area it
occupies (0.74-0.78x) where the old ladder *grew* it at the bottom — two 30px
fruit became a 46px fruit covering 1.18x the area, so the early merges the
policy was rewarded for were actively filling the well.

**The unchanged heuristic teacher went from 85.6 drops to 375.** No training
involved. That is the size of the thing five runs of PPO were being asked to
climb over.

### What to take from this

**Before tuning an agent that has plateaued, check whether the goal is
reachable at all.** Write the arithmetic down: what is the sink, what rate does
it need to run at, and does the state that triggers it physically exist? A
reward can only rank policies that the environment permits. This cost roughly a
week of compute across five runs, and the check that would have caught it takes
a few minutes with a calculator.

Guards now in place, because both failures were silent:

- `watermelon/training/test_top_tier_sink.py` — asserts the top-tier pair
  actually merges and that area leaves the well. Run it before training.
- `tools/check_watermelon_sync.py` — compares DIAMETERS, POINTS, canvas and
  encoder constants between `watermelon_env.py` and `game.js`.
  `install_model.sh` cannot catch a diameter change, because changing fruit
  sizes does not change any array's shape: every assertion passes, the model
  loads, the page renders, and the policy is quietly playing a different game
  from the one it trained on.

---

## Before you start a training run

Snake v2 burned several hours across three separate failures, and **every one
was detectable in under a minute**. The rules below exist because of that; they
cost seconds and save hours.

1. **Run `sanity_check.py` first.** It verifies the *problem definition* rather
   than the optimiser: observation shape, flood fill not starting inside its own
   blocked set, and — most importantly — that playing well out-earns dying
   instantly. A reward that ranks death above competence cannot be rescued by
   any hyperparameter.
2. **Change one thing at a time.** v2 changed observation, architecture, reward,
   entropy, LR schedule and BC learning rate simultaneously. When it broke there
   was no way to attribute the failure, so each fix was a guess.
3. **Never trust an aggregate metric alone.** `train_acc=0.787` looked healthy
   while the policy predicted one constant action; 0.787 *was* the class prior.
   Check per-class recall.
4. **Gate every stage on the previous one.** `run_v2_pipeline.sh` aborts if BC
   scores below `MIN_BC_SCORE`, and warns loudly if PPO ends up worse than BC.
   The old version ran BC → PPO unconditionally and spent hours fine-tuning a
   policy that had already evaluated at 0.00.
5. **Compare against the teacher, not against random.** `evaluate.py heuristic`
   gives the number that matters. On Snake v1, 20M steps of PPO produced a
   policy still 28% worse than its own teacher and nothing in the training
   curves said so.
6. **Float comparisons in shell gates use `awk`, not `bc`** — `bc` is not
   installed here, and `[ "$(… | bc -l)" = "1" ]` fails *open*, silently
   disabling the gate it implements.
7. **Give the run its own checkpoint subdirectory.** `CheckpointCallback` names
   its saves `<game>_ckpt_<steps>_steps.zip`, which collides across runs: the
   100M Watermelon attempt and the reward-v2 run both wrote into
   `watermelon/training/checkpoints/` and their step ranges overlapped, so the
   later run silently overwrote the earlier one's files. 156 files that looked
   like a single 0.25M-39M series were two runs, and only separable at all
   because they happened on different days. Point `save_path` at a per-run
   directory before starting.

---

## Project overview

"human vs ai" — a multi-game portfolio site (a Digital Technologies school
assignment) where a human plays browser games side-by-side against a trained
RL opponent. Design language: monochrome, Georgia serif display type,
dark background (`#080808`), animated triangle-mesh canvas background on
`index.html` / `select.html`.

| Game | Algorithm | Status |
|---|---|---|
| **Tetris** | MaskablePPO, 238-float obs, 40 discrete placements | Trained, playable, persists across tab close |
| **Snake** | PPO (BC warm start → fine-tune), 773-float obs, `Discrete(3)` | Trained (37.50 avg), live in browser |
| **Watermelon** | none yet — AI board is a scaffold | Playable human side, ported from a browser-extension build |

Hardware: a Mac laptop (primary dev machine) + a Windows PC with an RTX 4060 Ti
(training). Synced via **Git** (code) + **Syncthing** (large model
binaries, deliberately out of Git).

**The site runs from `file://` by design** — double-click `index.html`, no
local server. This one constraint causes most of the non-obvious bugs below.

---

## Folder structure

```
humanvsai/                     ← project root
├── CLAUDE.md                  ← this file
├── index.html                 ← landing
├── select.html                ← game picker
├── inside.html                ← how the opponents were trained
├── shared/                    ← every site-wide asset, used by ALL pages
│   ├── themes.css             ← the six palettes + shared chrome
│   ├── mesh.js / backgrounds.js   ← the cursor-reactive backgrounds
│   ├── settings.js            ← difficulty, theme, persisted choices
│   ├── inspector.js / .css    ← the "what it sees" panel
│   ├── checkpoint_switcher.js / checkpoints.css   ← the model-version control
│   ├── checkpoints.js         ← GENERATED ladder manifest (build_checkpoints.py)
│   └── results.js             ← head-to-head record
├── inside/                    ← GENERATED data for inside.html
│   └── data.js                ← built by tools/build_training_data.py
├── tools/                     ← automation, not part of the site
│   ├── start_training.ps1     ← launch a run detached (survives SSH logout)
│   ├── deploy_pages.sh        ← build + publish to gh-pages
│   ├── install_model.sh       ← evaluate, gate, export, embed, release
│   ├── build_checkpoints.py   ← the switcher's ladders
│   ├── build_training_data.py ← inside/data.js
│   ├── check_training.sh      ← OK / STALLED / DEAD / FINISHED / IDLE
│   └── probe_checkpoints.py   ← which archived .zip files still load
├── tetris/
│   ├── game.html / game.js / style.css / assets/
│   ├── checkpoints/           ← switcher rungs, .onnx (gitignored)
│   └── training/              ← tetris_env.py, train.py, export.py, backups/
├── snake/
│   ├── game.html / game.js / style.css / images/
│   ├── embed_model.py / model_data.js / snake_ai.onnx (gitignored)
│   ├── checkpoints/           ← switcher rungs, .onnx (gitignored)
│   └── training/              ← snake_env.py, heuristic.py, pretrain.py, train.py,
│                                evaluate.py, export.py, archive/, logs/
└── watermelon/
    ├── game.html / game.js / style.css
    ├── embed_assets.py / image_data.js (generated)
    ├── assets/                ← fruit + cloud PNGs
    ├── lib/                   ← p5.min.js, planck.min.js, physics.min.js
    ├── checkpoints/           ← switcher rungs, .onnx (gitignored)
    └── training/              ← watermelon_env.py, train.py, export.py,
                                 archive/models/, logs/

PATHS ARE LOAD-BEARING. The root pages reference shared/x.js and the game pages
reference ../shared/x.js, and deploy_pages.sh reproduces shared/ in the built
site rather than flattening it. The three page filenames are public URLs and
must not move.
```

`.gitignore` should exclude: `models/`, `checkpoints/`, `tb_logs/`, `*.onnx`,
`model_data.js`, `image_data.js`, `__pycache__/`, `.DS_Store`.

**`model_data.js` and `image_data.js` are generated and machine-local** — after
pulling on a new machine, re-run `embed_model.py` (snake/tetris) and
`embed_assets.py` (watermelon).

**`models/` was deleted (2026-07-26) as outdated** — it held the old Tetris SB3
checkpoints. Consequence to know about:

- `tetris/training/export.py` still defaults to `models/tetris_ppo_final.zip`
  and `models/best_model.zip`. Those paths now dangle, so **Tetris cannot be
  re-exported** until a checkpoint is restored or retrained. It degrades
  gracefully (it guards with `os.path.exists("models")`) rather than crashing.
- The game itself is unaffected — `tetris/training/tetris_ai.onnx` and the
  generated `model_data.js` are what the browser loads, and both survive.
- This also blocks the 4-output re-export the neural-net visualisation needs;
  that work now requires retraining Tetris first.

---

## The `file://` constraint — the single biggest source of bugs

Browsers treat every `file://` document as its own opaque origin. Anything
that goes through `fetch()` or a CORS check therefore **fails silently**.
Three separate bugs in this project trace back to this:

1. **ONNX models.** `ort.InferenceSession.create("model.onnx")` calls `fetch()`
   internally → blocked. **Fix:** `embed_model.py` writes the model as base64
   into `model_data.js`, loaded as a plain (non-deferred) `<script>` before
   `game.js`, then `atob()`-decoded to a `Uint8Array` and passed directly to
   `InferenceSession.create()`. One constant per game
   (`TETRIS_MODEL_B64`, `SNAKE_MODEL_B64`).

2. **Watermelon images.** p5's `loadImage()` sets `crossOrigin="Anonymous"` on
   every URL that is *not* already a `data:` URI → blocked under `file://`.
   The physics still ran and the score still climbed, so the game looked alive
   while drawing nothing. **Fix:** `embed_assets.py` → `image_data.js`.

3. **Script ordering.** The `*_data.js` files must load **non-deferred, before**
   `game.js`, or the global isn't defined when the game runs.

**Symptom to recognise:** logic works, nothing renders. Suspect asset loading,
not rendering.

---

## Tetris

- Observation: **238 floats** — 180 board + 10 column heights + 7 current-piece
  one-hot + **35 next-5-pieces one-hot** + 4 scalars (holes / bumpiness /
  aggregate height / max height) + 1 combo + 1 well depth.
- Arena 10 × 18. Action space: 40 discrete placements.
- **AI drop animation** (`AIPlayer.applyMove`): two-phase tween instead of
  teleporting — 130ms ease-out horizontal shift, then 170ms ease-in drop
  (accelerating, mimicking gravity). Total 300ms, matched to
  `AI_THINK_INTERVAL`. Rotation is *not* tweened — swapped in instantly, since
  tweening a rotating tetromino reads as confusing. Tuning knobs: `shiftDur` /
  `dropDur`.
- **Persistence** (localStorage): high score and full board state for *both*
  boards, under separate keys (`tetris.human.*`, `tetris.ai.*`). State is plain
  data (`arena`, `bag`, `player`) so it round-trips through JSON directly.
  - Saves on `beforeunload` **and** `visibilitychange` — the latter is the
    reliable one when a tab is actually closed.
  - High score is written the moment it increases, so a crash can't lose it.
  - Finished boards are not saved (never reopen onto a game-over screen), and a
    saved board whose arena dimensions don't match is rejected, not restored.
  - `suppressSave` guards the restart buttons: they reload the page, which
    fires `beforeunload`, which would otherwise write the abandoned game
    straight back after `clearSavedGames()` wiped it.

## Snake

- Board 16 × 16, rendered 480 × 480 (30px cells).
- Action space `Discrete(3)` — turn left / straight / turn right, relative to
  heading. Every action is always legal, so **plain PPO, not MaskablePPO**.

### Observation v2 — 1294 floats

v1 was 773 floats (a 16×16×3 grid + direction + length) into a flat MLP. It
plateaued at 37.50 while its own teacher scored 51.96, because the teacher
decides on flood-fill reachable space and the observation contained nothing
from which a flat MLP could practically compute connectivity. v2 hands the
agent the teacher's actual signal:

```
grid   16*16*5, row-major (y, x, c)
  0 body excluding head   1 head   2 food
  3 tail (vacates next move, so entering it is safe)
  4 reachable (flood fill from head)
scalars (14)
  4  direction one-hot
  1  normalised length
  3  "this move kills me" flag, per relative action (left, straight, right)
  3  reachable free space after that move, board fraction
  2  signed food delta (dx, dy) / tile_count
  1  normalised steps since food
```

Layout stays **flat** so ONNX export and the JS mirror stay simple;
`policy_config.SnakeGridExtractor` reshapes the grid part to `(5, 16, 16)`.

### Reward v2

`+1` food, `-1` death (hard terminal), and:
- **Distance shaping fades with length** (`1 - len/40`, floored at 0). Beelining
  at food is right when short and is exactly what traps you when long.
- **Graded trap penalty** — when reachable space from the head drops below body
  length, subtract up to `0.1` proportionally. Directly analogous to Tetris's
  per-hole penalty: punish the structural error, not the death it causes later.
- Still **no flat per-step penalty** — on Tetris that taught the agent to end
  episodes early.

### Architecture and hyperparameters

- **CNN extractor** (`policy_config.py`, shared by `pretrain.py` and
  `train.py` so they cannot drift). Three 3×3 convs at `padding=1`, no pooling
  — at 16×16 there's no spatial redundancy worth discarding, and the exact cell
  a wall sits in is the whole point. Scalars bypass the conv and are
  concatenated to its output. ~9.1M params.
- **`ent_coef = 0.01`.** v1 ran at SB3's default `0.0` which, with
  `lr=5e-5` + `target_kl=0.03`, pinned the policy so tightly it could not
  explore out of its starting basin. Tetris uses `0.02` over 40 actions; Snake
  has 3 (max entropy ln 3 = 1.1 vs ln 40 = 3.7), hence the smaller value.
- **Linear LR anneal 3e-4 → 1e-5** on fresh runs (Tetris does the same).
  Resumed runs keep the fixed small `5e-5` + `target_kl`, because that phase is
  protecting weights, not exploring. **`target_kl` is not set on fresh runs** —
  it exists to stop a good policy drifting; from step zero it just throttles.
- **Env cost:** the observation runs up to 5 flood fills per step, so the env
  is the throughput bottleneck (~480 µs/step single-env). Flood fill uses flat
  integer cell indices into a reusable `bytearray`, not tuple-keyed sets —
  that alone was a 1.55× speedup (746 → 482 µs/step).

### Cross-checking the JS encoder

`snake/game.js` `buildObservation()` must mirror `_get_obs()` exactly. Because
there is no JS runtime on this box, the check is: dump Python observations for
a spread of states to JSON, then re-compute them with an ES5 transliteration
run under `cscript //Nologo`, and compare elementwise. Last run: **23 states ×
1294 values, 0 mismatches.** Redo this whenever either side changes — flood
fill runs the component to completion precisely so the result is independent
of traversal order and the two implementations can agree exactly.

`runAiInference()` also logs a console error if the encoder length ever stops
matching `OBS_SIZE`, since the alternative is silent garbage.
- Both snakes render green (`rgb(50, 255, 50)`); they're on separate boards.
- **Movement timing:** grid alignment via an exact frame counter
  (`frameCount % STEPS_PER_CELL === 0`), never float rounding. `SPEED` must
  evenly divide `scl` (currently `SPEED=3`, `scl=30` → `STEPS_PER_CELL=10`).

## Watermelon

Ported from a browser-extension build. Two structural changes of substance:

- **Instance mode.** The original ran as a global p5 sketch, which allows
  exactly one physics world per page. p5play attaches `world` / `Sprite` /
  `Group` / `allSprites` / `kb` to the *sketch instance*, so running each board
  as its own `new p5(sketch, container)` gives two independent worlds.
  **Every p5 call must go through `p.`** — dropping the prefix silently binds
  to whichever board initialised last.
- **Persistence.** Extension-synced storage → `localStorage`, gated on
  `cfg.persist` so only the human board writes and the AI scaffold can never
  overwrite a real score.
- Game over resets the board **in place**; the original reloaded the page,
  which would take the other board down with it.
- The red stack-limit line is drawn with `p.line()` in `draw()`, **not** as a
  sprite — see gotcha #10 below.
- **Wiring an AI later:** `window.watermelonBoards.ai.setPolicy(fn)`. `fn`
  receives a state snapshot (`buildState()`) and returns a 0–1 fraction of
  board width to aim at, or `null` to hold. Nothing else needs to change.

### Watermelon training pipeline

`watermelon/training/` mirrors `snake/training/`. **Physics is pymunk
(Chipmunk2D), while the browser runs planck.js (Box2D) via p5play** — different
engines, so an identical sequence of drops will *not* produce identical stacks.
That is a deliberate trade: a faithful Box2D binding means compiling pybox2d
from source on Windows/Python 3.14, and planck diverges from Box2D anyway. What
transfers is the decision problem — same board, fruit sizes, merge rule and
failure condition. A policy that learns "merge same tiers, keep the stack low"
transfers; one exploiting exact bounce trajectories will not. **If browser
scores differ wildly from `evaluate.py`, suspect physics divergence before the
policy, and check the settle thresholds first** — they decide when a drop is
"done".

- **Action space** `Discrete(24)` — which column to drop from. One env step is
  a whole drop *plus* the settle, so an episode is a sequence of placements,
  not of frames. Browser mapping: `fraction = (action + 0.5) / 24`.
- **Observation** 1332 floats, flat: `22×30×2` grid (occupancy, tier/MAX_TIER)
  row-major, then 12 scalars (held tier one-hot ×5, next tier one-hot ×5, stack
  height fraction, fruit count fraction).
- **Reward** merge points × 0.1, −1 on loss, and a graded height penalty as the
  stack rises past the loss line — the same "punish the structural error"
  reasoning as Tetris's per-hole penalty.
- **Speed** ~650 drops/s single-env, so RL is practical.
- Measured baselines: random ≈ 420, `heuristic.py` ≈ **780–840**. Always
  compare a trained model against the heuristic, not just against random.

---

## Bugs already found and fixed — don't reintroduce these

**Rendering**

00. **Fruit sprites are drawn at the image's natural size, not the collider's.**
    p5play sizes the physics circle from `sprite.diameter` and draws the
    animation at the PNG's own dimensions; nothing converts between them. The
    assets were authored against the original ladder and their widths still
    track it almost exactly (watermelon.png 291px for a 290px fruit,
    apple.png 128 for 125), so the two agreed by coincidence and no code ever
    had to reconcile them.

    Compressing the ladder on 2026-08-08 broke the coincidence: colliders
    shrank, images did not, and every fruit rendered 1.4-1.9x larger than the
    circle it occupied. Fruit *looked* like they were sitting inside each other
    while the physics was completely correct — querying the board reported
    exact diameters and zero overlaps, which is what made it read as a physics
    bug when it was purely cosmetic.

    Fixed by resizing each `p5.Image` to `DIAMETERS[tier]` in `setup()`,
    preserving aspect. **Not** `sprite.scale`: that setter calls
    `_resizeColliders()` and would move the physics to match the picture,
    which is backwards. `SpriteAnimation.scale` would also work — it is applied
    in the draw path only — but resizing the image fixes every draw site at
    once (balls, cloud ball, preview) instead of each assignment of `.img`.

    **If DIAMETERS changes again, the images must be rescaled with it.** The
    coupling is invisible: nothing errors and the game remains playable.

**Observation / training**

0. **Watermelon trained under a different merge rule than it plays under.**
   `watermelon_env.py` blocked a top-tier merge outright (`ta >= MAX_TIER:
   return`), leaving both fruit on the board permanently. `game.js` removes
   BOTH and only then skips creating the replacement (`a.remove(); b.remove();
   if (tier === MAX_TIER) return;`) — so in the browser two watermelons
   annihilate and free the space.

   The AI therefore learned in a world where the top tier was permanent dead
   area — one is 290px across in a 448x484 box, so two cannot coexist — and
   then played a browser where it clears. Found on 2026-08-05 while looking for
   ways to make endless play possible; it had been there the whole time, and
   the env's own docstring asserts the opposite ("same board, same fruit sizes,
   same merge rule").

   Fixed in the ENV, not the browser: the browser was right. **Any claim that
   the two sides agree should be checked against the code, not the comment** —
   this one was documented as true for months while being false.

1. **JS encoder drifted from the Python env** — Tetris `buildObs()` was still
   the old 210-float single-lookahead encoder while the model expected 238
   (5-piece lookahead). `238 − 210 = 28 = 35 − 7`, which confirmed it.
   **Rule: the JS encoder must mirror the Python `_get_obs()` field-for-field,
   in the same order. If you change one, change the other in the same commit.**
   Verified for Snake: both produce 773, same row-major order.
2. **`pretrain.py`'s KL anchor pulled the policy toward its own random init.**
   "Low LR + KL penalty" protects an *already-competent* policy; applied to a
   cold-start BC pass it fights the entire point. With an LR ~200× too small
   the net barely learned (scored 0.09 — random). Fixed: no KL anchor, normal
   supervised LR (`1e-3`).
3. **`train.py` used PPO's default LR (3e-4) on top of a good BC policy** and
   bulldozed 21.59 → 7.37 within 3M steps. Same failure class as #2 but in the
   phase where "protect existing weights" *does* belong. Fixed:
   `learning_rate=5e-5` + `target_kl=0.03` on the branch that loads
   `snake_pretrained.zip`.
4. **`model.verbose` silently inherited as `0`** — `pretrain.py` sets
   `verbose=0` and that is serialised into the `.zip`. Training looked frozen
   but was running fine, just silent. Fixed by forcing `model.verbose = 1`
   after `PPO.load()`.
5. **`train.py` resumed from `snake_pretrained.zip`, not `snake_final.zip`** —
   re-earning finished work every run. Now prefers a finished RL model
   (`RESUME_PATH`) and falls back to the BC policy.

14. **Behavioural cloning collapsed onto the majority class.** Snake's teacher
    plays ~79% STRAIGHT / 9% left / 12% right. Unweighted cross-entropy found
    that predicting STRAIGHT for *every* input was the cheapest way to reduce
    loss: it reported `train_acc=0.787` — exactly the class prior — while
    scoring **0.00**, because a snake that never turns walks into a wall.
    Fixed with inverse-frequency class weights (normalised to mean 1 so the
    effective LR is unchanged) plus dropping the BC LR from 1e-3 to 3e-4, which
    took per-class recall from `[0.00, 1.00, 0.00]` to `[1.00, 0.99, 1.00]`
    within 6 epochs.
    **The real lesson is the metric.** Aggregate accuracy cannot distinguish a
    working classifier from a constant one. Both `pretrain.py` scripts now log
    per-class recall; if any class the teacher actually uses sits near 0.00,
    the run is broken no matter how healthy the headline number looks.

15. **The v2 reward was inverted — the single most expensive bug so far.**
    `step()` computed free space as `_free_space(self.body[0], self.body[:-1])`,
    but `body[:-1]` *includes the head*, so the flood fill started inside its
    own blocked set and returned **0 every time**. `free < len(body)` was
    therefore always true, turning a graded trap penalty into a **flat −0.1 per
    step**. Measured on the teacher: a full 50-food episode returned **−18.13**
    while dying instantly returned **−1.00**. PPO was not failing — it was
    correctly learning to die immediately.
    Fixed by passing `body[1:-1]` (exclude the head, exclude the tail because it
    vacates), matching what `_get_obs` already did for the reachable mask. After
    the fix: good episode **+49.87**, penalty fires on 4% of steps instead of
    100%.
    Two lessons. First, this is the *flat per-step penalty* that the Tetris
    notes already warn about — reintroduced accidentally through a bug rather
    than a design choice, which is why the warning did not catch it. Second,
    **verify the reward ranks outcomes correctly before optimising it**;
    `sanity_check.py` now does exactly that.

16. **An entropy bonus destroyed a good BC policy.** After #14 was fixed, BC
    scored 47.43 (teacher: 51.96). Fine-tuning it with `ent_coef=0.01` drove it
    to **17.50 within 2.5M steps** — a 63% loss. An entropy bonus rewards being
    random; Snake has 3 actions and usually 2 are fatal, so pushing toward
    uniform is pushing toward death.
    v1's `ent_coef=0.0` was not the mistake it looked like — it was correct for
    that phase. The earlier reasoning ("it could not explore out of its basin")
    applied to a *mediocre* start (21.59, far below the teacher). Once BC lands
    near the teacher there is no bad basin to escape and exploration pressure
    only does damage. `train.py` now splits `ENT_COEF_FRESH = 0.01` from
    `ENT_COEF_RESUME = 0.0`.
    **This is the third time the same shape of bug has appeared** (see #2, #3):
    a setting that is right when starting from noise is wrong when starting
    from competence, and vice versa. Before changing any regularisation, ask
    which of the two situations you are in.

17. **BC leaves the critic at random init, and PPO's first updates then destroy
    the clone.** Behavioural cloning is a supervised classifier over *actions*
    — it never touches the value head. So a BC checkpoint pairs a good policy
    with a random critic, and PPO computes its first advantages from noise.
    Measured on Watermelon: BC scored 749.50, PPO began with
    `explained_variance = -0.016`, and rollout score fell 721 → 520 while the
    critic slowly caught up (ending at 0.78). By then the policy was wrecked.
    Entropy was *not* the cause here — setting `ENT_COEF_RESUME = 0.0` changed
    the result only 535 → 517. The audit that settled it: PPO earned **less
    reward** than BC (50.43 vs 73.94), so it wasn't out-optimising the reward,
    it was failing to optimise at all.
    Fixed with `pretrain.fit_value_function()`: after BC, roll out the clone,
    compute discounted returns, and regress the value head onto them with the
    policy frozen. Critic goes 0.001 → **0.822** before PPO starts, and score
    then holds at ~700 instead of collapsing.
    **Sizing matters:** the first attempt used 40 episodes / 6 epochs — about
    40 gradient steps — and the critic just learned the mean
    (`explained_variance` 0.00) while MSE fell convincingly. 200 episodes / 30
    epochs reaches 0.82. *Falling loss is not evidence of a useful critic;
    check explained variance.*
    Snake escaped this because 3 actions plus `target_kl` clamping leaves a
    softmax little room to scatter. Watermelon has 24.

18. **Watermelon PPO is blocked by a shared feature extractor, and BC is the
    shipped model.** After fixing the critic (#17), PPO still finished below
    BC (614.40 vs 749.50). The diagnosis, in order:
    - `target_kl` aborted **100% of updates at step 0** — configured for
      `n_epochs=10`, effectively running one, making big unaveraged jumps.
    - Halving `LR_RESUME` (5e-5 → 2e-5) changed the per-epoch KL **not at all**
      (still 0.05–0.06), which ruled the learning rate out as the driver.
    - The cause: `value_loss` ≈ 82 against `policy_gradient_loss` ≈ 1e-4. With
      `vf_coef=0.5` the value term dominates the gradient by ~5 orders of
      magnitude, and SB3's default `ActorCriticPolicy` **shares the features
      extractor** between policy and value — so value gradients flow through
      the shared CNN and rewrite the policy's features. Hence large KL
      independent of LR.
    **Untried fix:** `policy_kwargs(share_features_extractor=False)` so value
    gradients never touch the policy trunk. Needs a BC re-run (architecture
    change invalidates the checkpoint), so it was not attempted — PPO had
    already cost ~3 hours across four runs, all below BC.
    **Current state: `watermelon_final.zip` is a copy of the BC policy**
    (749.50, vs the heuristic teacher's 781.85 — 96% of it). That is what is
    exported to the browser. `watermelon_ppo_614.zip` is kept for reference.

19. **The Snake AI applied every action one cell too late.** In the browser it
    scored ~3 (max 11) while the identical exported model scored 72.3 in the
    training env. `update()` consumes `newDir` at
    `frameCount % STEPS_PER_CELL === 0`, and that alignment steers the *next*
    cell-to-cell move. Deciding at phase 1 — just after a boundary — meant the
    snake had already committed to its current move, so the turn only took
    effect a cell later. Fixed by deciding at
    `Math.floor(STEPS_PER_CELL / 2) + 1`.
    **How it was found, because guessing failed repeatedly:** the browser
    mechanics were ported to Python (`sim_browser_snake.py` pattern) and run
    against the real ONNX. That reproduced 2.92 avg / **max 11**, matching the
    reported symptom exactly. Diffing that simulation's observation against
    `SnakeEnv._get_obs()` for the same board gave **zero mismatches on all 11
    fields across 204 decisions** — which eliminated the encoder and pointed at
    the dynamics. Sweeping the decision phase then gave 2.92 at phase 1 versus
    66.92 at phase 0.
    Phases from the halfway flip to the boundary (6, 7, 8, 9, 0) all produce an
    identical observation, because `Segment.xx` rounds. Phase 6 is chosen to
    leave the async WASM inference ~67 ms instead of ~17 ms.
    **General lesson: when the browser and the env disagree, port the browser's
    mechanics into Python and bisect there.** Verifying the encoder is not
    enough — it was provably correct while the game was still unplayable.

20. **The browser killed the snake on its own tail.** `checkDeath()` iterated
    `body[1..length-1]`, counting the tail as a collision, while the env uses
    `body[:-1]` because the tail vacates as the snake moves. Tail-following is
    the standard survival move for a long snake, so the policy did the right
    thing and was killed for it. Real bug, fixed — but note it was *not* the
    cause of the low scores; #19 was. Two independent browser-vs-env rule
    mismatches existed at once.

**Browser / rendering**

6. **Grid alignment via `toFixed(1)` rounding** (from the original Snake
   extension) broke when the board was resized — segments stopped following the
   head. Fixed with an exact frame counter.
7. **`Food.generateNew()` read a global `snake`** — fine for one board, wrong
   with two. Now takes the snake as an explicit argument.
8. **Human keydown handler set flags on an unused `keys` object** instead of
   calling `snake.turn()`, so human movement silently stopped working.
9. **`Snake` constructor defaulted `dir`/`newDir` to `{0,0}`** — fine for the
   human (first keypress sets direction before `update()`), but deadlocked the
   AI: its frame counter only advances once moving, and it only moves once the
   counter fires a decision. Fixed by presetting `{x:1, y:0}`; `matchStarted`
   already gates whether `update()` runs.
10. **p5play does not render a stroked static sprite.** Setting `.stroke` /
    `.strokeWeight` on the watermelon loss-line sprite reported
    `visible: true` and painted **zero pixels**. Draw lines with `p.line()` in
    `draw()` instead. *Verify rendering by sampling canvas pixels, not by
    trusting a visibility flag.*
11. **p5play treats a string as an image path only if it contains a `.`**
    (`changeAni`: `typeof r === "string" && r.length !== 1 && r.includes(".")`).
    Base64 has no dots, so `sprite.img = <data URI>` was parsed as an
    *animation name* → every fruit rendered as a plain coloured circle, and
    pushing an 80KB string through the label parser stalled the loop.
    **Fix:** decode data URIs to `p5.Image` objects in `p.preload()` and assign
    those — `changeAni` accepts `p5.Image` directly.
12. **`select.html` grid showed grey bars at each end.** The grid was
    `width: 100%; max-width: 820px` but its columns only totalled ~782px, and
    the leftover centred space exposed the grid's own `#1e1e1e` background
    (the trick that draws the 1px separators). Fixed with
    `width: fit-content; max-width: 100%`.
13. **Don't use `auto-fit` in `select.html`'s grid** — it intermittently
    computes a phantom 4th column at wider viewports. Fixed columns only.
14. **A "made with p5play" splash appeared on the live site only.**
    `watermelon/lib/physics.min.js` injected a full-screen black overlay
    (`div#p5play-intro`, holding `https://p5play.org/v3/made_with_p5play.png`)
    on load. It is skipped for an allowlist of hostnames that includes `""`
    (i.e. `file://`), `localhost` and `127.0.0.1` — so it was invisible in
    every local test and first showed up once the site was served from
    `kohan1.github.io`. The injector has been deleted from the vendored lib
    (see the comment at the `default:` case of the `location.hostname` switch,
    ~line 4044). **If `physics.min.js` is ever re-downloaded or updated, the
    splash comes back** — re-apply the removal and confirm with
    `grep -c p5play-intro watermelon/lib/physics.min.js` returning 0.
    General lesson: a vendored library can behave differently on the deployed
    host than on localhost. Verify user-visible changes against the live URL,
    not just the dev server.
15. **"Stuck on loading" — the model was a render-blocking script.** Every game
    loaded `model_data.js` (base64 model, 45 MB Snake / 30 MB Watermelon /
    9 MB Tetris) as a plain `<script>` in `<head>`. The browser paints
    **nothing at all** until such a script has downloaded *and* parsed, so a
    cold visit was a blank white page for as long as that took. It bit right
    after a `deploy_pages.sh` run, because force-pushing gh-pages gives every
    file a new blob and invalidates the visitor's cache.
    **Fix:** all three loaders now prefer `fetch()`ing the `.onnx` and fall
    back to the base64 constant only when it is defined; `deploy_pages.sh`
    ships the `.onnx` and strips the `model_data.js` script tag. Nothing
    blocks the render, the human board is playable while the model streams,
    and the payload dropped 85 MB → 67 MB (base64 costs a third in inflation).
    Measured cold on the live site afterwards: DOMContentLoaded 489–841 ms,
    with the model arriving after. `model_data.js` stays in the repo because
    it is still the only thing that works under `file://`.
    **This was diagnosed late because every check was on a warm cache** —
    `performance.getEntriesByType('resource')` reported 0 ms / 0 KB for every
    file. Always check `transferSize` before concluding a page loads fine.

---

## Testing in the Browser pane

**Serve the site over HTTP; do not open it as `file://` in the pane.** Opening a
local file renders a *static snapshot* — no JavaScript runs, `javascript_exec`
reports "No site is open in this tab", and screenshots go stale. That cost a
whole session's worth of verification, during which two browser-side bugs
(#19, #20) shipped unnoticed because they could only be caught by watching the
game actually play.

**Check the frame rate before believing anything about the physics.** When the
Browser pane is hidden or not compositing, the browser throttles
requestAnimationFrame to about 2 fps. p5play steps physics per frame, so fruit
crawl downward and a mid-fall board looks exactly like frozen or floating
sprites — the symptom that a reverted `p.world.step()` change once produced for
real. Measured 2026-08-08 at 2 fps against 292 fps in a fronted tab.

```js
let n=0, t0=performance.now();
// count requestAnimationFrame callbacks for ~1.5s; under ~30 means throttled
```

`tabs_create` then `tabs_select` restores compositing. Two other things also
throttle judgement here: a **stale HTTP cache** (python's http.server sends no
cache headers, so an edited `shared/*.js` can keep serving the old version —
`fetch(url, {cache:'reload'})` on each changed file, THEN reload, since
`location.reload(true)` does not bypass it), and training running in the
background, which is its own section below.

`.claude/launch.json` defines a static server. Start it and navigate:

```
preview_start  name="humanvsai"       # serves the project root on :8321
navigate       http://localhost:8321/snake/game.html
```

Then `javascript_exec`, `computer` screenshots and console reads all work
normally. Verified: the Snake AI reached 40 on screen this way.

Two caveats:
- The pane only treats files as previewable if they are **inside the session's
  project folder**, and `preview_start` looks for `.claude/launch.json` in the
  directory Claude Code was launched from. If the project was renamed or moved
  mid-session, that path is stale — either relaunch from the current folder or
  point the server at the real one with `python -m http.server PORT --directory
  <path>`.
- Over HTTP, `fetch()` works, so the base64-embedding workaround is bypassed.
  That is fine for testing gameplay, but it means **the pane cannot catch
  `file://`-specific bugs**. Confirm anything asset- or model-loading related by
  double-clicking the HTML file in a real browser.

---

## Do not judge browser behaviour while training is running

Training saturates the CPU (two runs = ~24 python processes on 16 cores), and
the games render on the browser's main thread. Measured during a two-run
session: **2.2 FPS instead of 60**.

At that frame rate fruit crawl ~27x slower than normal and look frozen in
mid-air, drops take tens of seconds, and physics appears broken. It is not.

This cost real time: a report of "a fruit lost collision" was investigated as a
physics bug, two changes were made to fix it (raising p5play's solver
iterations by wrapping , and gating the next spawn on the drop zone
being clear), and both were then reverted while chasing a jam that was actually
starvation. Worse, the overlap measurements used to justify those changes were
themselves taken under load, so they proved nothing either way.

**Before diagnosing anything visual, measure the frame rate:**



Below ~20 FPS, stop and either pause training or test later. Any physics or
timing conclusion drawn under load is worthless.

## Platform gotchas (Windows)

- **`UnicodeEncodeError` on `✅`** — `torch.onnx` prints ✅ and the default
  cp1252 console can't encode it, crashing `export.py`. Run with
  `PYTHONIOENCODING=utf-8`. Not a code bug.
- **Renaming the project folder mid-session fails** with "process cannot access
  the file". A process's working directory locks that directory, and the agent
  shells sit inside it. Workaround: create the new folder and move the
  *contents* across (`robocopy /E /MOVE` handles stubborn handles), then delete
  the empty original after the session ends.
- **`node` is not on PATH here.** To syntax-check JS, fetch the file in the
  browser and `new Function(src)` — that's a real parse.

---

## GPU vs CPU for training

The old comment in `train.py` said "small MLP — CPU beats MPS/GPU". **That is
wrong for this net on the 4060 Ti.** Measured with the actual
`[512, 512, 256]` architecture and 773 inputs:

| Phase | CPU | CUDA | |
|---|---|---|---|
| Rollout forward (batch 32) | 0.349 ms | 0.292 ms | 1.19× faster |
| Update fwd+bwd (batch 2048) | 59.1 ms | 3.08 ms | **19.2× faster** |

End-to-end the gain is much smaller — a smoke test with 8 envs measured 2008
fps (CUDA) vs 1838 fps (CPU), about **9%** — because environment stepping, not
the network, dominates wall-clock time. Both `train.py` and `pretrain.py` now
use `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`.

Two notes:
- SB3 prints a generic warning that PPO with an MLP policy "should" run on CPU.
  Our measurements contradict it at this net size. Measure, don't assume.
- Anything moved to GPU needs its input tensors on the same device —
  `pretrain.py`'s BC batches previously relied on both being implicitly CPU.
- `N_ENVS = 32` on a 16-core machine is 2× oversubscribed. Changing it also
  changes the effective batch (`n_steps × n_envs`), so it isn't a free knob.

### Watermelon end-to-end throughput, measured 2026-07-27

Machine: Ryzen 7 3700X (8 physical / 16 logical), RTX 4060 Ti 16 GB.
Measured with `watermelon/training/bench_envs.py`, which times real
`model.learn()` throughput after a warm-up rollout:

| n_envs | device | fps |
|---|---|---|
| 8 | cuda | 534 |
| 12 | cuda | 685 |
| 16 | cuda | 593 |
| **20** | **cuda** | **1079** |
| 24 | cuda | 1028 |
| 16 | cpu | 217 |

**These numbers are for the 1332-float, 2-channel encoder. Re-measured
2026-08-08 on the 2652-float, 4-channel one:**

| n_envs | device | fps |
|---|---|---|
| 16 | cuda | 449 |
| **20** | **cuda** | **602** |

Roughly 25-45% slower for double the observation, which is the cost of the
extra channels showing up where the env's own comment predicted it would not
("channels are almost free" is true of PARAMETER COUNT — the flatten and the
Linear are unchanged — but not of throughput). 20 envs is still the sweet spot
and still worth ~34% over 16.

**Measure the MARGINAL rate, not the fps SB3 prints.** That figure is
cumulative from process start, so it is dragged down by warm-up for a long
time: the first table of this run read 222 fps when the true rate was 602.
Take two tables and divide the step delta by the time delta. Acting on the
first number would have made a 14-hour run look like a 38-hour one.

At 602 fps, budget ~14 hours for 30M steps.

**CUDA is ~5× CPU here, not the 9% measured on Snake** — Watermelon's env is
pymunk physics driven per drop, so the per-step Python cost is higher and the
GPU update is a bigger share of the total. Oversubscribing past 16 logical
cores still helps, because env workers idle during the update phase. 20 is
the sweet spot; 24 is slightly worse.

The 16-env figure being below both 12 and 20 is non-monotonic and is probably
noise — the combos run sequentially in one process. Treat single runs of this
benchmark as approximate, but the CPU/CUDA gap is far too large to be noise.

SB3 warns "PPO on the GPU ... should run on the CPU when not using a CNN
policy". It is a false positive here — the policy IS a CNN, wrapped in a
custom features extractor SB3's heuristic cannot see. The benchmark settles
it; do not act on that warning.

---

## Current status

- **Snake: done and live.** Run 2 resumed from run 1's model with the corrected
  LR and trained 20,054,016 steps. `evaluate.py` (50 eps, deterministic):
  **37.50 avg, median 39, max 56** — up from run 1's 36.58. Exported, embedded,
  and loading in the browser.
  - Backups kept: `snake_final.backup_20260725.zip` (the 36.58 model) and
    `checkpoints_backup_20260725/`.
  - **Rollout vs deterministic:** TensorBoard's `rollout/ep_score_mean` runs
    ~6–8 points *below* `evaluate.py`, because rollouts sample stochastically
    while evaluation uses argmax. Run 2 plateaued around 30–31 in rollout,
    which is ~37.5 deterministic. Don't read a rollout plateau as failure.

### The Snake agent is losing to its own teacher (measured 2026-07-26)

| Policy | Avg over 50 eps | Max |
|---|---|---|
| `heuristic_action` (BFS + flood fill) | **51.96** | 83 |
| Trained PPO (`snake_final.zip`) | 37.50 | 56 |
| BC-only (`snake_pretrained.zip`) | 21.59 | — |

The hand-coded teacher beats the trained student by ~28%. So the plateau is
**not** a hyperparameter problem — more steps or another LR sweep will not fix
it. Two things follow:

- BC never came close to cloning the teacher (21.59 vs 51.96), and RL only
  clawed back part of the gap.
- **Likely root cause — information asymmetry.** The teacher decides using
  flood-fill reachable-space (`heuristic._flood_fill_size`), but the
  observation contains *no* such feature: just a raw 16×16×3 one-hot grid,
  flattened, through an MLP. A flat MLP essentially cannot compute connectivity
  over a one-hot grid, so the student is being asked to imitate decisions it
  cannot see the basis for.

Highest-value fixes, in order: add flood-fill/free-space features to the
observation; switch to a CNN feature extractor over the `16×16×3` grid so
spatial structure is learnable at all; only then revisit reward shaping and
entropy. Re-measure the teacher whenever the observation changes — it is the
reference point that makes "is this actually good?" answerable.
- **Tetris:** playable, drop animation done, persistence done. The live
  neural-net visualisation is half-built — `export.py` has the
  `ActorWithActivations` wrapper emitting 4 outputs (`action_logits`,
  `layer1_act` 512, `layer2_act` 512, `layer3_act` 256), but it has **never
  been run against the real model**, and `game.js` still reads only
  `action_logits`. With `models/` now deleted, finishing it needs a retrained
  Tetris checkpoint first.
- **Watermelon:** human board fully playable, AI board is a scaffold showing
  "Awaiting model". No training pipeline exists.

## The checkpoint switcher

Each game offers a ladder of earlier models under its AI board, so a visitor can
play a genuinely weaker network rather than a good one playing badly. It is the
second difficulty axis; the temperature setting in `settings.js` is the first,
and `settings.js` explains why one is not enough (Watermelon has a floor of
about 42% of full strength no matter how high T goes).

- `tools/build_checkpoints.py` — exports each rung, measures it, writes the
  manifest `checkpoints.js`.
- `checkpoint_switcher.js` + `checkpoints.css` — the control itself.
- `<game>/checkpoints/*.onnx` — the rungs. **Gitignored** (`checkpoints/` and
  `*.onnx` both match), so a fresh clone has the manifest but not the models.
  `deploy_pages.sh` aborts in that case and tells you to rebuild.

Three things here are easy to get wrong, and all three were:

**Tetris must be evaluated WITH ACTION MASKING.** Its 40 logits are candidate
placements and only the first `len(placements)` are legal; `tetris/game.js` caps
the argmax there. `tetris_env.step()` does *not* reject an out-of-range action —
it wraps it with `action % len(placements)` — so an unmasked argmax silently
plays an arbitrary legal move instead of raising. Unmasked, the 70M checkpoint
measured 559 and the 260M measured 211, i.e. the later model looked *worse*.
Masked: 746 and 1188. Nothing about the first numbers was real.

**`num_timesteps` is per-run, not cumulative.** It resets whenever a run starts
fresh rather than resuming, so Snake's 71-point rung reports 30.0M steps while
its stronger 129-point rung reports 6.0M. The manifest carries `stepsMeaning`
(`"lineage"` vs `"run"`) and the switcher only shows step counts for Tetris,
whose three rungs really are one continuous run.

**One process per rung.** All three games define a module named `policy_config`,
and the first imported wins for the life of the interpreter — loading Snake then
Watermelon silently gave Watermelon's checkpoints Snake's policy class. The env
modules collide the same way. Every export and evaluation runs in a fresh
interpreter.

Scores are measured by playing the exact `.onnx` the browser downloads, not
copied from the archive filenames. That doubles as an export check: Watermelon's
rungs reproduced their archived scores to the cent (841.87, 936.70, 1032.43),
because the env is deterministic under a fixed seed.

---

## Open / next up

- Watermelon AI. A physics-based merge game is a genuinely hard RL problem —
  a Python env would need to reimplement the planck physics to train against.
- Tetris neural-net visualisation: run `export.py`, confirm it prints
  `Found 3 hidden layer(s) to expose`, re-run `embed_model.py`, then build the
  `game.js` side that reads the activation outputs.
- Snake score is plateaued; see the ideas discussed for pushing past ~37.

---

## Useful commands

```bash
# Snake training (from snake/training/)
pip install -r requirements.txt
python pretrain.py                        # BC warm start  → snake_pretrained.zip
python train.py                           # PPO fine-tune  → snake_final.zip
python evaluate.py snake_final.zip 50     # headless deterministic score
python -m tensorboard.main --logdir tb_logs/

# Snake export → browser
PYTHONIOENCODING=utf-8 python export.py   # → snake_ai.onnx  (utf-8 matters on Windows)
cp snake_ai.onnx ../snake_ai.onnx
cd .. && python embed_model.py            # → model_data.js

# Watermelon assets (from watermelon/, after changing assets/)
python embed_assets.py                    # → image_data.js

# Checkpoint ladders (from the repo root) — rebuilds <game>/checkpoints/*.onnx
# and the checkpoints.js manifest. Slow: it plays real games to score each rung.
python tools/build_checkpoints.py                  # all three games
python tools/build_checkpoints.py snake --episodes 10   # one game, rougher
python tools/probe_checkpoints.py         # which archived .zip files still load

# After ANY training run — see "After every training run" at the top of this
# file. The Inside page does not update itself.
cp <game>/training/remote_training.log <game>/training/logs/train_<name>.log
#   ... then add a RUN_NOTES entry in tools/build_training_data.py ...
python tools/build_training_data.py       # -> inside/data.js
bash tools/deploy_pages.sh
```

# human vs ai

Three browser games, each with an opponent that learned to play by reinforcement
learning. You play on the left, the model plays on the right, and a fourth page
shows exactly how it was trained — including the runs that failed.

**▶ [kohan1.github.io/humanvsai](https://kohan1.github.io/humanvsai/)**

No install, no server, no account. The models run in your browser via
onnxruntime-web.

---

## The opponents

| game | opponent | how it plays |
|---|---|---|
| **Snake** | averages **145.7** | PPO on a 16×16 grid, 3 relative actions (left / straight / right) |
| **Tetris** | averages **~119k** | MaskablePPO over 40 candidate placements, trained to 1B steps |
| **Watermelon** | averages **1032** | PPO on a Suika-style physics game, 24 drop columns |

Every one of those numbers is measured, not estimated — by playing the exact
`.onnx` file the browser downloads, over a fixed set of seeds. The method is on
the Inside page.

## What is actually interesting here

**It shows its working.** Most projects like this show you the model that
worked. [Inside](https://kohan1.github.io/humanvsai/inside.html) shows all 24
training runs across **1.45 billion steps and 451 hours of GPU** — including
the seven that were rejected and why. There's a family tree of which run
descended from which, annotated learning curves, score distributions, a
CPU-vs-GPU benchmark, and a live view of a Tetris network's neurons firing as
it plays.

**You can see what the model sees.** Every game has a panel showing the exact
observation tensor being fed to the network that frame, its action
probabilities, its confidence, and the critic's estimate of how well the
current position will turn out.

**You can play earlier versions of the AI.** A model-version switcher lets you
face checkpoints from partway through training — Snake at 71 points instead of
149, Tetris at 746 instead of 107k. They're fetched only when selected.

**Two independent difficulty dials.** Difficulty is a *sampling temperature*:
instead of always taking its best move, the model samples from
`softmax(logits / T)`, so it plays like a weaker player rather than a broken
one. The temperatures are measured per game because the games punish a mistake
very differently — Snake loses half its score from an 8% move change, Tetris
loses 88% from 16%.

**Six themes**, each a palette plus a cursor-reactive background — a wireframe
mesh, flow trails, iron filings, drifting sand, a constellation, and a wordmark
you can physically scatter.

## Running it locally

```bash
python -m http.server 8321
```

Then open <http://localhost:8321/select.html>.

The `.onnx` models are gitignored (they're tens of MB and change wholesale each
training cycle). Grab them from the
[Releases](https://github.com/kohan1/humanvsai/releases) page and drop them
beside each game's `game.html`, or the pages will load with the AI board
showing "Awaiting model" while the human board still plays.

## Training

Each game has a self-contained `training/` folder — a Gymnasium environment
mirroring the browser game, a heuristic teacher, behavioural-cloning warm start,
PPO fine-tuning, and ONNX export.

```bash
cd watermelon/training
pip install -r requirements.txt
python sanity_check.py     # verifies the problem definition before you spend hours
python train.py
```

`sanity_check.py` exists because three separate Snake runs were lost to
problems detectable in under a minute — a reward that ranked dying above
playing well, a teacher barely better than random, an observation that didn't
match its own declaration. It checks those before PPO ever starts.

## Layout

```
index.html select.html inside.html   entry, picker, the training write-up
shared/                              themes, backgrounds, settings, inspector
snake/ tetris/ watermelon/           game.html + game.js + training/
tools/                               train, evaluate, install, deploy
```

`CLAUDE.md` is the engineering log: every non-obvious bug, its root cause, and
the measurements behind each design decision. It's the most useful file here if
you want to know *why* something is the way it is.

## Built with

Stable-Baselines3 · PyTorch · Gymnasium · pymunk · ONNX Runtime Web · p5play /
planck.js · plain HTML, CSS and JavaScript on the front end — no framework, no
build step.

---

A Digital Technologies project. The write-up on the Inside page is the real
deliverable; the games are how you get to it.

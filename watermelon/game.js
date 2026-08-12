/* ──────────────────────────────────────────────────────────────────────────
   Watermelon Game — human vs ai

   Ported from the original single-board build. Two changes of substance:

   1. INSTANCE MODE. The original ran as a global sketch, which allows exactly
      one physics world per page. The physics library attaches `world`,
      `Sprite`, `Group`, `allSprites` and `kb` to the sketch instance rather
      than to `window`, so running each board as its own instance gives two
      fully independent worlds. Everything the sketch touches is therefore
      reached through `p.` — that prefix is not decoration, dropping it would
      silently bind to whichever board initialised last.

   2. PERSISTENCE. The original stored state in a synced storage API that
      does not exist on a plain web page. Replaced with localStorage
      behind the `store` helpers below. Only the human board persists —
      `cfg.persist` gates it — so the AI board can never overwrite a real
      score.

   Game-over no longer reloads the page. With two boards on one document a
   reload would reset both, so each board resets itself in place instead.

   ── Wiring an AI ──
   The AI board is scaffolded but idle. To make it play, assign a policy
   function to the returned controller:

       boards.ai.setPolicy((state) => 0.5);

   It receives a state snapshot each frame and returns either a number in
   0..1 (drop at that fraction of board width) or null to keep holding. See
   `buildState()` for the snapshot's shape. Nothing else needs to change.
   ────────────────────────────────────────────────────────────────────────── */

(() => {
    "use strict";

    /* ── Game constants (unchanged from the original) ─────────────────────── */

    const ASSET_PATHS = [
        "assets/cherry.png",
        "assets/strawberry.png",
        "assets/grape.png",
        "assets/lemon.png",
        "assets/orange.png",
        "assets/apple.png",
        "assets/whitefruit.png",
        "assets/peach.png",
        "assets/pineapple.png",
        "assets/honeydew.png",
        "assets/watermelon.png",
    ];
    const CLOUD_PATH = "assets/cloud.png";

    /* Resolve an asset to an embedded data URI when one is available.

       p5's loadImage() sets crossOrigin="Anonymous" on any URL that is not
       already a data: URI. Under file:// the origin is opaque, so that
       request is rejected and the image never loads — the sprite still
       exists and the physics still runs, so the game looks alive while
       drawing nothing. Data URIs are the one case p5 skips the flag for.

       Falls back to plain paths so the game still works when served over
       http(s) without image_data.js present. */
    const EMBEDDED =
        typeof WATERMELON_IMAGES !== "undefined" ? WATERMELON_IMAGES : null;

    const asset = (path) =>
        EMBEDDED && EMBEDDED[path] ? EMBEDDED[path] : path;

    const IMAGES    = ASSET_PATHS.map(asset);
    const CLOUD_IMG = asset(CLOUD_PATH);

    const POINTS     = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78];
    /* A geometric ladder, 24 -> 200 across eleven tiers. Mirrored in
       watermelon/training/watermelon_env.py and the two MUST stay identical.

       The previous sizes ([30, 46, 70, 80, 100, 125, 150, 177, 200, 230, 290])
       made the game finite no matter how well it was played. Two 290px fruit
       cannot touch anywhere below the loss line — side by side they need 580px
       against a 448px well, stacked they need 580px against 484px of height,
       and diagonally the walls force their centres far enough apart vertically
       that the upper one is already past the loss line. Since a vanishing
       top-tier pair is the ONLY way area leaves the well, area could only ever
       accumulate. A watermelon was worse than useless: reachable from two tier
       9s, then permanently unmergeable, squatting on 30% of the board forever.

       At 200px a top pair needs 400px and fits with 48px to spare, so the sink
       is live and an unbounded game is possible in principle. The even 24%
       steps also mean every merge shrinks the area it occupies (0.74-0.78x);
       the old ladder grew it at the bottom, where two 30px fruit became a 46px
       fruit covering 1.18x the area. */
    const DIAMETERS  = [24, 30, 37, 45, 56, 69, 86, 106, 131, 162, 200];
    const MAX_TIER   = IMAGES.length - 1;
    /* MAX_TIER comes from the image list, so a ladder of the wrong length
       would silently give some tier no size (undefined diameter) or no sprite.
       Cheap to check, and invisible if it ever goes wrong. */
    if (DIAMETERS.length !== IMAGES.length) {
        throw new Error(
            `DIAMETERS has ${DIAMETERS.length} entries but there are ` +
            `${IMAGES.length} fruit images — they must match one-to-one.`
        );
    }

    /* ── Board palette ────────────────────────────────────────────────────
       The play area follows the site theme, like Snake and Tetris already do.
       Watermelon was the one game that did not: the canvas was a hardcoded
       cream #FDFBEC, so on the two paper themes it was a slightly different
       white than the page, and on the four dark ones a bright slab in the
       middle of a near-black layout. --board-bg and friends are the same
       tokens the other two games read, so all three now share one palette.

       Cached rather than read per frame. Snake and Tetris call getComputedStyle
       inside their draw loop, which is fine for one canvas; Watermelon runs TWO
       p5 instances at 60fps, so that would be 240 style resolutions a second to
       return the same four strings. The MutationObserver invalidates the cache
       the instant data-theme changes, which is the only time they can move. */
    let _palette = null;
    function boardColour(name, fallback) {
        if (!_palette) {
            const cs = getComputedStyle(document.documentElement);
            _palette = {
                bg:   cs.getPropertyValue('--board-bg').trim(),
                ink:  cs.getPropertyValue('--board-ink').trim(),
                edge: cs.getPropertyValue('--board-edge').trim(),
            };
        }
        return _palette[name] || fallback;
    }
    new MutationObserver(() => { _palette = null; })
        .observe(document.documentElement,
                 { attributes: true, attributeFilter: ['data-theme'] });

    function boardBg()   { return boardColour('bg',   '#FDFBEC'); }
    function boardInk()  { return boardColour('ink',  'black'); }
    function boardEdge() { return boardColour('edge', 'gray'); }

    const WEIGHTED = {
        initGame: [0, 1, 2, 3, 4],
        midGame:  [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
        endGame:  [0, 1, 2, 2, 2, 3, 3, 3, 4, 4],
    };

    const CANVAS_W = 448;
    const CANVAS_H = 599;
    const LOSS_LINE_Y = 115;   // stack above this and the game ends

    const BALL_TIMEOUT      = 1000;

    // AI speed multiplier, matching Tetris's speed buttons. Applies to the AI
    // board only — the human board's pacing is the player's business.
    //
    // Two things have to scale together, or the board fights itself: the
    // cooldown between drops, and how fast the cloud slides to its target. At
    // 2x with unchanged easing the AI would be ready to drop before the cloud
    // had arrived, and just stall on the alignment check.
    const AI_SPEEDS = [0.25, 0.5, 0.75, 1, 1.5, 1.75, 2];
    let AI_SPEED = 1;
    const CLOUD_EASE_BASE = 0.2;
    const SHAKE_STRENGTH    = 50;
    const SHAKE_COOLDOWN_S  = 15;
    const DROPS_PER_SHAKE   = 25;

    /* ── Storage ──────────────────────────────────────────────────────────── */

    const store = {
        read(key, fallback = null) {
            try {
                const raw = localStorage.getItem(key);
                return raw === null ? fallback : JSON.parse(raw);
            } catch (e) {
                return fallback;
            }
        },
        write(key, value) {
            try {
                localStorage.setItem(key, JSON.stringify(value));
            } catch (e) {
                /* private browsing / quota — non-fatal, the game plays on */
            }
        },
        clear(key) {
            try {
                localStorage.removeItem(key);
            } catch (e) {}
        },
    };

    function weightedTier(dropped) {
        const pool =
            dropped > 50 ? WEIGHTED.endGame
          : dropped > 25 ? WEIGHTED.midGame
          :                WEIGHTED.initGame;
        return pool[Math.floor(Math.random() * pool.length)];
    }

    /* ── Board factory ────────────────────────────────────────────────────── */

    function createBoard(cfg) {
        const el = (suffix) => document.getElementById(`${suffix}-${cfg.id}`);

        const domScore         = el("score");
        const domNextBall      = el("nextball");
        const domHighScore     = el("highscore");
        const domGameOver      = el("gameover");
        const domGameOverScore = el("gameover-score");
        const domShakeBtn      = el("shake");
        const domShakeCount    = el("shakecount");
        const domShakeCountdown= el("shakecountdown");
        const domNewGame       = el("newgame");
        const domPlayAgain     = el("playagain");

        const KEY_HIGH    = `watermelon.${cfg.id}.highScore`;
        const KEY_SAVED   = `watermelon.${cfg.id}.savedGame`;
        const KEY_DROPPED = `watermelon.${cfg.id}.ballsDropped`;

        let policy = null;   // AI hook — null means "do not act"
        let api    = null;   // controller returned to the caller

        const sketch = (p) => {
            let balls, bounds, loss;
            let wall1, wall2, ground, lossLine;
            let nextBall, cloud, cloudBall;

            let score        = 0;
            let highScore    = 0;
            let ballsDropped = 0;
            let canDrop      = false;
            let isGameOver   = false;
            let loading      = true;

            let numOfShakes   = 0;
            let canShake      = true;
            let doShake       = false;

            // Decoded p5.Image objects, one set per instance.
            let FRUIT_IMG   = [];
            let CLOUD_IMAGE = null;

            /* ── Preload ──────────────────────────────────────────────────────

               Sprites are handed p5.Image objects, never raw source strings.

               The physics library's `sprite.img` setter forwards to changeAni(),
               which only treats a string as an image to load when it contains a
               "." — otherwise the string is read as an *animation name*. Base64
               contains no dots, so a data URI silently becomes a lookup for an
               animation that does not exist: every sprite falls back to a plain
               coloured circle, and pushing an 80 KB string through the label
               parser stalls the loop. changeAni() accepts a p5.Image directly,
               which sidesteps that path entirely.

               Loading here rather than in setup() also means p5's preload
               counter holds setup() until every image is decoded, so no sprite
               is ever built from a half-loaded image. Per instance, so the two
               boards never share a p5.Image across canvas contexts. */

            p.preload = () => {
                FRUIT_IMG   = IMAGES.map((src) => p.loadImage(src));
                CLOUD_IMAGE = p.loadImage(CLOUD_IMG);
            };

            /* ── Setup ────────────────────────────────────────────────────── */

            p.setup = () => {
                new p.Canvas(CANVAS_W, CANVAS_H);
                p.world.gravity.y = 20;

                /* Resize every fruit image to its collider.

                   p5play draws a sprite's animation at the image's NATURAL
                   size — `sprite.diameter` sizes the physics circle and does
                   not stretch the picture to match. The PNGs were authored
                   against the original ladder and their widths still track it
                   almost exactly (watermelon.png is 291px for the old 290,
                   apple.png 128 for 125), so for years natural size and
                   collider size agreed by construction and nothing had to
                   convert between them.

                   Compressing the ladder broke that coincidence. Colliders
                   shrank, images did not, and every fruit rendered 1.4-1.9x
                   larger than the circle it actually occupied — so fruit
                   appeared to sit inside one another while the physics was
                   entirely correct and reported no overlaps at all.

                   Scaling by width preserves each sprite's aspect ratio and
                   restores the intended relationship. Not `sprite.scale`,
                   which calls _resizeColliders() and would move the physics to
                   match the picture — exactly backwards. */
                FRUIT_IMG.forEach((img, tier) => {
                    const aspect = img.height / img.width;
                    img.resize(DIAMETERS[tier], Math.round(DIAMETERS[tier] * aspect));
                });

                // NOTE: an attempt to raise the solver's iteration counts by
                // wrapping p.world.step() was reverted — it froze fruit in
                // mid-air (they hung in a vertical line with 90px gaps and
                // never fell). The interpenetration it was meant to fix is
                // cosmetic; stopping the physics is not. If this is revisited,
                // verify fruit still FALL before measuring overlap.

                bounds = new p.Group();
                loss   = new p.Group();

                // Walls and floor take the theme's board edge. They are 1px
                // and mostly sit outside the visible area, but "gray" against
                // a near-white board on the paper themes was invisible.
                wall1 = new bounds.Sprite(0, CANVAS_H / 2, 1, CANVAS_H, "s");
                wall1.color = boardEdge();
                wall2 = new bounds.Sprite(CANVAS_W, CANVAS_H / 2, 1, CANVAS_H, "s");
                wall2.color = boardEdge();

                ground = new bounds.Sprite(CANVAS_W / 2, CANVAS_H, CANVAS_W * 2, 1, "s");
                ground.color = boardEdge();
                ground.bounciness = 0;

                // Collision sensor only — it is never drawn. Setting .stroke
                // on a static sprite renders nothing (verified: the sprite
                // reports visible but paints no pixels), so the visible red
                // line is drawn with p.line() in draw() instead.
                lossLine = new loss.Sprite(CANVAS_W / 2, LOSS_LINE_Y, CANVAS_W, 1, "s");
                lossLine.visible = false;

                nextBall = new p.Sprite(CANVAS_W - 100, 100);
                nextBall.tier = Math.round(p.random(0, 4));
                nextBall.collider = "n";
                nextBall.diameter = DIAMETERS[nextBall.tier];
                nextBall.img = FRUIT_IMG[nextBall.tier];
                nextBall.visible = false;
                renderNextBall();

                cloud = new p.Sprite(CANVAS_W / 2, 50, 75, 50, "n");
                cloud.img = CLOUD_IMAGE;
                cloud.scale = 0.8;

                balls = new p.Group();
                balls.tier;
                balls.isCombining = false;
                balls.isCloud;
                balls.rotationDrag = 0.7;
                balls.textColor = "white";
                balls.textSize = 24;
                balls.bounciness = 0;

                balls.collide(balls, combineFruits);

                createCloudBall(cloud.x, cloud.y, Math.round(p.random(0, 4)));

                canDrop = true;
                highScore = store.read(KEY_HIGH, 0) || 0;
                ballsDropped = cfg.persist ? (store.read(KEY_DROPPED, 0) || 0) : 0;
                renderScore();

                if (cfg.persist && store.read(KEY_SAVED)) loadSavedGame();
                else loading = false;
            };

            /* ── Draw ─────────────────────────────────────────────────────── */

            p.draw = () => {
                p.background(boardBg());

                // The human board's cloud tracks the pointer; the AI board's
                // tracks whatever its policy asks for, and parks centre when
                // there is no policy yet.
                let targetX;
                if (cfg.interactive) {
                    targetX =
                        p.mouseX > 0 && p.mouseX < CANVAS_W
                            ? p.mouseX
                            : p.mouseX > CANVAS_W / 2 ? wall2.x - 5 : wall1.x + 5;
                } else {
                    const want = policy ? policy(buildState()) : null;
                    targetX = want === null || want === undefined
                        ? CANVAS_W / 2
                        : Math.min(CANVAS_W - 5, Math.max(5, want * CANVAS_W));
                }
                // The AI board's cloud tracks at the selected speed; the human
                // board is never sped up. Capped below 1 because moveTowards
                // treats the factor as a fraction of the remaining distance —
                // at >=1 it teleports and the motion reads as a jump cut.
                cloud.moveTowards(
                    targetX,
                    cloud.y,
                    cfg.interactive ? CLOUD_EASE_BASE
                                    : Math.min(0.9, CLOUD_EASE_BASE * AI_SPEED)
                );

                if (cloudBall) {
                    cloudBall.x = cloud.x;
                    cloudBall.y = cloud.y + 50;
                }

                if (cfg.interactive && p.kb.pressed("e")) shakeClicked();
                doEarthquake();

                for (const x of balls) {
                    if (x.isCloud) continue;
                    if (x.overlapping(lossLine) > 60) gameOver();
                }

                // The drop guide. Was "gray", which sat at 2.3:1 on the paper
                // themes' near-white board and vanished; --board-edge is the
                // token the other two games already use for exactly this.
                p.stroke(boardEdge());
                p.strokeWeight(6);
                p.line(cloud.x, cloud.y, cloud.x, CANVAS_H);

                // Stack limit — fruit resting above this line ends the game.
                // Stays red on every theme: it is the one mark on the board
                // that means "you are about to lose", and a warning that
                // restyles itself per palette stops reading as a warning.
                p.stroke("red");
                p.strokeWeight(3);
                p.line(0, LOSS_LINE_Y, CANVAS_W, LOSS_LINE_Y);

                p.stroke(boardInk());
                p.strokeWeight(1);
            };

            /* ── Input ────────────────────────────────────────────────────── */

            p.mouseReleased = () => {
                if (!cfg.interactive) return;
                // Bounds-check both axes: each instance reports pointer
                // position relative to its own canvas, so this is what stops
                // a click on the other board from dropping here.
                if (p.mouseX <= -25 || p.mouseX >= CANVAS_W + 25) return;
                if (p.mouseY <= -25 || p.mouseY >= CANVAS_H + 25) return;
                drop();
            };

            /* ── Core actions ─────────────────────────────────────────────── */

            function drop() {
                if (!canDrop || doShake || isGameOver || loading) return;
                if (!cloudBall) return;

                canDrop = false;
                ballsDropped++;
                if (cfg.persist) store.write(KEY_DROPPED, ballsDropped);

                const ball = cloudBall;
                cloudBall = undefined;

                if (ballsDropped % DROPS_PER_SHAKE === 0) {
                    numOfShakes++;
                    if (domShakeCount) domShakeCount.innerText = numOfShakes;
                    if (domShakeBtn) domShakeBtn.disabled = false;
                }

                ball.collider = "d";
                ball.isCloud = false;
                ball.x = ball.x + p.random(-1, 1);
                ball.diameter = DIAMETERS[ball.tier];
                ball.bounciness = 0;
                ball.resetMass();

                setTimeout(() => {
                    if (isGameOver) return;
                    createCloudBall(cloud.x, cloud.y, nextBall.tier);
                    queueBall();
                    saveGame();
                    canDrop = true;
                    // AI board only: shorten the cooldown by the speed
                    // multiplier. The human board keeps the original 1000ms.
                }, cfg.interactive ? BALL_TIMEOUT : BALL_TIMEOUT / AI_SPEED);
            }

            async function combineFruits(a, b) {
                if (a.isCloud || b.isCloud) return;
                if (a.isCombining || b.isCombining) return;
                if (a.tier !== b.tier) return;

                a.isCombining = true;
                b.isCombining = true;

                const tier = a.tier;
                const aX = a.x, aY = a.y, aIndex = balls.indexOf(a);
                const bX = b.x, bY = b.y, bIndex = balls.indexOf(b);

                score += POINTS[tier];
                if (score > highScore) {
                    highScore = score;
                    saveHighScore(highScore);
                }
                renderScore();

                a.overlaps(b);
                a.direction = a.angleTo(b);
                b.direction = b.angleTo(a);
                a.speed = 5;
                b.speed = 5;

                for (const x of balls) {
                    if (x.isCombining) continue;
                    if (p.dist(x.x, x.y, (aX + bX) / 2, (aY + bY) / 2) > CANVAS_W / 2.5) continue;
                    x.moveAway((aX + bX) / 2, (aY + bY) / 2, 0.01);
                }

                await p.delay(100);

                a.remove();
                b.remove();

                if (tier === MAX_TIER) return;

                const merged = createBall((aX + bX) / 2, (aY + bY) / 2, tier + 1);
                merged.moveTowards(
                    aIndex >= bIndex ? aX : bX,
                    aY >= bY ? bY : aY,
                    0.02
                );
            }

            function createBall(x, y, tier, vel = undefined) {
                const ball = new balls.Sprite(x, y);
                ball.tier = tier;
                ball.img = FRUIT_IMG[tier];
                ball.collider = "d";
                ball.bounciness = 0;
                ball.diameter = DIAMETERS[tier];
                ball.resetMass();
                if (vel) ball.velocity = vel;
                return ball;
            }

            function createCloudBall(x, y, tier) {
                cloudBall = new balls.Sprite(x, y);
                cloudBall.tier = tier;
                cloudBall.img = FRUIT_IMG[tier];
                cloudBall.collider = "s";
                cloudBall.diameter = DIAMETERS[tier];
                cloudBall.isCloud = true;
            }

            function queueBall(t = weightedTier(ballsDropped)) {
                nextBall.tier = t;
                nextBall.diameter = DIAMETERS[t];
                nextBall.img = FRUIT_IMG[t];
                renderNextBall();
            }

            /* ── Shake ────────────────────────────────────────────────────── */

            async function doEarthquake() {
                if (!doShake) return;
                for (const ball of balls) {
                    if (ball.isCloud) continue;
                    ball.moveTowards(
                        ball.x + p.random(-SHAKE_STRENGTH, SHAKE_STRENGTH),
                        ball.y + p.random(-SHAKE_STRENGTH, SHAKE_STRENGTH)
                    );
                }
                await p.delay(2000);
                doShake = false;
            }

            function shakeClicked() {
                if (!canShake || numOfShakes < 1 || isGameOver) return;

                let remaining = SHAKE_COOLDOWN_S;
                numOfShakes--;
                if (domShakeCount) domShakeCount.innerText = numOfShakes;
                doShake = true;
                canShake = false;

                if (domShakeBtn) domShakeBtn.disabled = true;
                if (domShakeCountdown) {
                    domShakeCountdown.style.display = "";
                    domShakeCountdown.innerText = remaining;
                }

                const countdown = setInterval(() => {
                    if (remaining < 0) return clearInterval(countdown);
                    if (domShakeCountdown) domShakeCountdown.innerText = --remaining;
                }, 1000);

                setTimeout(() => {
                    clearInterval(countdown);
                    if (domShakeCountdown) domShakeCountdown.style.display = "none";
                    canShake = true;
                    if (numOfShakes > 0 && domShakeBtn) domShakeBtn.disabled = false;
                }, SHAKE_COOLDOWN_S * 1000);
            }

            /* ── Persistence ──────────────────────────────────────────────── */

            function saveGame() {
                if (!cfg.persist) return;
                const state = { balls: [], cloudTier: null, nextTier: null,
                                score: 0, dropped: 0, shakes: 0 };
                for (const a of balls) {
                    if (a.isCloud) continue;
                    state.balls.push({
                        x: a.x, y: a.y, tier: a.tier,
                        diameter: a.diameter, vel: { x: a.vel.x, y: a.vel.y },
                    });
                }
                state.cloudTier = cloudBall ? cloudBall.tier : null;
                state.nextTier  = nextBall.tier;
                state.score     = score;
                state.dropped   = ballsDropped;
                state.shakes    = numOfShakes;
                store.write(KEY_SAVED, state);
            }

            function loadSavedGame() {
                const state = store.read(KEY_SAVED);
                if (!state) {
                    loading = false;
                    return;
                }
                for (const a of state.balls || []) {
                    createBall(a.x, a.y, a.tier, a.vel);
                }
                if (typeof state.cloudTier === "number") {
                    if (cloudBall) cloudBall.remove();
                    createCloudBall(cloud.x, cloud.y, state.cloudTier);
                }
                if (typeof state.nextTier === "number") queueBall(state.nextTier);
                if (typeof state.score === "number") { score = state.score; renderScore(); }
                if (typeof state.dropped === "number") ballsDropped = state.dropped;
                if (typeof state.shakes === "number") {
                    numOfShakes = state.shakes;
                    if (domShakeCount) domShakeCount.innerText = numOfShakes;
                    if (domShakeBtn) domShakeBtn.disabled = numOfShakes < 1;
                }
                loading = false;
            }

            function saveHighScore(value) {
                if (!cfg.persist) return;
                store.write(KEY_HIGH, value);
            }

            /* ── Game over / reset ────────────────────────────────────────── */

            function gameOver() {
                if (isGameOver || doShake || loading) return;
                isGameOver = true;
                canDrop = false;

                // Human board only, and gameOver() is already guarded against
                // re-entry, so this fires once per match.
                if (cfg.persist && typeof MatchResults !== "undefined") {
                    let aiScore = 0;
                    try { aiScore = getAI().getScore(); } catch (e) { /* AI not up yet */ }
                    MatchResults.record("watermelon", score, aiScore, Date.now());
                }

                store.clear(KEY_SAVED);
                if (domGameOverScore) domGameOverScore.innerText = `Score ${score}`;
                if (domGameOver) domGameOver.hidden = false;
            }

            // Resets this board only. The original reloaded the document,
            // which would take the other board down with it.
            function reset() {
                for (const b of [...balls]) b.remove();
                cloudBall = undefined;

                score = 0;
                ballsDropped = 0;
                numOfShakes = 0;
                canShake = true;
                doShake = false;
                isGameOver = false;

                store.clear(KEY_SAVED);
                if (cfg.persist) store.write(KEY_DROPPED, 0);

                queueBall(Math.round(p.random(0, 4)));
                createCloudBall(cloud.x, cloud.y, Math.round(p.random(0, 4)));

                if (domGameOver) domGameOver.hidden = true;
                if (domShakeCount) domShakeCount.innerText = 0;
                if (domShakeBtn) domShakeBtn.disabled = true;
                if (domShakeCountdown) domShakeCountdown.innerText = "";

                renderScore();
                loading = false;
                canDrop = true;
            }

            /* ── Rendering to the DOM ─────────────────────────────────────── */

            function renderScore() {
                if (domScore) domScore.innerText = score;
                if (domHighScore) domHighScore.innerText = highScore;
            }

            function renderNextBall() {
                if (!domNextBall) return;
                domNextBall.setAttribute("src", IMAGES[nextBall.tier]);
                domNextBall.setAttribute("alt", `Next fruit, tier ${nextBall.tier}`);
            }

            /* ── State snapshot handed to an AI policy ────────────────────── */

            function buildState() {
                const fruit = [];
                for (const b of balls) {
                    if (b.isCloud) continue;
                    fruit.push({ x: b.x, y: b.y, tier: b.tier, diameter: b.diameter });
                }
                return {
                    width: CANVAS_W,
                    height: CANVAS_H,
                    lossLineY: lossLine.y,
                    holdingTier: cloudBall ? cloudBall.tier : null,
                    nextTier: nextBall.tier,
                    cloudX: cloud.x,
                    canDrop,
                    isGameOver,
                    score,
                    fruit,
                };
            }

            /* ── Wire up controls ─────────────────────────────────────────── */

            if (domShakeBtn)   domShakeBtn.addEventListener("click", shakeClicked);
            // Restart immediately, no confirm dialog. reset() clears the saved
            // game but leaves the high score alone, so nothing is lost that a
            // prompt would need to protect.
            if (domNewGame)    domNewGame.addEventListener("click", reset);
            if (domPlayAgain)  domPlayAgain.addEventListener("click", reset);

            // Expose the bits the controller needs.
            api = {
                id: cfg.id,
                reset,
                drop,
                getState: buildState,
                getScore: () => score,
                setPolicy(fn) { policy = fn; },
            };
        };

        new p5(sketch, cfg.container);
        return () => api;
    }

    /* ── Boot both boards ─────────────────────────────────────────────────── */

    const getHuman = createBoard({
        id: "human",
        container: document.getElementById("canvas-human"),
        interactive: true,
        persist: true,
    });

    const getAI = createBoard({
        id: "ai",
        container: document.getElementById("canvas-ai"),
        interactive: false,
        persist: false,
    });

    window.watermelonBoards = {
        get human() { return getHuman(); },
        get ai()    { return getAI(); },
    };

    /* ── AI opponent ──────────────────────────────────────────────────────

       The encoder below MUST mirror watermelon_env.py's _get_obs()
       field-for-field, in the same order. If it drifts, inference silently
       produces nonsense rather than failing — the same class of bug as the
       Tetris 210-vs-238 mismatch.

         grid  GRID_H x GRID_W x 4, row-major (row, col, channel)
           0 occupancy               1 if the cell CENTRE falls inside a fruit
           1 (tier+1)/(MAX_TIER+1)   that fruit's normalised tier
           2 merges with held        1 where the fruit matches the held tier
           3 merges with next        1 where the fruit matches the next tier
         scalars (12)
           5 held tier one-hot   (spawn pool is tiers 0-4)
           5 next tier one-hot
           1 stack height as a fraction of the board
           1 fruit count / MAX_FRUIT_NORM

       30*22*4 + 12 = 2652 floats -> 48 action logits, one per drop column.

       Channel 1 is (tier+1)/(MAX_TIER+1), NOT tier/MAX_TIER. The old form
       encoded a tier-0 fruit as 0.0 — identical to empty space — so telling
       the smallest fruit from a gap required cross-referencing channel 0, and
       the smallest fruit are exactly the ones the policy failed to pair.

       Channels 2 and 3 answer "where can I merge?" directly instead of making
       the network infer it from channel 1. They are derived, not new
       information, but they are the question the policy actually has to
       answer on every single drop.

       48 columns, not 24: the board is 448px wide, so 24 columns step 18.7px
       while the smallest fruit is 24px across — the action grid was coarser
       than the thing it had to line up with. */

    const GRID_W = 22, GRID_H = 30, GRID_CHANNELS = 4, N_SCALARS = 12;
    const N_DROP_COLUMNS = 48;
    const SPAWN_TIERS = 5;
    const MAX_FRUIT_NORM = 60;
    const OBS_SIZE = GRID_W * GRID_H * GRID_CHANNELS + N_SCALARS;

    function buildObservation(state) {
        const obs = new Float32Array(OBS_SIZE);
        const cellW = state.width / GRID_W;
        const cellH = state.height / GRID_H;
        // The env's held_tier is always a real tier, so null (no fruit in hand
        // between drops) falls back to 0 here — the same substitution the
        // one-hot below already makes, kept identical so the two fields can
        // never disagree about what is being held.
        const heldTier = state.holdingTier === null ? 0 : state.holdingTier;

        for (const f of state.fruit) {
            const r = f.diameter / 2;
            // Math.trunc, not Math.floor: Python's int() truncates toward
            // zero, and the max(0, ...) below relies on that for negatives.
            const colLo = Math.max(0, Math.trunc((f.x - r) / cellW));
            const colHi = Math.min(GRID_W - 1, Math.trunc((f.x + r) / cellW));
            const rowLo = Math.max(0, Math.trunc((f.y - r) / cellH));
            const rowHi = Math.min(GRID_H - 1, Math.trunc((f.y + r) / cellH));

            for (let row = rowLo; row <= rowHi; row++) {
                const py = (row + 0.5) * cellH;
                for (let col = colLo; col <= colHi; col++) {
                    const px = (col + 0.5) * cellW;
                    const dx = px - f.x, dy = py - f.y;
                    if (dx * dx + dy * dy <= r * r) {
                        const base = (row * GRID_W + col) * GRID_CHANNELS;
                        obs[base] = 1;
                        obs[base + 1] = (f.tier + 1) / (MAX_TIER + 1);
                        if (f.tier === heldTier)        obs[base + 2] = 1;
                        if (f.tier === state.nextTier)  obs[base + 3] = 1;
                    }
                }
            }
        }

        let p = GRID_W * GRID_H * GRID_CHANNELS;
        const held = state.holdingTier === null ? 0 : state.holdingTier;
        obs[p + Math.min(held, SPAWN_TIERS - 1)] = 1;
        p += SPAWN_TIERS;
        obs[p + Math.min(state.nextTier, SPAWN_TIERS - 1)] = 1;
        p += SPAWN_TIERS;

        // Highest point of any fruit; the empty board reads as the floor.
        let top = state.height;
        for (const f of state.fruit) top = Math.min(top, f.y - f.diameter / 2);
        obs[p] = Math.max(0, Math.min(1, 1 - top / state.height));
        obs[p + 1] = Math.min(1, state.fruit.length / MAX_FRUIT_NORM);

        return obs;
    }

    /* ── Neural-network inspector ─────────────────────────────────────────
       The grid is stored interleaved — (row * GRID_W + col) * GRID_CHANNELS
       + channel — so readCell has to index it the same way buildObservation
       writes it. Getting this wrong would draw a plausible-looking but wrong
       picture, which is worse than drawing nothing. */
    const SPAWN_TIER_NAMES = ["cherry", "strawberry", "grape", "dekopon", "orange"];
    const inspector = createInspector({
        mount: document.getElementById("insp-mount-ai"),
        grid: {
            w: GRID_W,
            h: GRID_H,
            channels: [
                { label: "occupancy", hint: "1 where a fruit covers the cell" },
                { label: "tier", hint: "fruit size in that cell, 0-1" },
                { label: "merges with held", hint: "drop here and it combines" },
                { label: "merges with next", hint: "where the fruit after this one pairs" },
            ],
        },
        readCell: (obs, row, col, ch) => obs[(row * GRID_W + col) * GRID_CHANNELS + ch],
        actions: {
            count: N_DROP_COLUMNS,
            orientation: "columns",
            label: (i) => String(i),
        },
        scalars: (obs) => {
            const p = GRID_W * GRID_H * GRID_CHANNELS;
            const oneHot = (off) => {
                for (let i = 0; i < SPAWN_TIERS; i++) if (obs[off + i]) return SPAWN_TIER_NAMES[i];
                return "—";
            };
            return [
                { label: "holding", value: oneHot(p) },
                { label: "next", value: oneHot(p + SPAWN_TIERS) },
                { label: "stack height", value: (obs[p + 2 * SPAWN_TIERS] * 100).toFixed(0) + "%" },
                { label: "fruit", value: Math.round(obs[p + 2 * SPAWN_TIERS + 1] * MAX_FRUIT_NORM) },
            ];
        },
        valueLabel: "expected score from here",
        valueHint: "the critic's estimate, in reward units (~score / 10)",
        onReveal: loadCritic,
    });

    /* The critic is a SEPARATE 22.6 MB model, loaded the first time the panel
       is opened and never otherwise. Bundling it into the playing model would
       double what every visitor downloads (22.6 -> 45.3 MB measured) to power
       a readout inside a panel that is closed by default.

       Under file:// this fetch is blocked, so the value readout simply stays
       hidden — the rest of the inspector works either way. */
    let criticSession = null;
    let criticPending = null;

    /* watermelon_critic.onnx is the value head of the SHIPPED policy, and the
       checkpoint switcher can put an earlier policy in play. The ladder rungs
       carry no critic of their own (another 22.6 MB each for one readout), and
       showing the shipped critic's opinion of an earlier policy's position
       would be a confident number from a network that never saw those weights.
       So the readout is suppressed while an earlier rung is selected. */
    let criticMatchesModel = true;

    function loadCritic() {
        if (!criticMatchesModel) return Promise.resolve();
        if (criticSession || criticPending) return criticPending;
        criticPending = ort.InferenceSession
            .create("watermelon_critic.onnx", { executionProviders: ["wasm"] })
            .then((s) => { criticSession = s; })
            .catch((err) => {
                console.warn("Watermelon critic unavailable — value readout hidden.", err);
            });
        return criticPending;
    }

    const aiStatusEl = document.getElementById("status-ai");
    let aiSession = null;
    let aiBusy = false;          // an inference is in flight
    let aiTargetFrac = null;     // where this drop is aimed, 0-1 of board width
    const AI_ALIGN_TOLERANCE = 6; // px; the cloud eases in, so wait for it

    /* Two ways in, and which one is used matters a great deal for load time.

       model_data.js embeds the model as a base64 string in a <script> tag.
       That is the only thing that works under file://, but it is a ~30 MB
       RENDER-BLOCKING script in <head>: the browser paints nothing at all —
       a blank white page — until the whole file has downloaded and parsed.
       Base64 also inflates the model by a third.

       Over HTTP we fetch the .onnx instead. It is smaller, it downloads in
       parallel with the page, and the human board is playable immediately
       while the AI's "Awaiting model" overlay clears on its own.

       So: use the embedded copy only when it is actually there (local
       file:// use), and otherwise fetch. The deployed site ships the .onnx
       and no model_data.js — see tools/deploy_pages.sh. */
    async function loadAiModel() {
        try {
            ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

            let src;
            if (typeof WATERMELON_MODEL_B64 !== "undefined") {
                const raw = atob(WATERMELON_MODEL_B64);
                src = new Uint8Array(raw.length);
                for (let i = 0; i < raw.length; i++) src[i] = raw.charCodeAt(i);
            } else {
                src = "watermelon_ai.onnx";
            }

            aiSession = await ort.InferenceSession.create(src, {
                executionProviders: ["wasm"],
            });
            if (aiStatusEl) aiStatusEl.hidden = true;
            mountCheckpointSwitcher();
        } catch (err) {
            console.error("Failed to load Watermelon AI model:", err);
            if (aiStatusEl) aiStatusEl.querySelector(".overlay-sub").textContent =
                "model failed to load";
        }
    }

    /* Mounted under the AI board once the shipped model is live. Compact here:
       this column already carries a score bubble, the board, and the speed
       controls, and the "scroll down to see inside" cue sits just below — a
       full-height control was what clipped it before. */
    function mountCheckpointSwitcher() {
        if (typeof CheckpointSwitcher === "undefined") return;
        const col = document.querySelector("#board-ai .board-column");
        if (!col) return;
        const sw = CheckpointSwitcher.mount({
            game: "watermelon",
            container: col,
            initial: aiSession,
            onSession: (session, rung) => {
                aiSession = session;
                criticMatchesModel = rung.shipped;
                if (!rung.shipped) {
                    criticSession = null;
                    criticPending = null;
                } else if (inspector.isOpen) {
                    loadCritic();
                }
            },
        });
        if (sw) col.querySelector(".ckpt").classList.add("ckpt--compact");
    }

    async function chooseColumn(state) {
        const obs = buildObservation(state);
        if (obs.length !== OBS_SIZE) {
            console.error(
                `Watermelon observation is ${obs.length} floats, expected ${OBS_SIZE}. ` +
                "buildObservation() and watermelon_env._get_obs() have drifted apart."
            );
        }
        const tensor = new ort.Tensor("float32", obs, [1, obs.length]);
        const out = await aiSession.run({ observation: tensor });
        const logits = out.action_logits.data;

        // The inspector gets the very tensor that was just fed to the model,
        // not a re-derivation of it — so what it draws cannot drift from what
        // the network actually saw.
        if (inspector.isOpen) {
            let value;
            if (criticSession) {
                const v = await criticSession.run({ observation: tensor });
                value = v.value.data[0];
            }
            inspector.update({ obs, logits, value });
        }

        // Difficulty — see settings.js. Full strength is the original argmax.
        if (typeof Settings !== "undefined") return Settings.chooseAction("watermelon", logits);
        let best = 0;
        for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }

    // The board's policy hook is called every frame and must return
    // synchronously, so it just reports the current target. Inference and the
    // decision to release are driven from the loop below.
    function driveAI() {
        const ai = window.watermelonBoards.ai;
        if (!aiSession || !ai) return;

        const state = ai.getState();
        if (state.isGameOver) { aiTargetFrac = null; return; }

        // No target yet for this fruit — pick one.
        if (aiTargetFrac === null && state.canDrop && !aiBusy) {
            aiBusy = true;
            chooseColumn(state)
                .then(col => { aiTargetFrac = (col + 0.5) / N_DROP_COLUMNS; })
                .catch(err => console.error("Watermelon AI inference failed:", err))
                .finally(() => { aiBusy = false; });
            return;
        }

        // Aimed and lined up — release, then wait for the next fruit.
        if (aiTargetFrac !== null && state.canDrop) {
            const targetX = aiTargetFrac * state.width;
            if (Math.abs(state.cloudX - targetX) <= AI_ALIGN_TOLERANCE) {
                ai.drop();
                aiTargetFrac = null;
            }
        }
    }

    // ── AI speed buttons ─────────────────────────────────────────────────
    const speedBox = document.getElementById("speed-ai");
    if (speedBox) {
        speedBox.addEventListener("click", (e) => {
            const btn = e.target.closest("button[data-speed]");
            if (!btn) return;
            const v = parseFloat(btn.getAttribute("data-speed"));
            if (!isFinite(v) || v <= 0) return;
            AI_SPEED = v;
            speedBox.querySelectorAll("button[data-speed]").forEach((b) =>
                b.classList.toggle("active", b === btn)
            );
        });
    }

    window.addEventListener("load", async () => {
        await loadAiModel();
        const ai = window.watermelonBoards.ai;
        if (ai && aiSession) {
            ai.setPolicy(() => aiTargetFrac);   // steer the cloud
            // Poll faster than the quickest cooldown (2x -> 500ms) so a drop
            // is never delayed by the polling interval itself.
            setInterval(driveAI, 100);          // think + release
        }
    });
})();

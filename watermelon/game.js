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
      behind the `store` helpers below. Only the human BOARD persists —
      `cfg.persist` gates it. Both boards keep a high score, each under its
      own key (watermelon.<id>.highScore), so the AI's can never overwrite
      yours; the AI's used to reset on every reload.

   Game-over no longer reloads the page. With two boards on one document a
   reload would reset both, so each board resets itself in place instead.

   ── Wiring an AI ──
   The AI board steers its cloud by a policy function on the controller:

       boards.ai.setPolicy(() => 0.5);

   It returns either a number in 0..1 (aim at that fraction of board width)
   or null to keep holding. Releasing is driven separately, by driveAI().

   ── Head-to-head ──
   A match starts with the human's first drop of a game. At that moment the
   AI board resets and starts too — it never plays ahead of you. The rules
   (the AI's first life is its match score, and losing first means waiting
   for it to pass you or lose) are shared/results.js's MatchResults.begin().
   See "Match" near the bottom of this file.
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

    /* ── Board colour ─────────────────────────────────────────────────────
       Deliberately NOT themed, unlike Snake's and Tetris's boards.

       Those two draw flat-coloured shapes, so a board that inverts with the
       palette still reads. Watermelon draws eleven illustrated fruit that were
       all painted against this cream, and the well is the lit surface they sit
       on — the game's whole look is a bright playfield inside a dark page.

       Wiring it to --board-bg produced a #000000 board on the #080808 default
       page: the play area vanished, leaving fruit floating in the middle of
       nothing. The theme still reaches this page through the palette around
       the board and the animated background behind it, which is enough. */
    const BOARD_BG = "#FDFBEC";

    const WEIGHTED = {
        initGame: [0, 1, 2, 3, 4],
        midGame:  [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
        endGame:  [0, 1, 2, 2, 2, 3, 3, 3, 4, 4],
    };

    const CANVAS_W = 448;
    const CANVAS_H = 599;
    const LOSS_LINE_Y = 115;   // stack above this and the game ends

    /* Drop positions are clamped the way watermelon_env.py clamps them: the
       fruit's own radius plus 1px from each wall. The browser used a fixed 5px
       margin, so a large fruit could be released half inside a wall and pushed
       sideways by the physics, from a position the AI never trained on. */
    function clampDropX(x, tier) {
        const r = DIAMETERS[tier] / 2;
        return Math.min(CANVAS_W - r - 1, Math.max(r + 1, x));
    }

    const BALL_TIMEOUT      = 1000;

    // AI speed multiplier, matching Tetris's speed buttons. Applies to the AI
    // board only — the human board's pacing is the player's business.
    //
    // Two things have to scale together, or the board fights itself: the
    // cooldown between drops, and how fast the cloud slides to its target. At
    // 2x with unchanged easing the AI would be ready to drop before the cloud
    // had arrived, and just stall on the alignment check.
    // 1.75x was dropped: all three games offer the same six speeds.
    const AI_SPEEDS = [0.25, 0.5, 0.75, 1, 1.5, 2];
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
        const domStartHint     = el("starthint");
        const domResult        = el("result");
        const domLive          = el("live");

        const KEY_HIGH    = `watermelon.${cfg.id}.highScore`;
        const KEY_SAVED   = `watermelon.${cfg.id}.savedGame`;
        const KEY_DROPPED = `watermelon.${cfg.id}.ballsDropped`;

        let policy = null;   // AI hook — null means "do not act"
        let api    = null;   // controller returned to the caller

        const sketch = (p) => {
            let balls, bounds, loss;
            let wall1, wall2, ground, lossLine;
            let nextBall, cloud, cloudBall;
            // Where the held fruit is, and where it will drop. The cloud
            // SPRITE follows it but is clamped inside the walls when drawn:
            // aimed at the far edge, the drop x is right against the wall and
            // the cloud, centred on it, was cut in half by the canvas edge.
            let cloudX = CANVAS_W / 2;

            let score        = 0;
            let highScore    = 0;
            let ballsDropped = 0;
            let canDrop      = false;
            let isGameOver   = false;
            let loading      = true;
            // True once setup() has built the board. The model now loads in
            // parallel with the page, and a cached one can be ready before p5
            // has even started — buildState() would then throw on every poll.
            let ready        = false;

            let numOfShakes   = 0;
            // Bumped by reset(). Timers and awaits started before a restart
            // compare against it and stand down instead of acting on the new board.
            let gen = 0;
            // p.frameCount at the last drop. Physics steps in draw(), which the
            // browser stops calling in a background tab; the drop cooldown is a
            // timer, which keeps running. The AI used to keep dropping while
            // nothing fell, piling fruit on the spawn point until it lost the
            // moment frames resumed.
            let dropFrame = -Infinity;
            let shakeCountdownTimer = null, shakeCooldownTimer = null;
            let canShake      = true;
            let doShake       = false;
            // Frames of shaking left. Counted in frames because the physics
            // steps per frame; it also replaced an `await p.delay(2000)` that
            // doEarthquake() started on EVERY frame of the shake, ~120 pending
            // timers per press, and that a restart could not cancel.
            let shakeFrames   = 0;
            const SHAKE_FRAMES = 120;   // 2 s at p5's 60 fps cap
            // Where the human is aiming, in canvas px (clamped to the held
            // fruit at use). Set by the mouse, a touch, or the arrow keys.
            // Starts centred: p5 reports mouseX as 0 until the pointer has
            // actually moved, which used to slide the cloud into the left wall.
            let aimX          = CANVAS_W / 2;
            // Human board: true from the first drop of a game. Until then the
            // start hint shows, and the match has not begun.
            let started       = false;
            // This game was restored from a save after a reload (see Match).
            let restored      = false;
            let gameOverAt    = 0;    // performance.now() at game over

            // Decoded p5.Image objects, one set per instance.
            let FRUIT_IMG   = [];
            let CLOUD_IMAGE = null;
            let cloudHalfW  = 60;

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

                // Walls and floor are physics boundaries only, so they are
                // never drawn (p5play outlined them with the sprite stroke).
                wall1 = new bounds.Sprite(0, CANVAS_H / 2, 1, CANVAS_H, "s");
                wall1.color = BOARD_BG;
                wall1.visible = false;
                wall2 = new bounds.Sprite(CANVAS_W, CANVAS_H / 2, 1, CANVAS_H, "s");
                wall2.color = BOARD_BG;
                wall2.visible = false;

                ground = new bounds.Sprite(CANVAS_W / 2, CANVAS_H, CANVAS_W * 2, 1, "s");
                ground.color = BOARD_BG;
                ground.visible = false;
                ground.bounciness = 0;

                // Collision sensor only — it is never drawn. Setting .stroke
                // on a static sprite renders nothing (verified: the sprite
                // reports visible but paints no pixels), so the visible red
                // line is drawn with p.line() in draw() instead.
                lossLine = new loss.Sprite(CANVAS_W / 2, LOSS_LINE_Y, CANVAS_W, 1, "s");
                lossLine.visible = false;

                nextBall = new p.Sprite(CANVAS_W - 100, 100);
                nextBall.tier = Math.floor(p.random(0, 5));
                nextBall.collider = "n";
                nextBall.diameter = DIAMETERS[nextBall.tier];
                nextBall.img = FRUIT_IMG[nextBall.tier];
                nextBall.visible = false;
                renderNextBall();

                cloud = new p.Sprite(CANVAS_W / 2, 50, 75, 50, "n");
                cloud.img = CLOUD_IMAGE;
                cloud.scale = 0.8;
                // Half the drawn cloud (148px image at 0.8), plus a pixel.
                cloudHalfW = Math.ceil(CLOUD_IMAGE.width * 0.8 / 2) + 1;

                balls = new p.Group();
                balls.tier;
                balls.isCombining = false;
                balls.isCloud;
                balls.rotationDrag = 0.7;
                balls.textColor = "white";
                balls.textSize = 24;
                balls.bounciness = 0;

                balls.collide(balls, combineFruits);

                createCloudBall(cloudX, cloud.y, Math.floor(p.random(0, 5)));

                canDrop = true;
                highScore = store.read(KEY_HIGH, 0) || 0;
                ballsDropped = cfg.persist ? (store.read(KEY_DROPPED, 0) || 0) : 0;
                renderScore();

                // p5play makes every canvas a tab stop (tabIndex 0) with no
                // name. The human's is a real control — keys work while it
                // is focused, like anywhere else on the page — so it keeps
                // the stop and gets a name; the AI's is only a picture.
                const cv = cfg.container && cfg.container.querySelector("canvas");
                if (cv) {
                    cv.setAttribute("role", "img");
                    cv.setAttribute("aria-label", cfg.interactive
                        ? "Your board. Arrows or A and D to aim, Space to drop."
                        : "The AI's board");
                    cv.tabIndex = cfg.interactive ? 0 : -1;
                }

                if (cfg.persist && store.read(KEY_SAVED)) loadSavedGame();
                else loading = false;
                ready = true;
            };

            /* ── Draw ─────────────────────────────────────────────────────── */

            p.draw = () => {
                p.background(BOARD_BG);

                // The human board's cloud tracks the pointer; the AI board's
                // tracks whatever its policy asks for, and parks centre when
                // there is no policy yet.
                let targetX;
                const heldTier = cloudBall ? cloudBall.tier : 0;
                if (cfg.interactive) {
                    // Held arrow keys slide the aim; ~1 s wall to wall.
                    if (keyDir) aimX = Math.max(0, Math.min(CANVAS_W, aimX + keyDir * KEY_AIM_PX));
                    targetX = clampDropX(aimX, heldTier);
                } else {
                    // The policy only reports a stored target, so no full state
                    // snapshot is built for it every frame any more.
                    const want = policy ? policy() : null;
                    // No target between drops: hold where the last fruit went.
                    // Parking at the centre instead swung the cloud back to the
                    // middle after every single drop and then out again to the
                    // next column, a constant pendulum on the AI board.
                    targetX = !policy ? CANVAS_W / 2
                            : want === null || want === undefined ? clampDropX(cloudX, heldTier)
                            : clampDropX(want * CANVAS_W, heldTier);
                }
                // The AI board's cloud tracks at the selected speed; the human
                // board is never sped up. Eases a fraction of the remaining
                // distance per frame, as the sprite's moveTowards() did.
                // Capped below 1: at >=1 it teleports and the motion reads as
                // a jump cut.
                const ease = cfg.interactive ? CLOUD_EASE_BASE
                                             : Math.min(0.9, CLOUD_EASE_BASE * AI_SPEED);
                cloudX += (targetX - cloudX) * ease;
                cloud.vel.x = 0;
                cloud.x = Math.min(CANVAS_W - cloudHalfW, Math.max(cloudHalfW, cloudX));

                if (cloudBall) {
                    cloudBall.x = cloudX;
                    cloudBall.y = cloud.y + 50;
                }

                if (cfg.interactive && p.kb.pressed("e")) shakeClicked();
                doEarthquake();

                for (const x of balls) {
                    if (x.isCloud) continue;
                    if (x.overlapping(lossLine) > 60) gameOver();
                }

                // The drop guide, on the cream board.
                p.stroke("gray");
                p.strokeWeight(6);
                p.line(cloudX, cloud.y, cloudX, CANVAS_H);

                // Stack limit — fruit resting above this line ends the game.
                // Stays red on every theme: it is the one mark on the board
                // that means "you are about to lose", and a warning that
                // restyles itself per palette stops reading as a warning.
                p.stroke("red");
                p.strokeWeight(3);
                p.line(0, LOSS_LINE_Y, CANVAS_W, LOSS_LINE_Y);

                p.stroke("black");
                p.strokeWeight(1);
            };

            /* ── Input ────────────────────────────────────────────────────── */

            // Touch is handled by the pointer listeners below, not by p5.
            // With no touch handlers defined, p5 forwards touchend to
            // mouseReleased, so ANY touch that ended on the page dropped a
            // fruit — including the vertical swipe meant to scroll down to the
            // AI board. Empty handlers switch that fallback off.
            p.touchStarted = p.touchMoved = p.touchEnded = () => {};

            // A tap is followed by emulated mouse events; those must not aim
            // or drop a second time.
            let lastTouchAt = -Infinity;
            const fromTouch = (e) =>
                performance.now() - lastTouchAt < 800 ||
                !!(e && e.sourceCapabilities && e.sourceCapabilities.firesTouchEvents);

            p.mouseMoved = p.mouseDragged = (e) => {
                if (!fromTouch(e)) aimX = p.mouseX;
            };

            p.mouseReleased = (e) => {
                if (!cfg.interactive || fromTouch(e)) return;
                // p5 listens on the whole window, so a release on a control
                // near the board (Play Again sits ON it; on narrow layouts the
                // Restart/Shake row sits just under it) also arrived here.
                if (e && e.target && e.target.closest &&
                    e.target.closest("button, a, input, select, label")) return;
                // Only a release ON this board drops. The old check allowed
                // 25px outside the canvas, which took in the Score and Next
                // bubbles just above it: clicking either dropped a fruit.
                // The target also rules out the other board, the page around
                // it and a drag that started here and ended elsewhere.
                if (!e || !e.target || !cfg.container ||
                    !cfg.container.contains(e.target)) return;
                aimX = p.mouseX;
                dropAtAim();
            };

            /* Touch: a TAP drops, a swipe does not.

               The canvas is `touch-action: pan-y` (style.css), so a vertical
               swipe scrolls the page and the browser cancels the pointer — on
               a phone the board is nearly the full width of the screen, and
               dropping on every swipe left almost nowhere to scroll from. A
               mostly-sideways drag is the browser's to give us: it aims, and
               lifting the finger drops, as it always has.

               A press that does not move is a tap however long it lasts. It
               used to need to end within 500 ms, so a slow, deliberate press
               did nothing at all, which reads as a broken game. */
            const TAP_SLOP_PX = 10;
            let touch = null;

            // p5play (lib/physics.min.js, ~line 4122) calls preventDefault()
            // on every touchstart on the canvas, which switches off scrolling
            // for any gesture that starts on a board — touch-action alone
            // cannot undo that. Stopping the event in the capture phase on the
            // board's container keeps it from ever reaching that listener.
            // Nothing here needs touch events: aiming and dropping use the
            // pointer events below, and p5's own touch handlers are no-ops.
            // Both boards, so a swipe on the AI board scrolls too.
            if (cfg.container) {
                cfg.container.addEventListener("touchstart", (e) => {
                    if (e.target.closest && e.target.closest("button, a")) return;
                    e.stopPropagation();
                }, { capture: true, passive: true });
            }
            const canvasX = (clientX) => {
                const r = cfg.container.getBoundingClientRect();
                return r.width ? (clientX - r.left) * CANVAS_W / r.width : CANVAS_W / 2;
            };
            if (cfg.interactive && cfg.container) {
                const c = cfg.container;
                c.addEventListener("pointerdown", (e) => {
                    if (e.pointerType === "mouse") return;
                    if (e.target.closest && e.target.closest("button, a")) return;
                    touch = { id: e.pointerId, x0: e.clientX, y0: e.clientY,
                              t0: performance.now(), aiming: false };
                });
                c.addEventListener("pointermove", (e) => {
                    if (!touch || e.pointerId !== touch.id) return;
                    const dx = e.clientX - touch.x0, dy = e.clientY - touch.y0;
                    if (!touch.aiming && Math.abs(dx) > TAP_SLOP_PX && Math.abs(dx) > 2 * Math.abs(dy))
                        touch.aiming = true;
                    if (touch.aiming) aimX = canvasX(e.clientX);
                });
                c.addEventListener("pointerup", (e) => {
                    if (!touch || e.pointerId !== touch.id) return;
                    const t = touch;
                    touch = null;
                    lastTouchAt = performance.now();
                    const dx = e.clientX - t.x0, dy = e.clientY - t.y0;
                    const tap = Math.hypot(dx, dy) <= TAP_SLOP_PX;
                    if (!tap && !t.aiming) return;
                    aimX = canvasX(e.clientX);
                    dropAtAim();
                });
                c.addEventListener("pointercancel", () => {
                    touch = null;
                    lastTouchAt = performance.now();
                });
                // A long press would otherwise open the context menu (and on
                // Android cancel the pointer), so the drop never happened.
                // Only during a touch: a mouse right-click keeps its menu.
                c.addEventListener("contextmenu", (e) => {
                    if (touch || performance.now() - lastTouchAt < 800) e.preventDefault();
                });
            }

            // Drop where the player is aiming, not where the cloud has eased
            // to so far: on touch the cloud is still travelling when the
            // finger lifts, and each fruit used to land where you had tapped
            // the time before.
            function dropAtAim() {
                if (cloudBall && canDrop && !isGameOver && !loading) {
                    cloudX = clampDropX(aimX, cloudBall.tier);
                    cloudBall.x = cloudX;
                }
                drop();
            }

            /* Keyboard: ←/→ or A/D aim, Space, Enter or ↓ drop, E shakes
               (polled in draw()). Space/Enter on Game Over plays again.

               Keys that belong to a focused control stay with it, the same
               split shared/keyscroll.js makes: buttons and links own Space and
               Enter but have no use for arrows; text fields and selects own
               everything. */
            let keyDir = 0;
            const held = { left: false, right: false };
            const KEY_AIM_PX = 8;
            const syncKeyDir = () => { keyDir = (held.right ? 1 : 0) - (held.left ? 1 : 0); };
            if (cfg.interactive) {
                const ownsAll = /^(input|select|textarea|option)$/i;
                const ownsActivate = /^(button|a|summary)$/i;
                window.addEventListener("keydown", (e) => {
                    if (e.ctrlKey || e.metaKey || e.altKey) return;
                    const t = e.target;
                    if (t && (t.isContentEditable || ownsAll.test(t.tagName || ""))) return;
                    const k = e.key;
                    const left  = k === "ArrowLeft"  || k === "a" || k === "A";
                    const right = k === "ArrowRight" || k === "d" || k === "D";
                    const act   = k === " " || k === "Spacebar" || k === "Enter";
                    if (left || right) {
                        if (left) held.left = true; else held.right = true;
                        syncKeyDir();
                        return;
                    }
                    if (!act && k !== "ArrowDown") return;
                    if (act && t && (ownsActivate.test(t.tagName || "") ||
                                     (t.getAttribute && t.getAttribute("role") === "button"))) return;
                    if (isGameOver) {
                        // A fresh press only, and not straight away: a key
                        // still held (or mashed) from the last drop must not
                        // skip the Game Over screen.
                        if (act && !e.repeat && performance.now() - gameOverAt >= 700) {
                            e.preventDefault();
                            reset();
                        }
                        return;
                    }
                    if (e.repeat) return;
                    e.preventDefault();
                    dropAtAim();
                });
                window.addEventListener("keyup", (e) => {
                    const k = e.key;
                    if (k === "ArrowLeft"  || k === "a" || k === "A") held.left = false;
                    if (k === "ArrowRight" || k === "d" || k === "D") held.right = false;
                    syncKeyDir();
                });
                // A key released while the window was not focused never sends
                // keyup, which would leave the cloud sliding forever.
                window.addEventListener("blur", () => { held.left = held.right = false; syncKeyDir(); });
            }

            /* ── Core actions ─────────────────────────────────────────────── */

            function drop() {
                if (!canDrop || doShake || isGameOver || loading) return;
                if (!cloudBall) return;

                canDrop = false;
                dropFrame = p.frameCount;
                ballsDropped++;
                if (cfg.persist) store.write(KEY_DROPPED, ballsDropped);

                const ball = cloudBall;
                cloudBall = undefined;

                const first = !started;
                if (first) {
                    started = true;
                    if (domStartHint) domStartHint.hidden = true;
                }
                if (cfg.onDrop) cfg.onDrop(first, restored);

                if (ballsDropped % DROPS_PER_SHAKE === 0) {
                    numOfShakes++;
                    if (domShakeCount) domShakeCount.innerText = numOfShakes;
                    // Not while the shake is cooling down: the button used to
                    // light up and then do nothing when pressed.
                    if (domShakeBtn) domShakeBtn.disabled = !canShake;
                }

                ball.collider = "d";
                ball.isCloud = false;
                ball.x = ball.x + p.random(-1, 1);
                ball.diameter = DIAMETERS[ball.tier];
                ball.bounciness = 0;
                ball.resetMass();

                const g = gen;
                setTimeout(() => {
                    // A restart during the cooldown bumps gen. Without this the
                    // old timer spawned a second held fruit on the fresh board,
                    // and the one it replaced hung at y~100 as an obstacle.
                    if (isGameOver || g !== gen) return;
                    createCloudBall(cloudX, cloud.y, nextBall.tier);
                    queueBall();
                    saveGame();
                    canDrop = true;
                    // AI board only: shorten the cooldown by the speed
                    // multiplier. The human board keeps the original 1000ms.
                }, cfg.interactive ? BALL_TIMEOUT : BALL_TIMEOUT / AI_SPEED);
            }

            async function combineFruits(a, b) {
                // No scoring once the board is over. Physics keeps running under
                // the overlay, and merges kept raising the saved high score
                // while "Game Over" still showed the old one.
                if (isGameOver) return;
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
                if (cfg.onScore) cfg.onScore(score);

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

                const g = gen;
                await p.delay(100);
                // Restarted while the pair was animating together: the fresh
                // board must not receive the merged fruit.
                if (g !== gen) return;

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

            function doEarthquake() {
                if (!doShake) return;
                if (--shakeFrames < 0) { doShake = false; return; }
                for (const ball of balls) {
                    if (ball.isCloud) continue;
                    ball.moveTowards(
                        ball.x + p.random(-SHAKE_STRENGTH, SHAKE_STRENGTH),
                        ball.y + p.random(-SHAKE_STRENGTH, SHAKE_STRENGTH)
                    );
                }
            }

            function shakeClicked() {
                if (!canShake || numOfShakes < 1 || isGameOver) return;

                let remaining = SHAKE_COOLDOWN_S;
                numOfShakes--;
                if (domShakeCount) domShakeCount.innerText = numOfShakes;
                doShake = true;
                shakeFrames = SHAKE_FRAMES;
                canShake = false;

                if (domShakeBtn) domShakeBtn.disabled = true;
                if (domShakeCountdown) {
                    domShakeCountdown.style.display = "";
                    domShakeCountdown.innerText = remaining;
                }

                const countdown = shakeCountdownTimer = setInterval(() => {
                    if (remaining < 0) return clearInterval(countdown);
                    if (domShakeCountdown) domShakeCountdown.innerText = --remaining;
                }, 1000);

                shakeCooldownTimer = setTimeout(() => {
                    clearInterval(countdown);
                    if (domShakeCountdown) domShakeCountdown.style.display = "none";
                    canShake = true;
                    if (numOfShakes > 0 && domShakeBtn) domShakeBtn.disabled = false;
                }, SHAKE_COOLDOWN_S * 1000);
            }

            /* ── Persistence ──────────────────────────────────────────────── */

            function saveGame() {
                // Not before the saved board has been restored (that would
                // overwrite it with an empty one), and not after game over,
                // which has already cleared it on purpose.
                if (!cfg.persist || loading || isGameOver) return;
                const state = { balls: [], cloudTier: null, nextTier: null,
                                score: 0, dropped: 0, shakes: 0 };
                for (const a of balls) {
                    if (a.isCloud) continue;
                    state.balls.push({
                        x: a.x, y: a.y, tier: a.tier,
                        diameter: a.diameter, vel: { x: a.vel.x, y: a.vel.y },
                    });
                }
                // Saved during the post-drop cooldown (now possible, since the
                // board is also saved when the page is hidden) there is no held
                // fruit yet: the pending timer would have promoted the queued
                // one. Record that, rather than restoring a random fruit in
                // hand and the same queued one again.
                state.cloudTier = cloudBall ? cloudBall.tier : nextBall.tier;
                state.nextTier  = cloudBall ? nextBall.tier : null;
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
                    createCloudBall(cloudX, cloud.y, state.cloudTier);
                }
                if (typeof state.nextTier === "number") queueBall(state.nextTier);
                else queueBall();
                if (typeof state.score === "number") {
                    score = state.score;
                    // The high score is stored separately; never show a lower
                    // best than the score already on the board.
                    if (score > highScore) { highScore = score; saveHighScore(score); }
                    renderScore();
                }
                if (typeof state.dropped === "number") ballsDropped = state.dropped;
                if (typeof state.shakes === "number") {
                    numOfShakes = state.shakes;
                    if (domShakeCount) domShakeCount.innerText = numOfShakes;
                    if (domShakeBtn) domShakeBtn.disabled = numOfShakes < 1;
                }
                // A game already under way, not a fresh one — see Match. Its
                // first drop was long ago, so no "Click to drop" prompt.
                restored = (state.balls && state.balls.length > 0) || score > 0;
                if (restored && domStartHint) domStartHint.hidden = true;
                loading = false;
            }

            // Both boards, each under its own key. Only the board itself is
            // gated on cfg.persist.
            function saveHighScore(value) {
                store.write(KEY_HIGH, value);
            }

            /* ── Game over / reset ────────────────────────────────────────── */

            function gameOver() {
                if (isGameOver || doShake || loading) return;
                isGameOver = true;
                canDrop = false;
                gameOverAt = performance.now();
                // Never printed over the Game Over card (a game that ended
                // before its first drop counted, e.g. a restored one).
                if (domStartHint) domStartHint.hidden = true;

                // gameOver() is guarded against re-entry, so this fires once
                // per game. The human board's hook ends its side of the match
                // and hands back the result line (or null when nothing was at
                // stake); later changes arrive through showResult().
                const result = cfg.onGameOver ? cfg.onGameOver(score, restored) : null;
                showResult(result);
                announce((cfg.interactive ? "Game over." : "The AI's game is over.") +
                         ` Score ${score}.` + (result && result.text ? " " + result.text : ""), true);

                store.clear(KEY_SAVED);
                // Otherwise closing the tab on this screen left the old drop
                // count behind, and the next fresh board opened on the late-game
                // fruit mix from its very first drop.
                if (cfg.persist) store.write(KEY_DROPPED, 0);
                if (domGameOverScore) domGameOverScore.textContent = `Score ${score}`;
                if (domGameOver) domGameOver.hidden = false;

                // The AI board had no way back: no button and nothing to reset
                // it, so every later human game was recorded against its frozen
                // score. It now restarts itself, as Snake's and Tetris's do.
                if (!cfg.interactive) {
                    const g = gen;
                    setTimeout(() => { if (isGameOver && g === gen) reset(); }, 1500);
                }
            }

            // The match line on the Game Over card. { text, note } or null.
            function showResult(result) {
                if (!domResult) return;
                const text = result && result.text ? result.text : "";
                domResult.textContent = text;
                domResult.classList.toggle("is-note", !!(result && result.note));
                domResult.hidden = !text;
            }

            // Resets this board only. The original reloaded the document,
            // which would take the other board down with it.
            function reset() {
                gen++;
                clearInterval(shakeCountdownTimer);
                clearTimeout(shakeCooldownTimer);
                for (const b of [...balls]) b.remove();
                cloudBall = undefined;

                score = 0;
                ballsDropped = 0;
                numOfShakes = 0;
                canShake = true;
                doShake = false;
                shakeFrames = 0;
                isGameOver = false;
                started = false;
                restored = false;
                if (domStartHint) domStartHint.hidden = false;

                store.clear(KEY_SAVED);
                if (cfg.persist) store.write(KEY_DROPPED, 0);

                queueBall(Math.floor(p.random(0, 5)));
                createCloudBall(cloudX, cloud.y, Math.floor(p.random(0, 5)));

                if (domGameOver) domGameOver.hidden = true;
                if (domShakeCount) domShakeCount.innerText = 0;
                if (domShakeBtn) domShakeBtn.disabled = true;
                if (domShakeCountdown) {
                    domShakeCountdown.innerText = "";
                    domShakeCountdown.style.display = "none";
                }

                renderScore();
                loading = false;
                dropFrame = -Infinity;
                canDrop = true;
                if (cfg.onReset) cfg.onReset();
            }

            /* ── Rendering to the DOM ─────────────────────────────────────── */

            function renderScore() {
                if (domScore) domScore.innerText = score;
                if (domHighScore) domHighScore.innerText = highScore;
                if (score > 0 && cfg.announceScore) announce(`Score ${score}`);
            }

            /* The visually hidden live line under each board. Scores change
               on every merge, often several per second in a chain, so they
               are coalesced and read at most once every couple of seconds;
               game over replaces anything pending and is read at once. */
            let liveTimer = null;
            function announce(text, now) {
                if (!domLive) return;
                clearTimeout(liveTimer);
                if (now) { domLive.textContent = text; return; }
                liveTimer = setTimeout(() => { domLive.textContent = text; }, 2000);
            }

            function renderNextBall() {
                if (!domNextBall) return;
                domNextBall.setAttribute("src", IMAGES[nextBall.tier]);
                domNextBall.setAttribute("alt", `Next fruit, tier ${nextBall.tier}`);
            }

            /* ── State snapshot handed to an AI policy ────────────────────── */

            function buildState() {
                // Before setup() there are no balls to list (it threw
                // "balls is not iterable"); say "not ready" instead.
                if (!ready) return null;
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
                    cloudX,
                    canDrop,
                    framesSinceDrop: p.frameCount - dropFrame,
                    isGameOver,
                    score,
                    fruit,
                };
            }

            /* ── Wire up controls ─────────────────────────────────────────── */

            if (domShakeBtn)   domShakeBtn.addEventListener("click", shakeClicked);
            // The board used to be saved only one second after each drop, so
            // closing the tab kept a snapshot of fruit still in mid-air and
            // lost any chain merge (and its score) that came after it. Save
            // again whenever the page is hidden — visibilitychange is the
            // event that reliably fires when a tab is actually closed.
            if (cfg.persist) {
                document.addEventListener("visibilitychange", () => {
                    if (document.hidden && ready) saveGame();
                });
                window.addEventListener("pagehide", () => { if (ready) saveGame(); });
            }
            // Restart immediately, no confirm dialog. reset() clears the saved
            // game but leaves the high score alone, so nothing is lost that a
            // prompt would need to protect.
            if (domNewGame)    domNewGame.addEventListener("click", () => {
                if (cfg.onRestartClick) cfg.onRestartClick();
                reset();
            });
            if (domPlayAgain)  domPlayAgain.addEventListener("click", reset);
            // "Tap to play again": anywhere on your Game Over card, not only
            // the button. Not straight away, for the same reason as the keys
            // below: a tap still landing from the last drop must not skip it.
            if (cfg.interactive && domGameOver) {
                domGameOver.addEventListener("click", (e) => {
                    if (e.target.closest && e.target.closest("button")) return;
                    if (isGameOver && performance.now() - gameOverAt >= 700) reset();
                });
            }

            // Expose the bits the controller needs.
            api = {
                id: cfg.id,
                reset,
                drop,
                getState: buildState,
                getScore: () => score,
                isStarted: () => started,
                isRestored: () => restored,
                // Bumped by every reset, so async callers can tell whether the
                // board they started on is still the one in play.
                getGen: () => gen,
                isReady: () => ready,
                setPolicy(fn) { policy = fn; },
                showResult,
                say: (text) => announce(text, true),
                isGameOver: () => isGameOver,
            };
        };

        new p5(sketch, cfg.container);
        return () => api;
    }

    /* ── Boot both boards ─────────────────────────────────────────────────── */

    /* ── Match ────────────────────────────────────────────────────────────
       The rules live in shared/results.js (MatchResults.begin), shared with
       Snake and Tetris so the three cannot drift apart again:

       - A match starts with the human's first drop of a game, if the model
         is loaded. The AI board resets at that moment and starts with you;
         before it, the AI waits.
       - The AI's match score is its FIRST life. If you lose first you have
         not won yet: your card reads "AI still playing, needs N to win" and
         follows it live until the AI passes you or loses. It used to freeze
         the AI's unfinished score the moment you lost, so losing early won.
         The AI keeps playing (and restarting) afterwards for show; later
         lives are not part of the match.
       - A game RESTORED after a reload is not a match. Your board comes back
         mid-game; the AI's does not, and there is no fair way to line a fresh
         AI up against a head start. You finish the game, the AI plays
         alongside for show, and the card says "Resumed game, not scored".
       - Restarting the AI or changing its speed voids the match; results.js
         itself voids it on a model-version or difficulty change.
       - A model that arrives mid-game joins on your next drop, unranked. */
    let aiRunning = false;       // the AI may drop fruit
    let match = null;            // the MatchResults match in play, if any
    let matchAiGen = -1;         // the AI board's gen during its first life
    let resumed = false;         // your current game was restored on load

    function startAiBoard() {
        const ai = window.watermelonBoards.ai;
        if (!ai || !ai.isReady()) return false;
        ai.reset();
        aiTargetFrac = null;
        aiRunning = true;
        if (aiStatusEl) aiStatusEl.hidden = true;
        return true;
    }

    // Waiting for your first drop: the AI holds still under its "AI ready" card.
    function pauseAiBoard() {
        aiRunning = false;
        aiTargetFrac = null;
        showAiWaiting();
    }

    function aiAvailable() { return !!aiSession && !aiStopped; }

    // The line for your Game Over card: { text, note } or null.
    function matchResult() {
        if (resumed) return { text: "Resumed game, not scored", note: true };
        if (!match) return null;
        const text = match.line();
        return text ? { text, note: match.state === "void" } : null;
    }

    function watchMatch(m) {
        m.onChange(() => {
            if (m !== match) return;
            const human = getHuman();
            if (!human) return;
            const settled = m.state === "done" || m.state === "void";
            if (human.isGameOver()) {
                // Live while the AI plays on, then the verdict.
                human.showResult(matchResult());
                if (settled) human.say(m.line());
            } else if (settled && !human.isStarted() && aiRunning) {
                // You already restarted and the AI has now settled the old
                // match: it waits for your first drop like any other game.
                pauseAiBoard();
            }
        });
    }

    const getHuman = createBoard({
        id: "human",
        container: document.getElementById("canvas-human"),
        interactive: true,
        persist: true,
        announceScore: true,
        onDrop(first, restored) {
            if (first) {
                // A new game: an earlier match still waiting on the AI's
                // first life is abandoned, not settled.
                if (match) match.cancel("a new game started");
                match = null;
                resumed = restored;
                const up = aiAvailable() && startAiBoard();
                if (up && !restored && typeof MatchResults !== "undefined") {
                    match = MatchResults.begin("watermelon", { speed: AI_SPEED });
                    matchAiGen = getAI().getGen();
                    watchMatch(match);
                }
            } else if (!aiRunning && aiAvailable()) {
                // The model arrived mid-game: the AI joins in, unranked.
                startAiBoard();
            }
        },
        onGameOver(score, restored) {
            // A restored game can end before its first drop (it came back
            // already over the line), so the board's own flag counts too.
            if (restored) resumed = true;
            if (match) match.humanDied(score);
            return matchResult();
        },
        onReset() {
            resumed = false;
            // Restarted mid-game: nothing to settle.
            if (match && match.state === "playing") match.cancel("a new game started");
            // Lost, and the AI is still on its first life: let it finish, so
            // the result is recorded. It pauses once it has (watchMatch).
            if (match && match.state === "waiting") return;
            match = null;
            pauseAiBoard();
        },
    });

    const getAI = createBoard({
        id: "ai",
        container: document.getElementById("canvas-ai"),
        interactive: false,
        persist: false,
        onScore(score) {
            if (match && getAI().getGen() === matchAiGen) match.aiScore(score);
        },
        onGameOver(score) {
            if (match && getAI().getGen() === matchAiGen) match.aiDied(score);
            return null;
        },
        onRestartClick() {
            if (match && !match.aiDone) match.cancel("the AI was restarted");
        },
    });

    window.watermelonBoards = {
        get human() { return getHuman(); },
        get ai()    { return getAI(); },
    };

    /* Read by shared/confirm-exit.js. In progress once the human board has any
       fruit or score, and not once it is already showing Game Over. */
    window.gameInProgress = function () {
        try {
            var st = getHuman().getState();
            return !!st && !st.isGameOver && ((st.fruit && st.fruit.length > 0) || st.score > 0);
        } catch (e) { return false; }
    };
    // The human board is saved after every drop and restored on return, so the
    // default "your run will be lost" would be untrue here. A restored game is
    // not a match (see Match), so say that too — the same words as Tetris.
    window.gameExitMessage = "Leave the game? Your board is saved and will be here when you come back, but a resumed game is not scored.";

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
            .then((s) => { if (criticMatchesModel) criticSession = s; criticPending = null; })
            .catch((err) => {
                console.warn("Watermelon critic unavailable — value readout hidden.", err);
            });
        return criticPending;
    }

    const aiStatusEl = document.getElementById("status-ai");
    let aiSession = null;
    let aiBusy = false;          // an inference is in flight
    let aiFailures = 0, aiStopped = false;

    // The AI board's status card: loading, failed, waiting for you, stopped.
    // The wording is shared with Snake and Tetris.
    function setAiStatus(title, sub) {
        if (!aiStatusEl) return;
        const t = aiStatusEl.querySelector(".overlay-title");
        const s = aiStatusEl.querySelector(".overlay-sub");
        if (t) t.textContent = title;
        if (s) s.textContent = sub;
        aiStatusEl.hidden = false;
    }

    // Model loaded, waiting for the human's first drop. Not shown while the
    // model is still loading or has failed: those cards say more.
    function showAiWaiting() {
        if (!aiAvailable()) return;
        setAiStatus("AI ready", "Starts with your first move");
    }

    // After repeated inference failures, say so on the board and stop retrying.
    function stopAi() {
        aiStopped = true;
        aiRunning = false;
        setAiStatus("AI stopped", "It stopped responding. Refresh to try again.");
    }
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
            ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/";

            // .onnx over HTTP, base64 only under file:// — see
            // shared/model-source.js.
            // Progress, so a 22 MB download does not read as "stuck" on
            // "Awaiting model". Passed fourth with dataUrl undefined, as Snake
            // does. The total is 0 without a usable Content-Length (e.g. a
            // compressed response), hence the MB form.
            const onProgress = (got, total) => {
                setAiStatus("Loading AI…", total > 0
                    ? "Downloading the model · " + Math.min(99, Math.floor(100 * got / total)) + "%"
                    : "Downloading the model · " + (got / 1048576).toFixed(0) + " MB");
            };
            const src = await modelSource("watermelon_ai.onnx", "WATERMELON_MODEL_B64",
                                          undefined, onProgress);
            setAiStatus("Loading AI…", "Starting the model…");

            aiSession = await ort.InferenceSession.create(src, {
                executionProviders: ["wasm"],
            });
            mountCheckpointSwitcher();
        } catch (err) {
            console.error("Failed to load Watermelon AI model:", err);
            setAiStatus("Couldn't load the AI", "Check your connection and refresh");
        }
    }

    /* Mounted in its own slot once the shipped model is live: beside the AI
       board under Restart on a wide screen, under the board's controls on a
       phone (see .board-body in style.css). */
    function mountCheckpointSwitcher() {
        if (typeof CheckpointSwitcher === "undefined") return;
        const col = document.getElementById("ckpt-ai");
        if (!col) return;
        CheckpointSwitcher.mount({
            game: "watermelon",
            container: col,
            initial: aiSession,
            onSession: (session, rung) => {
                aiSession = session;
                criticMatchesModel = rung.shipped;
                if (!rung.shipped) {
                    criticSession = null;
                    criticPending = null;
                } else if (inspector.isRevealed) {
                    // isRevealed, not isOpen: isOpen has a 400px warm-up margin
                    // and is true at scroll 0, which fetched the critic on load.
                    loadCritic();
                }
            },
        });
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
        // Not awaited: the critic run and redraw are for the panel only and
        // should never hold up the AI's move.
        if (inspector.isOpen) {
            const critic = criticSession;
            (critic ? critic.run({ observation: tensor }).then(v => v.value.data[0])
                    : Promise.resolve(undefined))
                .then(value => inspector.update({ obs, logits, value }))
                .catch(() => inspector.update({ obs, logits }));
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
        if (!aiSession || !aiRunning || !ai || aiStopped || !ai.isReady()) return;

        const state = ai.getState();
        if (state.isGameOver) { aiTargetFrac = null; return; }

        // No target yet for this fruit — pick one.
        // Wait for ~half a second of SIMULATED physics since the last drop, not
        // wall-clock time: frames only advance while the page is drawing, so
        // this also keeps the AI from observing a board that has not moved.
        const settled = state.framesSinceDrop >= 30;
        if (aiTargetFrac === null && state.canDrop && settled && !aiBusy) {
            aiBusy = true;
            const g = ai.getGen();
            chooseColumn(state)
                .then(col => {
                    aiFailures = 0;
                    // The board ended (or restarted) while this was in flight:
                    // the answer is for a position that no longer exists, and
                    // would have aimed the next board's first fruit.
                    if (g !== ai.getGen() || ai.getState().isGameOver) return;
                    aiTargetFrac = (col + 0.5) / N_DROP_COLUMNS;
                })
                .catch(err => {
                    // Used to retry every 100 ms forever, logging each time,
                    // with the board frozen and nothing on screen.
                    console.error("Watermelon AI inference failed:", err);
                    if (++aiFailures >= 5) stopAi();
                })
                .finally(() => { aiBusy = false; });
            return;
        }

        // Aimed and lined up — release, then wait for the next fruit.
        if (aiTargetFrac !== null && state.canDrop && settled) {
            // The same clamp the cloud uses. Near a wall the cloud would
            // otherwise stop short of an unclamped target and wait forever.
            const targetX = clampDropX(aiTargetFrac * state.width,
                                       state.holdingTier === null ? 0 : state.holdingTier);
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
            if (match) match.speed(v);
            speedBox.querySelectorAll("button[data-speed]").forEach((b) => {
                b.classList.toggle("active", b === btn);
                b.setAttribute("aria-pressed", b === btn ? "true" : "false");
            });
        });
    }

    /* Started as soon as this script runs, not on window "load". This file is
       deferred, so the DOM is already parsed; waiting for "load" only added
       every remaining subresource to the front of the model's critical path. */
    loadAiModel().then(() => {
        const ai = window.watermelonBoards.ai;
        if (ai && aiSession) {
            ai.setPolicy(() => aiTargetFrac);   // steer the cloud
            // Poll faster than the quickest cooldown (2x -> 500ms) so a drop
            // is never delayed by the polling interval itself.
            setInterval(driveAI, 100);          // think + release
            // Not playing yet: it waits for your first drop (see Match). If
            // you are already mid-game it joins on your next drop, unranked.
            const human = window.watermelonBoards.human;
            if (human && human.isStarted && human.isStarted()) {
                setAiStatus("AI ready", "Joins on your next drop");
            } else {
                showAiWaiting();
            }
        }
    });

    /* A mouse click leaves the button focused, and Space on a focused button
       belongs to the button — so after Restart or a speed button, Space pressed
       it again instead of dropping. Drop focus after a POINTER click; keyboard
       activation (detail 0) keeps it, so tabbing through still works. Same
       handler as Tetris and Snake. */
    const arenaEl = document.getElementById("arena");
    if (arenaEl) arenaEl.addEventListener("click", (e) => {
        if (e.detail === 0) return;
        const b = e.target.closest && e.target.closest("button");
        if (b) b.blur();
    });
})();

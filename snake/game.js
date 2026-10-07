(() => {
    "use strict";

    // ── Config (ported from the original app.js; board resized to fit the site) ──
    const TILE_COUNT = 16;
    const SPEED = 3;           // must evenly divide `scl` — see STEPS_PER_CELL below
    const START_LENGTH = 3;

    const canvas = document.getElementById("canvas-human");
    const ctx = canvas.getContext("2d");
    const restartBtn = document.getElementById("restart-human");
    const controlsOverlay = document.querySelector("#board-human .controls");

    const canvasAi = document.getElementById("canvas-ai");
    const ctxAi = canvasAi.getContext("2d");
    const restartAiBtn = document.getElementById("restart-ai");
    const aiStatusEl = document.getElementById("ai-status");

    // One visually hidden polite live line per board, for screen readers:
    // the canvases themselves say nothing. Cleared first so repeating the
    // same sentence is still announced.
    const srHuman = document.getElementById("sr-human");
    const srAi = document.getElementById("sr-ai");
    function announce(el, text) {
        if (!el) return;
        el.textContent = "";
        requestAnimationFrame(() => { el.textContent = text; });
    }

    /* The board's coordinate space, fixed at 480 whatever size it is shown at.
       The canvases' backing stores are sized separately (fitCanvas below) and
       scaled onto this, so the cell size — and therefore STEPS_PER_CELL and
       every observation the AI sees — never depends on the layout. scl used
       to be canvas.width / TILE_COUNT, which tied game speed to the bitmap. */
    const BOARD = 480;
    const scl = BOARD / TILE_COUNT;        // both boards render at the same scale
    const STEPS_PER_CELL = scl / SPEED;    // frames to cross one cell — must be a whole number

    // Cardinal directions, clockwise, matching snake_env.py's DIRS exactly —
    // this ordering is what the model was trained against, so it has to
    // line up index-for-index with the Python side.
    // 0=up, 1=right, 2=down, 3=left
    const DIRS = [
        { x: 0, y: -1 },
        { x: 1, y: 0 },
        { x: 0, y: 1 },
        { x: -1, y: 0 },
    ];
    const TURN_LEFT = 0, STRAIGHT = 1, TURN_RIGHT = 2;

    /* Two separate gates. matchStarted is the HUMAN's current game: false
       until the first input after load or after a restart, so a restarted
       board waits for the player exactly like the first one did (it used to
       stay true, and the new snake drove itself into the wall in ~1.3 s).
       aiActive is whether the AI board is moving at all; once started it keeps
       playing, and restarting itself, between the human's games. */
    let matchStarted = false;
    let aiActive = false;

    /* The head-to-head, run by MatchResults.begin() in shared/results.js so
       the rules are the same in all three games. A match begins on the
       human's first input; at that moment the AI board resets and starts with
       them, and only the AI's FIRST life in it is fed to the match. A human
       who dies first has not won: the match waits until the AI passes them or
       dies. This used to freeze the AI's unfinished score at the human's
       death, so crashing in the first second beat the AI 1-0, and the AI's
       restart button locked in a win the same way.

       Null when nothing is being counted: before the first input, or when the
       model was not loaded when the human began. Kept after the human's game
       ends, so the game-over card can show match.line() — live, while the AI
       is still playing — until the next game starts. */
    let match = null;
    let aiInMatch = false;         // the AI's current life is the match's first one
    let aiSpeed = 1;               // the AI SPEED control; see stepAi's accumulator

    // Game over shows after the death flash; a restart key is honoured only a
    // little later, so the key that was being mashed at the moment of death
    // cannot skip the final score.
    const GAME_OVER_SHOW_MS = 500;
    const RESTART_DELAY_MS = 700;

    // Phones and tablets: no hover, coarse pointer. Only changes prompt wording;
    // swipe controls are wired up regardless.
    const TOUCH = !!(window.matchMedia && window.matchMedia("(hover: none) and (pointer: coarse)").matches);

    class Segment {
        constructor(x, y, dir) {
            this.x = x;
            this.y = y;
            this.dir = dir;
        }
        collides(other) { return this.xx === other.xx && this.yy === other.yy; }
        get xx() { return Math.round(this.x / scl); }
        get yy() { return Math.round(this.y / scl); }
    }

    class Snake {
        constructor(x, y, length) {
            this.x = x;
            this.y = y;
            // Colours come from the theme at draw time (--snake-body /
            // --snake-dead in style.css); this only says which of the two.
            this.red = false;
            this.body = [];
            // Preset to rightward rather than zero: movement is gated
            // externally (matchStarted / aiActive), not by dir being non-zero, so
            // starting non-zero here means the frame counter below can
            // actually advance from tick one. Leaving this at {0,0} was
            // the root of the AI's deadlock — its first-ever decision
            // depends on frameCount advancing, which depends on the snake
            // already moving, which never happened.
            this.dir = { x: 1, y: 0 };
            this.newDir = { x: 1, y: 0 };
            this.pending = [];      // queued turns, consumed one per cell
            this.frameCount = 0;
            for (let n = 0; n < length; n++) {
                this.body.push(new Segment((this.x - n) * scl, this.y * scl, { x: 1, y: 0 }));
            }
        }

        // Absolute direction, exactly like the original — used directly by
        // the keyboard handler, and by the AI's applyAiAction() after it
        // translates a relative action into an absolute dx/dy.
        turn(dx, dy) {
            if (this.isDead) return;

            /* Queued, not overwritten.
       
               The snake only commits a turn at a cell boundary, which is every
               STEPS_PER_CELL frames. This used to validate the press against
               `this.dir` — the direction currently being travelled — and write
               a single `newDir` slot. Two consequences, both of which read as
               the controls being laggy or ignored:

                 - Travelling right, press Up then Right quickly to round a
                   corner. Up is accepted. Right is then tested against dir,
                   which is STILL right because the boundary has not arrived,
                   so it is rejected as a no-op and the second half of the turn
                   is silently dropped.
                 - Any second press inside one cell overwrote the first, so
                   fast inputs lost the earlier one instead of being played in
                   order.

               Validating against the LAST QUEUED direction instead, and
               keeping a short queue, means a sequence pressed faster than the
               snake moves is played back in order rather than discarded. The
               queue is capped at 2: enough for one corner, short enough that
               the snake never feels like it is running on rails. */
            const last = this.pending.length
                ? this.pending[this.pending.length - 1]
                : this.dir;

            let nx = 0, ny = 0;
            if (dx !== 0 && last.x === 0)      { nx = dx; ny = 0; }
            else if (dy !== 0 && last.y === 0) { ny = dy; nx = 0; }
            else return;                        // reversal or no change

            if (this.pending.length >= 2) return;
            this.pending.push({ x: nx, y: ny });
            // Kept in step so anything still reading newDir sees the queue head.
            this.newDir.x = this.pending[0].x;
            this.newDir.y = this.pending[0].y;
        }

        update() {
            if (this.isDead) return;
            if (this.dir.x === 0 && this.dir.y === 0 && this.newDir.x === 0 && this.newDir.y === 0) return;

            // At an exact cell boundary — resolve the turn and re-link every
            // segment's direction to the segment ahead of it. Using a frame
            // count instead of comparing floating-point pixel positions means
            // this always fires precisely, at any board/speed combination.
            if (this.frameCount % STEPS_PER_CELL === 0) {
                if (this.checkDeath()) return this.die();

                // Take the next queued turn, if there is one. Anything the
                // player pressed faster than the snake moves is played back in
                // order here, one turn per cell, instead of the last press
                // winning and the rest being dropped.
                const next = this.pending.length ? this.pending.shift() : this.newDir;
                this.newDir.x = next.x;
                this.newDir.y = next.y;

                // Never turn straight into the neck — the one case the queue
                // cannot rule out, since the body has moved since the press.
                if (!(this.body[1].xx === this.head.xx + next.x && this.body[1].yy === this.head.yy + next.y)) {
                    this.dir.x = next.x;
                    this.dir.y = next.y;
                    this.head.dir.x = this.dir.x;
                    this.head.dir.y = this.dir.y;
                }

                for (let i = this.length - 1; i > 0; i--) {
                    this.body[i].dir.x = (this.body[i - 1].x - this.body[i].x) / scl;
                    this.body[i].dir.y = (this.body[i - 1].y - this.body[i].y) / scl;
                }
            }

            this.body.forEach(seg => {
                seg.x += seg.dir.x * SPEED;
                seg.y += seg.dir.y * SPEED;
            });
            this.frameCount++;
        }

        /* One path, one fill. Separate fillRects leave faint seams between
           segments once the canvas is scaled by a non-integer factor, because
           each edge is antialiased on its own; a single path is antialiased
           as one shape.

           The head used to be images/head.png and redHead.png — the body
           colour baked into a bitmap, so it could not follow a theme, and
           its eyes always looked up whichever way the snake was going. It is
           drawn here instead, eyes forward. Geometry matches the bitmap: 6px
           eyes 7px ahead of centre and 7.5px either side, 4px pupils. */
        draw(ctx) {
            ctx.fillStyle = themeVar(this.red ? "--snake-dead" : "--snake-body",
                                     this.red ? "#ff0000" : "rgb(50, 255, 50)");
            ctx.beginPath();
            for (const seg of this.body) ctx.rect(seg.x, seg.y, scl, scl);
            ctx.fill();

            const h = this.head, fx = h.dir.x, fy = h.dir.y;
            const sx = -fy, sy = fx;                     // perpendicular
            const cx = h.x + scl / 2, cy = h.y + scl / 2;
            for (const side of [-1, 1]) {
                const ex = cx + fx * 7 + sx * 7.5 * side;
                const ey = cy + fy * 7 + sy * 7.5 * side;
                ctx.fillStyle = themeVar("--snake-eye", "#fff");
                ctx.fillRect(ex - 3, ey - 3, 6, 6);
                // Pupil one pixel forward and one inward, as in the bitmap.
                ctx.fillStyle = themeVar("--snake-pupil", "#000");
                ctx.fillRect(ex + fx - sx * side - 2, ey + fy - sy * side - 2, 4, 4);
            }
        }

        appendNew() {
            const t = this.tail;
            this.body.push(new Segment(t.x, t.y, { x: 0, y: 0 }));
        }

        checkDeath() {
            if (this.head.xx >= TILE_COUNT || this.head.yy >= TILE_COUNT || this.head.xx < 0 || this.head.yy < 0) return true;
            // Stop at length-1, i.e. EXCLUDE the tail. The tail vacates its
            // cell as the snake moves, so entering it is safe — and the
            // Python env agrees:
            //     blocking_body = self.body if ate else self.body[:-1]
            //
            // This used to include the tail, which made the browser stricter
            // than the environment the AI trained in. Tail-following is the
            // standard survival move for a long snake, so the policy would
            // coil correctly and then be killed for it — looking like a
            // stupid AI when the model was fine (72.3 avg in the Python env).
            for (let i = 1; i < this.length - 1; i++) {
                if (this.head.collides(this.body[i])) return true;
            }
            return false;
        }

        die() {
            this.isDead = true;

            // update() returns early once isDead, so this fires exactly once.
            // The identity checks are needed because both snakes share this
            // class.
            if (this === aiSnake && match && aiInMatch) {
                aiInMatch = false;               // later lives are for show only
                match.aiDied(aiScore);
            }
            if (this === window.snake) {
                // Only a real match counts: not a human run that began before
                // the AI had a model to play with (match is null then).
                if (match) match.humanDied(score);
                const line = matchLine();
                announce(srHuman, "Game over. Score " + score + "." +
                    (line ? " " + line.replace("—", "-") + "." : "") +
                    (TOUCH ? " Tap the board to play again." : " Press Space or Enter to play again."));
            } else if (this === aiSnake) {
                announce(srAi, "The AI died with " + aiScore + ".");
            }
            this.diedAt = performance.now();

            // Red, back to normal, red again: a short flash on death.
            this.red = true;
            setTimeout(() => {
                this.red = false;
                setTimeout(() => { this.red = true; }, 200);
            }, 200);
        }

        get length() { return this.body.length; }
        get head() { return this.body[0]; }
        get tail() { return this.body[this.body.length - 1]; }
        get xx() { return this.head.xx; }
        get yy() { return this.head.yy; }
    }

    class Food {
        constructor(x, y, padding) {
            this.xx = x;
            this.yy = y;
            this.padding = padding;
            this.p = padding;
        }
        // Takes the relevant snake explicitly rather than reading a global —
        // both boards have their own snake and food, so this has to know
        // which one it's avoiding.
        generateNew(snakeRef) {
            /* Excludes every cell a segment touches, not only the one it
               rounds to. Eating is detected halfway across a cell, when
               Segment.xx has already flipped to the cell being entered, so a
               segment moving left or up still counted as being in its OLD cell
               and the cell it was sliding into looked free. Food could appear
               underneath the body, hidden — a state the training env never
               produces. */
            const taken = new Set();
            for (const seg of snakeRef.body) {
                const x0 = Math.floor(seg.x / scl), x1 = Math.ceil(seg.x / scl);
                const y0 = Math.floor(seg.y / scl), y1 = Math.ceil(seg.y / scl);
                taken.add(y0 * TILE_COUNT + x0); taken.add(y0 * TILE_COUNT + x1);
                taken.add(y1 * TILE_COUNT + x0); taken.add(y1 * TILE_COUNT + x1);
            }
            const free = [];
            for (let k = 0; k < TILE_COUNT * TILE_COUNT; k++) if (!taken.has(k)) free.push(k);
            if (!free.length) return;          // board full: nowhere to put it
            // Uniform over the free cells. Math.round(random * 15) made the edge
            // rows and columns half as likely as the rest, and the old retry
            // was unbounded recursion.
            const k = free[Math.floor(Math.random() * free.length)];
            this.xx = k % TILE_COUNT;
            this.yy = Math.floor(k / TILE_COUNT);
            this.p = scl / 2;
        }
        // The grow-in animation advances once per simulation step, not per
        // paint, so it runs at the same speed whatever the display rate.
        step() { if (this.p > this.padding) this.p--; }
        draw(ctx) {
            ctx.fillStyle = themeVar("--snake-food", "#ff0000");
            ctx.fillRect(this.x + this.p, this.y + this.p, scl - 2 * this.p, scl - 2 * this.p);
        }
        get x() { return this.xx * scl; }
        get y() { return this.yy * scl; }
    }

    // Separate keys per board — the AI's record is its own, and mixing them
    // would let one board overwrite the other's best.
    // localStorage throws when site data is blocked. Unguarded, that killed the
    // load handler and the game never started.
    function loadHighScore(key = "snake_high_score") {
        try {
            const v = parseInt(localStorage.getItem(key), 10);
            return Number.isFinite(v) ? v : 0;
        } catch (e) { return 0; }
    }
    function saveHighScore(v, key = "snake_high_score") {
        try { localStorage.setItem(key, String(v)); } catch (e) { /* non-fatal */ }
    }


    /* Board colours come from the theme (see --board-* in shared/themes.css).
       Cached and invalidated when data-theme changes. Reading them fresh cost
       several getComputedStyle calls per tick at 90 Hz, to return the same
       strings; a runtime theme switch still repaints the play area. */
    let themeCache = {};
    new MutationObserver(() => { themeCache = {}; needsPaint = true; })
        .observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    function themeVar(name, fallback) {
        if (!(name in themeCache)) {
            themeCache[name] = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
        }
        return themeCache[name] || fallback;
    }
    function boardBg() { return themeVar("--board-bg", "#000"); }
    function boardInk() { return themeVar("--board-ink", "#fff"); }
    function boardScrim(a) { return "rgba(" + themeVar("--board-scrim", "0, 0, 0") + ", " + a + ")"; }

    /* A font size in board units that never renders below `minCss` CSS
       pixels. Text is drawn in the fixed 480-unit space, so on a phone, where
       the board is shown at ~300px, a 13.5-unit line came out at 8px. */
    function fontUnits(c, units, minCss) {
        const shown = c.canvas._shownCss || BOARD;   // cached by fitCanvas; no layout read per paint
        return Math.max(units, minCss * BOARD / shown);
    }

    function roundRect(c, x, y, w, h, r) {
        c.beginPath();
        if (c.roundRect) c.roundRect(x, y, w, h, r);
        else c.rect(x, y, w, h);
        c.fill();
    }

    /* The score and high score. Drawn faint, as a watermark, BEFORE the
       pieces, so the snake passing through the top rows visibly runs over it
       rather than the number fighting the snake at full ink. The game-over
       card repaints it at full ink above its scrim. */
    /* Ink at partial alpha over white fades much faster than over black: the
       watermark's 0.5 measured 3.51:1 on Filings and 4.00:1 on Dispersion,
       where the same 0.5 on the dark boards gives 5.3:1. On a phone the score
       renders at ~23 CSS px, under the large-text size, so it needs 4.5:1.
       The light boards' watermark is held at 0.62 (5.1:1 on Filings, 6.2:1
       on Dispersion) and the small line above that, measured from the
       canvas's own pixels. */
    function lightBoard() {
        const t = document.documentElement.getAttribute("data-theme");
        return t === "filings" || t === "dispersion";
    }

    function drawScore(c, theScore, theHigh, alpha) {
        const light = lightBoard();
        if (light) alpha = Math.max(alpha, 0.62);
        c.fillStyle = boardInk();
        c.textAlign = "center";
        c.textBaseline = "alphabetic";
        c.globalAlpha = alpha;
        c.font = 1.5 * scl + "px Arial";
        c.fillText(theScore, BOARD / 2, 2.5 * scl);
        c.globalAlpha = Math.min(1, alpha + (light ? 0.15 : 0.1));
        c.font = fontUnits(c, 0.5 * scl, 11) + "px Arial";
        c.fillText("High score: " + theHigh, BOARD / 2, 3.5 * scl);
        c.globalAlpha = 1;
        c.textAlign = "start";
    }

    /* Game over: dim the board, bring the score back to full ink, and put the
       message on its own card. The text used to sit straight on the scrim, so
       the food and snake showed through it ("Enter" printed over a red
       square), and the final score was left faded under the scrim. `lines`
       is [{ text, size, minCss, bold, alpha }], top to bottom.

       A line wider than the card WRAPS at spaces rather than shrinking: the
       match line ("You 12 · AI 3 — AI still playing, needs 10 to win") is
       held at its 12 CSS px floor, which on a 220px phone board is ~26 board
       units and would otherwise run off both edges. */
    function drawGameOver(c, theScore, theHigh, lines) {
        c.fillStyle = boardScrim(0.62);
        c.fillRect(0, 0, BOARD, BOARD);
        drawScore(c, theScore, theHigh, 1);

        c.textAlign = "center";
        c.textBaseline = "middle";
        const pad = 0.7 * scl;
        const maxText = BOARD - 16 - 2 * pad;
        let width = 0, height = 0;
        const sized = [];
        for (const l of lines) {
            const px = fontUnits(c, l.size, l.minCss);
            c.font = (l.bold ? "bold " : "") + px + "px Arial";
            // Break after the dash first, so the scores stay on one row and
            // the verdict on the next, then at spaces if a part still overflows.
            const parts = c.measureText(l.text).width > maxText && l.text.includes(" — ")
                ? l.text.replace(" — ", " —\n").split("\n") : [l.text];
            const rows = [];
            for (const part of parts) {
                let row = "";
                for (const word of part.split(" ")) {
                    const next = row ? row + " " + word : word;
                    if (row && c.measureText(next).width > maxText) { rows.push(row); row = word; }
                    else row = next;
                }
                rows.push(row);
            }
            for (const text of rows) {
                width = Math.max(width, c.measureText(text).width);
                height += px * 1.45;
                sized.push(Object.assign({}, l, { px, text }));
            }
        }
        const w = Math.min(BOARD - 16, width + 2 * pad), h = height + pad;
        // 0.92 rather than the 0.82 of the cards over a live board: this one
        // sits on the scrim, and at 0.82 a red food square still showed
        // through as a pink block behind the text.
        c.fillStyle = boardScrim(0.92);
        roundRect(c, (BOARD - w) / 2, BOARD / 2 - h / 2, w, h, 10);

        let y = BOARD / 2 - height / 2;
        for (const l of sized) {
            c.font = (l.bold ? "bold " : "") + l.px + "px Arial";
            c.fillStyle = boardInk();
            c.globalAlpha = l.alpha == null ? 1 : l.alpha;
            c.fillText(l.text, BOARD / 2, y + l.px * 0.725);
            y += l.px * 1.45;
        }
        c.globalAlpha = 1;
        c.textAlign = "start";
        c.textBaseline = "alphabetic";
    }

    /* Background, score and pieces — shared by both boards. */
    function drawBoard(c, theSnake, theFood, theScore, theHigh) {
        c.fillStyle = boardBg();
        c.fillRect(0, 0, BOARD, BOARD);
        drawScore(c, theScore, theHigh, 0.5);
        theFood.draw(c);
        theSnake.draw(c);
    }

    /* Size each canvas's backing store to the pixels it actually covers,
       and draw in BOARD units through a transform. The bitmap used to be a
       fixed 480x480, which is soft on every high-DPI screen and on phones,
       where CSS shrinks the board and the browser resamples it.

       Whole multiples of BOARD only: an integer transform keeps every cell
       edge on a device pixel, and the browser's downscale from the next size
       up stays sharp. Assigning width clears the canvas, so this only runs
       when the size actually changes; the next frame repaints it. */
    function fitCanvas(cvs, c) {
        const css = cvs.clientWidth;
        if (!css) return;                      // hidden: nothing to measure
        cvs._shownCss = css;
        const k = Math.max(1, Math.ceil(css * (window.devicePixelRatio || 1) / BOARD - 0.01));
        if (cvs.width !== BOARD * k) {
            cvs.width = cvs.height = BOARD * k;
        }
        c.setTransform(k, 0, 0, k, 0, 0);
    }
    function fitCanvases() {
        fitCanvas(canvas, ctx);
        fitCanvas(canvasAi, ctxAi);
        needsPaint = true;
    }
    let needsPaint = true;

    // ── Human board ──────────────────────────────────────────────────────
    let food, score = 0, highScore;

    // One simulation step. Painting is separate (drawHuman), so a frame that
    // runs two steps to keep up with 90 Hz does not paint the board twice.
    function stepHuman() {
        food.step();
        if (matchStarted) snake.update();

        if (snake.head.collides(food)) {
            food.generateNew(snake);
            snake.appendNew();
            score++;
            announce(srHuman, "Score " + score);
        }
        if (score > highScore) {
            highScore = score;
            saveHighScore(highScore);
        }
    }

    function drawHuman() {
        drawBoard(ctx, snake, food, score, highScore);

        // Game over used to be only the snake turning red, with no
        // instruction, and any key at all reloaded the page.
        if (snake.isDead && performance.now() - (snake.diedAt || 0) > GAME_OVER_SHOW_MS) {
            const lines = [{ text: "GAME OVER", size: scl, minCss: 20, bold: true }];
            // The match result, at full ink like the score. Absent when the
            // AI never loaded, or was not loaded when this game began. Read
            // every paint, so "AI still playing, needs N" counts down live
            // and turns into the result the moment the match settles.
            const line = matchLine();
            if (line) lines.push({ text: line, size: 0.55 * scl, minCss: 12 });
            lines.push({ text: TOUCH ? "Tap to play again" : "Press Space or Enter to play again",
                         size: 0.45 * scl, minCss: 11, alpha: 0.75 });
            drawGameOver(ctx, score, highScore, lines);
        }
    }

    // ── AI board ─────────────────────────────────────────────────────────
    const AI_HIGH_SCORE_KEY = "snake_ai_high_score";
    let aiSnake, aiFood, aiScore = 0, aiHighScore = 0;
    // Mirrors snake_env's steps_since_food, which feeds the hunger scalar in
    // the observation. Counts AI decision steps (one per cell), not frames.
    let aiStepsSinceFood = 0;
    let aiAteThisCell = false, aiRestartTimer = null;
    let aiSession = null, aiReady = false, aiInferencePending = false;

    function dirToIndex(dir) {
        for (let i = 0; i < DIRS.length; i++) {
            if (DIRS[i].x === dir.x && DIRS[i].y === dir.y) return i;
        }
        return 1; // fallback: right
    }

    // ── Observation encoder (v2) ─────────────────────────────────────────
    //
    // MUST mirror snake_env.py's _get_obs() field-for-field, in the same
    // order. If this drifts, inference silently produces garbage — the same
    // class of bug as the Tetris 210-vs-238 mismatch.
    //
    //   grid  TILE_COUNT^2 x 5, row-major (y, x, c)
    //           0 body excluding head   1 head   2 food
    //           3 tail                  4 reachable (flood fill from head)
    //   scalars (14)
    //           4 direction one-hot
    //           1 normalised length
    //           3 "this move kills me" per relative action (left, straight, right)
    //           3 reachable free space after that move, board fraction
    //           2 signed food delta (dx, dy) / TILE_COUNT
    //           1 normalised steps since food
    //
    // 256*5 + 14 = 1294 floats.
    const GRID_CHANNELS = 5;
    const N_SCALARS = 14;
    const CELLS = TILE_COUNT * TILE_COUNT;
    const OBS_SIZE = CELLS * GRID_CHANNELS + N_SCALARS;
    const MAX_STEPS_WITHOUT_FOOD = CELLS * 2;

    const cellKey = (x, y) => y * TILE_COUNT + x;
    const outOfBounds = (x, y) => x < 0 || x >= TILE_COUNT || y < 0 || y >= TILE_COUNT;

    // Cells 4-connected to (sx, sy) avoiding `blocked` (a Set of cellKey).
    // Runs the component to completion, so the result does not depend on
    // traversal order — that is what lets this match Python exactly.
    function reachableCells(sx, sy, blocked) {
        const seen = new Set();
        if (outOfBounds(sx, sy)) return seen;
        seen.add(cellKey(sx, sy));
        const stack = [[sx, sy]];
        while (stack.length) {
            const [x, y] = stack.pop();
            for (let d = 0; d < DIRS.length; d++) {
                const nx = x + DIRS[d].x, ny = y + DIRS[d].y;
                if (outOfBounds(nx, ny)) continue;
                const k = cellKey(nx, ny);
                if (seen.has(k) || blocked.has(k)) continue;
                seen.add(k);
                stack.push([nx, ny]);
            }
        }
        return seen;
    }

    function freeSpace(sx, sy, blocked) {
        if (outOfBounds(sx, sy) || blocked.has(cellKey(sx, sy))) return 0;
        return reachableCells(sx, sy, blocked).size;
    }

    function buildObservation(snakeRef, foodRef, stepsSinceFood) {
        const obs = new Float32Array(OBS_SIZE);
        const body = snakeRef.body;
        const head = snakeRef.head;
        const hx = head.xx, hy = head.yy;
        const fx = foodRef.xx, fy = foodRef.yy;

        for (let i = 1; i < body.length; i++) {
            obs[cellKey(body[i].xx, body[i].yy) * GRID_CHANNELS + 0] = 1; // body
        }
        obs[cellKey(hx, hy) * GRID_CHANNELS + 1] = 1;                    // head
        obs[cellKey(fx, fy) * GRID_CHANNELS + 2] = 1;                    // food

        const tail = body[body.length - 1];
        obs[cellKey(tail.xx, tail.yy) * GRID_CHANNELS + 3] = 1;          // tail

        // Reachable from the head. Python blocks body[1:-1] — everything but
        // the head itself and the tail, which vacates as we move.
        const maskBlocked = new Set();
        for (let i = 1; i < body.length - 1; i++) {
            maskBlocked.add(cellKey(body[i].xx, body[i].yy));
        }
        for (const k of reachableCells(hx, hy, maskBlocked)) {
            obs[k * GRID_CHANNELS + 4] = 1;
        }

        // ── scalars ──
        let p = CELLS * GRID_CHANNELS;
        const dirIdx = dirToIndex(snakeRef.dir);
        obs[p + dirIdx] = 1;
        p += 4;

        obs[p] = body.length / CELLS;
        p += 1;

        // Python blocks body[:-1] for candidate moves — the tail vacates.
        const moveBlocked = new Set();
        for (let i = 0; i < body.length - 1; i++) {
            moveBlocked.add(cellKey(body[i].xx, body[i].yy));
        }
        const fatal = [0, 0, 0], space = [0, 0, 0];
        for (let a = 0; a < 3; a++) {
            let d;
            if (a === TURN_LEFT) d = (dirIdx + 3) % 4;        // -1 mod 4
            else if (a === TURN_RIGHT) d = (dirIdx + 1) % 4;
            else d = dirIdx;
            const nx = hx + DIRS[d].x, ny = hy + DIRS[d].y;
            if (outOfBounds(nx, ny) || moveBlocked.has(cellKey(nx, ny))) {
                fatal[a] = 1; space[a] = 0;
            } else {
                fatal[a] = 0; space[a] = freeSpace(nx, ny, moveBlocked) / CELLS;
            }
        }
        obs[p] = fatal[0]; obs[p + 1] = fatal[1]; obs[p + 2] = fatal[2];
        p += 3;
        obs[p] = space[0]; obs[p + 1] = space[1]; obs[p + 2] = space[2];
        p += 3;

        obs[p] = (fx - hx) / TILE_COUNT;
        obs[p + 1] = (fy - hy) / TILE_COUNT;
        p += 2;

        // Python truncates the episode at MAX_STEPS_WITHOUT_FOOD, so the model
        // never saw this above 1; the browser has no such cap.
        obs[p] = Math.min(1, (stepsSinceFood || 0) / MAX_STEPS_WITHOUT_FOOD);

        return obs;
    }

    /* The model source is resolved by shared/model-source.js: the .onnx over
       HTTP, the base64 in model_data.js only under file://. This comment used
       to describe that behaviour while the code did the opposite — game.html
       loaded model_data.js from a static script tag, so the base64 global was
       always defined and the 46 MB branch always won. */
    /* The AI board's status card: a title and a detail line on a pill below
       the spawn row, worded the same in all three games. It used to be one
       uppercase line on a scrim over the whole board, printed across the
       waiting snake. */
    let aiLoadFailed = false, aiLoadDetail = "Downloading the model";
    function setAiStatus(title, detail) {
        if (!title) { aiStatusEl.hidden = true; return; }
        aiStatusEl.hidden = false;
        aiStatusEl.firstElementChild.textContent = title;
        aiStatusEl.lastElementChild.textContent = detail || "";
    }
    function updateAiStatus() {
        if (aiLoadFailed) setAiStatus("Couldn't load the AI", "Check your connection and refresh");
        else if (!aiReady) setAiStatus("Loading AI…", aiLoadDetail);
        // Loaded, holding for the player's first move (an arrow, WASD, a
        // swipe or a tap), which starts both boards. "Press a key" promised
        // more than that: Space and Enter are not moves and start nothing.
        else if (!aiActive) setAiStatus("AI ready", "Starts with your first move");
        else setAiStatus(null);
    }

    async function loadModel() {
        updateAiStatus();
        try {
            ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/";

            // Over HTTP this is the .onnx URL, which streams and is a third
            // smaller than the base64. See shared/model-source.js.
            // Progress, so a 34 MB download does not read as "stuck". Passed
            // fourth, with dataUrl left undefined, so an older model-source.js
            // that takes only three arguments simply ignores it. The total can
            // be 0 (no Content-Length, or a compressed one), hence the MB form.
            const onProgress = (got, total) => {
                aiLoadDetail = total > 0
                    ? "Downloading the model · " + Math.min(99, Math.floor(100 * got / total)) + "%"
                    : "Downloading the model · " + (got / 1048576).toFixed(1) + " MB";
                updateAiStatus();
            };
            const src = await modelSource("snake_ai.onnx", "SNAKE_MODEL_B64", undefined, onProgress);
            aiLoadDetail = "Starting the model…";
            updateAiStatus();

            aiSession = await ort.InferenceSession.create(src, { executionProviders: ["wasm"] });
            aiReady = true;
            // Loaded mid-game: the AI plays along for show, but this game is
            // not a match (match stays null) because it did not start level.
            if (matchStarted) aiActive = true;
            updateAiStatus();
            announce(srAi, aiActive ? "The AI is ready and playing." : "The AI is ready. It starts when you do.");
            mountCheckpointSwitcher();
        } catch (err) {
            console.error("Failed to load Snake AI model:", err);
            aiLoadFailed = true;
            updateAiStatus();
            announce(srAi, "Couldn't load the AI. Check your connection and refresh.");
        }
    }

    /* ── Neural-network inspector ─────────────────────────────────────────
       The grid is stored as cellKey(x, y) * GRID_CHANNELS + channel, where
       cellKey is y * TILE_COUNT + x — so channel c of (row, col) lives at
       (row * TILE_COUNT + col) * GRID_CHANNELS + c. Indexing this differently
       from buildObservation would draw a convincing but wrong picture, which
       is worse than drawing none. */
    const inspector = createInspector({
        mount: document.getElementById("insp-mount-ai"),
        grid: {
            w: TILE_COUNT,
            h: TILE_COUNT,
            channels: [
                { label: "body", hint: "every segment except the head" },
                { label: "head", hint: "where the snake is now" },
                { label: "food", hint: "the target" },
                { label: "tail", hint: "the cell that frees up next move" },
                { label: "reachable", hint: "cells the head can still get to — this is what stops it boxing itself in" },
            ],
        },
        readCell: (obs, row, col, ch) => obs[(row * TILE_COUNT + col) * GRID_CHANNELS + ch],
        actions: {
            count: 3,
            orientation: "rows",
            label: (i) => ["turn left", "straight", "turn right"][i],
        },
        scalars: (obs) => {
            const p = CELLS * GRID_CHANNELS;
            const dir = ["up", "right", "down", "left"];
            let heading = "—";
            for (let i = 0; i < 4; i++) if (obs[p + i]) heading = dir[i] || String(i);
            return [
                { label: "heading", value: heading },
                { label: "length", value: Math.round(obs[p + 4] * CELLS) },
                // p+5..7 are the kill flags and p+8..10 the free space for
                // left / straight / right. This read p+8, the space after
                // turning LEFT, under the label "ahead".
                { label: "free space ahead", value: (obs[p + 9] * 100).toFixed(0) + "%" },
            ];
        },
        valueLabel: "expected score from here",
        valueHint: "the critic's estimate, in reward units",
        onReveal: loadCritic,
    });

    /* Loaded only when the panel is first opened — see the note in
       watermelon/game.js. Blocked under file://, where the value readout
       simply stays hidden. */
    let criticSession = null;
    let criticPending = null;

    /* snake_critic.onnx is the value head of the SHIPPED policy. The checkpoint
       switcher can put an earlier policy in play, and the ladder rungs ship no
       critic of their own (34 MB each, for one readout). Pairing the shipped
       critic with an earlier policy would print confident numbers from a
       network that never saw those weights — so while an earlier rung is
       selected the readout is suppressed rather than guessed at. */
    let criticMatchesModel = true;

    function loadCritic() {
        // No runtime (the CDN was unreachable): the panel shows no value
        // readout rather than throwing out of the inspector's reveal hook.
        if (!criticMatchesModel || typeof ort === "undefined") return Promise.resolve();
        if (criticSession || criticPending) return criticPending;
        criticPending = ort.InferenceSession
            .create("snake_critic.onnx", { executionProviders: ["wasm"] })
            // A switch to an earlier rung while this was downloading must not
            // leave the shipped critic paired with that rung.
            .then((s) => { if (criticMatchesModel) criticSession = s; criticPending = null; })
            .catch((err) => {
                console.warn("Snake critic unavailable — value readout hidden.", err);
            });
        return criticPending;
    }

    /* The checkpoint switcher, mounted under the AI board once the shipped
       model is live — so the control appears already answering "which version
       is this?" rather than as an empty ladder during the initial download.
       Called from loadModel() after its await, so the module's own bindings are
       all initialised by the time this runs. */
    /* Hold the switcher's space from first paint. It only mounts once the
       model has loaded, and without this the AI column grew by ~89px at that
       moment — which, on a vertically centred screen, jerked the whole page. */
    if (typeof CheckpointSwitcher !== "undefined") {
        CheckpointSwitcher.reserve("snake", document.getElementById("board-ai"));
    }

    function mountCheckpointSwitcher() {
        if (typeof CheckpointSwitcher === "undefined") return;
        CheckpointSwitcher.mount({
            game: "snake",
            container: document.getElementById("board-ai"),
            initial: aiSession,
            onSession: (session, rung) => {
                aiSession = session;
                criticMatchesModel = rung.shipped;
                if (!rung.shipped) {
                    criticSession = null;
                    criticPending = null;
                } else if (inspector.isRevealed) {
                    // isRevealed, not isOpen: isOpen uses a 400px warm-up
                    // margin and is true at scroll 0, which re-fetched the
                    // 34 MB critic on load for a panel nobody had reached.
                    loadCritic();
                }
            },
        });
    }

    async function runAiInference() {
        const obs = buildObservation(aiSnake, aiFood, aiStepsSinceFood);
        if (obs.length !== OBS_SIZE) {
            // Cheap guard against the encoder drifting from the model. The
            // failure is otherwise silent — ORT would either throw an opaque
            // dimension error or, worse, accept it and return nonsense.
            console.error(
                `Snake observation is ${obs.length} floats, expected ${OBS_SIZE}. ` +
                "buildObservation() and snake_env._get_obs() have drifted apart."
            );
        }
        const tensor = new ort.Tensor("float32", obs, [1, obs.length]);
        const results = await aiSession.run({ observation: tensor });
        const logits = results.action_logits.data;

        /* Feed the inspector the exact tensor the model just consumed, but do
           not wait for it. The critic run and the redraw used to be awaited
           before the action was returned, so with the panel open each decision
           paid for two model runs and a repaint inside a ~44 ms window, and a
           slow machine missed the cell boundary and turned a cell late. */
        if (inspector.isOpen) {
            const critic = criticSession;
            (critic ? critic.run({ observation: tensor }).then(v => v.value.data[0])
                    : Promise.resolve(undefined))
                .then(value => inspector.update({ obs, logits, value }))
                .catch(() => inspector.update({ obs, logits }));
        }

        // Difficulty: at full strength this is the old argmax; easing off
        // samples from softmax(logits / T) so the AI sometimes takes a move it
        // rated second best. See settings.js.
        return (typeof Settings !== "undefined")
            ? Settings.chooseAction("snake", logits)
            : (() => {
                let bestIdx = 0, bestVal = -Infinity;
                for (let i = 0; i < logits.length; i++) {
                    if (logits[i] > bestVal) { bestVal = logits[i]; bestIdx = i; }
                }
                return bestIdx;
            })();
    }

    function applyAiAction(action) {
        if (action === STRAIGHT || action == null) return; // no-op: keep current heading
        const curIdx = dirToIndex(aiSnake.dir);
        const newIdx = action === TURN_LEFT ? (curIdx + 3) % 4 : (curIdx + 1) % 4;
        const nd = DIRS[newIdx];
        aiSnake.turn(nd.x, nd.y);
    }

    /* A decision that has not landed by the boundary it was made for.

       update() commits a turn at the boundary frame, (cell + 1) *
       STEPS_PER_CELL. This used to drop a decision that resolved after that
       frame, so on a slow machine (measured at a 4x CPU throttle) inference
       overran the window and the AI simply went straight, into walls.
       Applying it late instead would turn a cell late: exactly bug #19 in
       CLAUDE.md, which scored 2.92 against 66.92.

       So the AI board HOLDS at that boundary until its decision arrives, then
       carries on. Every turn still lands on the cell the observation
       described, the same sequence the env plays; a slow machine sees a
       slower AI, not a worse one. The hold is capped, so a hung runtime
       cannot freeze the board: past AI_HOLD_MAX_MS the decision is given up
       and the snake goes straight, as before. Holds and drops are counted
       and logged at debug level, every AI_SLOW_LOG_EVERY of them. */
    let aiPending = null;          // { snake, deadline, since } while a decision is in flight
    const AI_HOLD_MAX_MS = 1000;
    const AI_SLOW_LOG_EVERY = 25;
    const aiSlow = { held: 0, dropped: 0 };
    function noteSlow(kind) {
        aiSlow[kind]++;
        if ((aiSlow.held + aiSlow.dropped) % AI_SLOW_LOG_EVERY === 1) {
            console.debug("Snake AI: inference missed its window — held at the boundary " +
                aiSlow.held + "x, gave up " + aiSlow.dropped + "x so far.");
        }
    }
    function aiHolding() {
        if (!aiPending || aiPending.snake !== aiSnake || aiSnake.frameCount < aiPending.deadline) return false;
        if (performance.now() - aiPending.since < AI_HOLD_MAX_MS) {
            if (!aiPending.held) { aiPending.held = true; noteSlow("held"); }
            return true;
        }
        if (!aiPending.dropped) { aiPending.dropped = true; noteSlow("dropped"); }
        return false;
    }

    function stepAi() {
        aiFood.step();
        if (aiActive && aiReady && !aiHolding()) aiSnake.update();

        if (aiSnake.head.collides(aiFood)) {
            aiFood.generateNew(aiSnake);
            aiSnake.appendNew();
            aiScore++;
            if (match && aiInMatch) match.aiScore(aiScore);
            aiStepsSinceFood = 0;
            aiAteThisCell = true;
        }
        if (aiScore > aiHighScore) {
            aiHighScore = aiScore;
            saveHighScore(aiHighScore, AI_HIGH_SCORE_KEY);
        }

        // Decide once per cell — but WHICH frame of the cell matters enormously.
        //
        // update() consumes newDir at `frameCount % STEPS_PER_CELL === 0`, and
        // that alignment is what steers the *next* cell-to-cell move. Deciding
        // at phase 1 (just after a boundary) meant the snake had already
        // committed to its current move, so every action landed one cell LATE.
        // Measured against the real model: phase 1 scored 2.92 avg / 11 max,
        // phase 0 scored 66.92 — the training env scores 72.3. That off-by-one
        // alone was the whole "the AI is stupid" problem.
        //
        // Segment.xx rounds, so the cell reading flips to the cell being
        // entered once the snake is halfway across. Every phase from that flip
        // up to the boundary yields an identical observation (verified: 6, 7,
        // 8, 9 and 0 all score 64.60). Picking the earliest one leaves the
        // async WASM inference ~4 frames (~67 ms) to resolve instead of ~17 ms,
        // so a slow frame cannot make the turn miss its boundary.
        const AI_DECISION_PHASE = Math.floor(STEPS_PER_CELL / 2) + 1;
        if (
            aiActive && aiReady && !aiSnake.isDead &&
            aiSnake.frameCount % STEPS_PER_CELL === AI_DECISION_PHASE &&
            !aiInferencePending
        ) {
            aiInferencePending = true;
            // snake_env: steps_since_food resets on the step that eats and
            // counts up otherwise. Incrementing unconditionally put the browser
            // one ahead of Python on every cell after the first food.
            if (aiAteThisCell) aiAteThisCell = false;
            else aiStepsSinceFood++;

            // snake_env truncates the episode at MAX_STEPS_WITHOUT_FOOD. The
            // browser used to let a looping AI circle forever, which left a
            // head-to-head match waiting on a first life that never ended.
            if (aiStepsSinceFood >= MAX_STEPS_WITHOUT_FOOD) {
                aiInferencePending = false;
                aiSnake.die();
                return;
            }

            // A decision belongs to THIS cell. update() commits a turn at the
            // boundary frame, (cell + 1) * STEPS_PER_CELL, and aiHolding()
            // keeps the snake at that frame until the decision is in. Only one
            // that outlived the hold cap arrives past it, and that one is
            // dropped: its observation is stale and the turn would land a cell
            // late. A decision for a snake that has since been reset is
            // dropped too.
            const pending = aiPending = {
                snake: aiSnake,
                deadline: (Math.floor(aiSnake.frameCount / STEPS_PER_CELL) + 1) * STEPS_PER_CELL,
                since: performance.now(),
            };
            const settle = () => {
                if (aiPending === pending) aiPending = null;
                aiInferencePending = false;
            };
            runAiInference()
                .then(action => {
                    if (pending.snake === aiSnake && aiSnake.frameCount <= pending.deadline) applyAiAction(action);
                    settle();
                })
                .catch(err => { console.error("AI inference failed:", err); settle(); });
        }

        // The AI board restarts itself, as Tetris's and Watermelon's do.
        if (aiSnake.isDead && !aiRestartTimer) aiRestartTimer = setTimeout(resetAi, 1500);
    }

    function drawAi() {
        drawBoard(ctxAi, aiSnake, aiFood, aiScore, aiHighScore);
        if (aiSnake.isDead && performance.now() - (aiSnake.diedAt || 0) > GAME_OVER_SHOW_MS) {
            drawGameOver(ctxAi, aiScore, aiHighScore, [
                { text: "AI GAME OVER", size: scl, minCss: 20, bold: true },
                { text: "Restarting\u2026", size: 0.45 * scl, minCss: 11, alpha: 0.75 },
            ]);
        }
    }

    // ── Boot ─────────────────────────────────────────────────────────────
    window.tileCount = TILE_COUNT;
    window.speed = SPEED;

    /* Read by shared/confirm-exit.js. A run counts as in progress once the
       player has actually moved and while they are still alive — a fresh
       board, or one already showing game over, has nothing left to lose. */
    window.gameInProgress = function () {
        return matchStarted && !!window.snake && !window.snake.isDead;
    };

    /* In-place resets. Death and both restart buttons used to reload the whole
       page, which rebuilt the 34 MB model, re-downloaded any earlier model
       picked in the switcher, and ended the other board's game too. */
    function resetHuman() {
        food = new Food(6, Math.floor(TILE_COUNT / 2), 3);
        window.snake = new Snake(4, Math.floor(TILE_COUNT / 2), START_LENGTH);
        score = 0;
        // Every restart waits for the first input, like the first game. A
        // match still in play is abandoned unrecorded. One that is WAITING on
        // the AI is left to settle (and record) while the board waits; the
        // next game's first input cancels it if it still has not.
        matchStarted = false;
        if (match && match.state === "playing") match.cancel("you restarted");
        if (controlsOverlay) controlsOverlay.hidden = false;
    }
    function resetAi() {
        clearTimeout(aiRestartTimer);
        aiRestartTimer = null;
        aiFood = new Food(6, Math.floor(TILE_COUNT / 2), 3);
        aiSnake = new Snake(4, Math.floor(TILE_COUNT / 2), START_LENGTH);
        aiScore = 0;
        aiStepsSinceFood = 0;
        aiAteThisCell = false;
        // Any new AI life is not the match's first life; startMatch() sets
        // this back after the reset that begins a match.
        aiInMatch = false;
    }

    /* Boot as soon as this (deferred) script runs, not on window "load".
       The model download used to start from the load handler, so it queued
       behind every image, stylesheet and the background renderer for no
       reason; the boards were blank until then as well. The DOM is complete
       by the time a deferred script executes. */
    resetHuman();
    highScore = loadHighScore();
    resetAi();
    aiHighScore = loadHighScore(AI_HIGH_SCORE_KEY);

    loadModel();

    fitCanvases();
    if (typeof ResizeObserver !== "undefined") {
        new ResizeObserver(fitCanvases).observe(canvas);
    }
    // Page zoom and moving to another monitor change devicePixelRatio
    // without necessarily resizing the canvas in CSS pixels.
    window.addEventListener("resize", fitCanvases);
    // Belt and braces: if anything was read before every stylesheet applied,
    // drop it and repaint once the page has fully loaded.
    window.addEventListener("load", () => { themeCache = {}; fitCanvases(); });

    /* A fixed 90 Hz simulation driven by requestAnimationFrame instead of
       setInterval. The step is unchanged, so speed and the AI's decision
       timing are exactly as before. Paints now land on display frames
       rather than drifting against them (which juddered on 60 Hz screens),
       and the loop stops in background tabs. Catch-up is capped so coming
       back to the tab cannot fast-forward the game.

       Steps and paints are separate. Each step used to repaint both boards,
       so a 60 Hz display (1.5 steps per frame) painted a third more than it
       showed, and a catch-up frame painted up to eight times. Now a frame
       paints once, after however many steps it ran. */
    const STEP_MS = 1000 / 90;
    let acc = 0, aiAcc = 0, last = performance.now();
    function frame(now) {
        acc += Math.min(250, now - last);
        last = now;
        let steps = 0;
        while (acc >= STEP_MS && steps < 8) {
            stepHuman();
            // AI SPEED: the AI board takes aiSpeed of its own steps per tick
            // (0.25x: one every fourth tick; 2x: two a tick), so its snake,
            // its food animation and its decision phase all scale together.
            aiAcc += aiSpeed;
            while (aiAcc >= 1) { stepAi(); aiAcc -= 1; }
            acc -= STEP_MS;
            steps++;
        }
        if (steps === 8) acc = 0;
        if (steps || needsPaint) {
            drawHuman();
            drawAi();
            needsPaint = false;
        }
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);

    const KEY_MAP = {
        arrowup: [0, -1], w: [0, -1],
        arrowdown: [0, 1], s: [0, 1],
        arrowleft: [-1, 0], a: [-1, 0],
        arrowright: [1, 0], d: [1, 0],
    };

    /* The player's first input of a game. If the model is loaded, this is
       also the start of a match: the AI board resets and sets off with them,
       so both start level. If it is not, the player just plays. */
    function startMatch() {
        if (matchStarted) return;
        matchStarted = true;
        if (controlsOverlay) controlsOverlay.hidden = true;
        // The previous match, if it is still waiting on the AI's first life:
        // the board that life is on is about to be reset.
        if (match) match.cancel("a new game started");
        match = null;
        if (aiReady) {
            resetAi();
            aiActive = true;
            aiInMatch = true;
            if (typeof MatchResults !== "undefined") {
                const m = match = MatchResults.begin("snake", { speed: aiSpeed });
                // A match that settles after the card is up (the AI passing
                // the player, or dying) is announced; the card itself re-reads
                // match.line() every paint.
                m.onChange(() => {
                    // diedAt is set after die()'s own announcement, so a match
                    // that settles AT the human's death is not read out twice.
                    if (m === match && window.snake.isDead && window.snake.diedAt &&
                        m.state !== "waiting") {
                        announce(srHuman, m.line().replace("—", "-") + ".");
                    }
                });
            }
        }
        updateAiStatus();
    }
    function matchLine() { return match ? match.line() : ""; }

    // Not during the death flash, or the final score is never seen.
    function canRestartHuman() {
        return !!window.snake && window.snake.isDead &&
               performance.now() - (window.snake.diedAt || 0) > RESTART_DELAY_MS;
    }

    document.addEventListener("keydown", e => {
        if (!window.snake) return;                       // pressed before load
        if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
        const key = e.key.toLowerCase();

        if (window.snake.isDead) {
            // A focused control owns Enter and Space (see shared/keyscroll.js).
            // After picking a model version the rung button keeps focus, and
            // Space used to restart the game AND be swallowed by
            // preventDefault, so the button a keyboard user was on did nothing.
            const t = e.target;
            if (t && t.closest && t.closest("button, a, input, select, textarea, [contenteditable]")) return;
            // A deliberate Enter or Space. This used to be ANY key — including
            // held-key repeats, Tab and modifiers — and it reloaded the page.
            if ((key === "enter" || key === " ") && canRestartHuman()) {
                e.preventDefault();
                resetHuman();
            }
            return;
        }
        if (!(key in KEY_MAP)) return;
        startMatch();
        const [dx, dy] = KEY_MAP[key];
        window.snake.turn(dx, dy);
    });

    /* Touch: swipe on the board to steer, tap to start or to play again. The
       game was keyboard-only, so on a phone both boards sat behind "Press any
       key to start" forever. A turn fires as soon as the finger has travelled
       far enough rather than on release, and one drag can chain turns.
       touch-action: none stops a swipe from scrolling the page. */
    canvas.style.touchAction = "none";
    let swipeFrom = null;
    canvas.addEventListener("pointerdown", e => {
        if (e.pointerType === "mouse") return;
        swipeFrom = { x: e.clientX, y: e.clientY, moved: false };
    });
    canvas.addEventListener("pointermove", e => {
        if (!swipeFrom || !window.snake || window.snake.isDead) return;
        const dx = e.clientX - swipeFrom.x, dy = e.clientY - swipeFrom.y;
        if (Math.max(Math.abs(dx), Math.abs(dy)) < 24) return;
        startMatch();
        if (Math.abs(dx) > Math.abs(dy)) window.snake.turn(dx > 0 ? 1 : -1, 0);
        else window.snake.turn(0, dy > 0 ? 1 : -1);
        swipeFrom = { x: e.clientX, y: e.clientY, moved: true };
    });
    canvas.addEventListener("pointerup", e => {
        if (e.pointerType === "mouse" || !swipeFrom || !window.snake) return;
        const wasTap = !swipeFrom.moved;
        swipeFrom = null;
        if (!wasTap) return;
        if (window.snake.isDead) { if (canRestartHuman()) resetHuman(); }
        else startMatch();
    });
    canvas.addEventListener("pointercancel", () => { swipeFrom = null; });
    canvasAi.addEventListener("pointerup", e => { if (e.pointerType !== "mouse") startMatch(); });

    /* Restart in place. Focus is left alone: a pointer click is blurred by
       the delegated handler below, and a keyboard user who pressed Enter on
       Restart stays on it rather than being dropped to the top of the page.
       Space and Enter on a focused button never reach the game (see the
       keydown handler), so it cannot double as a restart key there.

       Restarting the AI mid-match voids the match: the AI's restart used to
       end its first life early and freeze that score, so one click locked in
       a win. */
    if (restartBtn) restartBtn.addEventListener("click", resetHuman);
    if (restartAiBtn) restartAiBtn.addEventListener("click", () => {
        if (match) match.cancel("the AI was restarted");
        resetAi();
    });

    /* AI SPEED, the same row as Tetris's and Watermelon's. Any speed but 1x
       voids the match in progress (MatchResults.speed), because the result
       would then be about the speed rather than the model.

       2x is offered although inference may not keep up with it: at 2x the
       ~44 ms between the decision phase and the boundary halves to ~22 ms,
       which a slow machine misses. That costs speed, not play: aiHolding()
       keeps the snake at the boundary until the decision lands, so the AI
       runs as fast as inference allows and every turn still lands on time. */
    const speedBox = document.getElementById("speed-ai");
    if (speedBox) {
        speedBox.addEventListener("click", e => {
            const btn = e.target.closest("button[data-speed]");
            if (!btn) return;
            const v = parseFloat(btn.getAttribute("data-speed"));
            if (!(v > 0) || v === aiSpeed) return;
            aiSpeed = v;
            aiAcc = 0;
            for (const b of speedBox.querySelectorAll("button[data-speed]")) {
                const on = b === btn;
                b.classList.toggle("active", on);
                b.setAttribute("aria-pressed", on ? "true" : "false");
            }
            if (match) match.speed(v);
            announce(srAi, "AI speed " + btn.textContent + ".");
        });
    }

    /* A mouse click leaves the button focused, and a focused control owns
       Space and Enter (see the keydown handler and shared/keyscroll.js) — so
       after picking a model version, Space and Enter on the game-over screen
       pressed that rung again instead of restarting. Drop focus after a
       pointer click; keyboard activation (detail 0) keeps it, so tabbing
       through the controls still works. Same as Tetris. */
    document.getElementById("arena").addEventListener("click", e => {
        if (e.detail === 0) return;
        const b = e.target.closest && e.target.closest("button");
        if (b) b.blur();
    });
})();

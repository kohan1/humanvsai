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

    const scl = canvas.width / TILE_COUNT; // both boards render at the same scale
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

    let matchStarted = false;

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
        constructor(x, y, length, color) {
            this.x = x;
            this.y = y;
            this.color = color;
            this.body = [];
            // Preset to rightward rather than zero: movement is gated
            // externally by `matchStarted`, not by dir being non-zero, so
            // starting non-zero here means the frame counter below can
            // actually advance from tick one. Leaving this at {0,0} was
            // the root of the AI's deadlock — its first-ever decision
            // depends on frameCount advancing, which depends on the snake
            // already moving, which never happened.
            this.dir = { x: 1, y: 0 };
            this.newDir = { x: 1, y: 0 };
            this.frameCount = 0;
            this.greenFace = new Image();
            this.greenFace.src = "images/head.png";
            this.redFace = new Image();
            this.redFace.src = "images/redHead.png";
            this.face = this.greenFace;
            for (let n = 0; n < length; n++) {
                this.body.push(new Segment((this.x - n) * scl, this.y * scl, { x: 1, y: 0 }));
            }
        }

        // Absolute direction, exactly like the original — used directly by
        // the keyboard handler, and by the AI's applyAiAction() after it
        // translates a relative action into an absolute dx/dy.
        turn(dx, dy) {
            if (this.isDead) return;
            if (dx !== 0 && this.dir.x === 0) { this.newDir.x = dx; this.newDir.y = 0; }
            else if (dy !== 0 && this.dir.y === 0) { this.newDir.y = dy; this.newDir.x = 0; }
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

                if (!(this.body[1].xx === this.head.xx + this.newDir.x && this.body[1].yy === this.head.yy + this.newDir.y)) {
                    this.dir.x = this.newDir.x;
                    this.dir.y = this.newDir.y;
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

        draw(ctx) {
            this.body.forEach(seg => {
                ctx.fillStyle = this.color;
                ctx.fillRect(seg.x, seg.y, scl, scl);
            });
            ctx.drawImage(this.face, this.head.x, this.head.y, scl, scl);
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
            const original = this.color;
            this.color = "red";
            this.face = this.redFace;
            setTimeout(() => {
                this.color = original;
                this.face = this.greenFace;
                setTimeout(() => {
                    this.color = "red";
                    this.face = this.redFace;
                }, 200);
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
            this.color = "red";
        }
        // Takes the relevant snake explicitly rather than reading a global —
        // both boards have their own snake and food, so this has to know
        // which one it's avoiding.
        generateNew(snakeRef) {
            this.xx = Math.round(Math.random() * (TILE_COUNT - 1));
            this.yy = Math.round(Math.random() * (TILE_COUNT - 1));
            const onSnake = snakeRef.body.some(seg => seg.xx === this.xx && seg.yy === this.yy);
            if (onSnake) this.generateNew(snakeRef);
            else this.p = scl / 2;
        }
        draw(ctx) {
            ctx.fillStyle = this.color;
            ctx.fillRect(this.x + this.p, this.y + this.p, scl - 2 * this.p, scl - 2 * this.p);
            if (this.p > this.padding) this.p--;
        }
        get x() { return this.xx * scl; }
        get y() { return this.yy * scl; }
    }

    // Separate keys per board — the AI's record is its own, and mixing them
    // would let one board overwrite the other's best.
    function loadHighScore(key = "snake_high_score") {
        const v = parseInt(localStorage.getItem(key), 10);
        return Number.isFinite(v) ? v : 0;
    }
    function saveHighScore(v, key = "snake_high_score") {
        localStorage.setItem(key, String(v));
    }

    // ── Human board ──────────────────────────────────────────────────────
    let food, score = 0, highScore;

    function tick() {
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        food.draw(ctx);
        if (matchStarted) snake.update();
        snake.draw(ctx);

        ctx.font = 1.5 * scl + "px Arial";
        ctx.fillStyle = "#fff";
        ctx.fillText(score, canvas.width / 2 - ctx.measureText(score).width / 2, 2.5 * scl);

        ctx.font = 0.5 * scl + "px Arial";
        const label = "High score: " + highScore;
        ctx.fillText(label, canvas.width / 2 - ctx.measureText(label).width / 2, 3.5 * scl);

        if (snake.head.collides(food)) {
            food.generateNew(snake);
            snake.appendNew();
            score++;
        }
        if (score > highScore) {
            highScore = score;
            saveHighScore(highScore);
        }
    }

    // ── AI board ─────────────────────────────────────────────────────────
    const AI_HIGH_SCORE_KEY = "snake_ai_high_score";
    let aiSnake, aiFood, aiScore = 0, aiHighScore = 0;
    // Mirrors snake_env's steps_since_food, which feeds the hunger scalar in
    // the observation. Counts AI decision steps (one per cell), not frames.
    let aiStepsSinceFood = 0;
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

        obs[p] = (stepsSinceFood || 0) / MAX_STEPS_WITHOUT_FOOD;

        return obs;
    }

    /* See the equivalent comment in watermelon/game.js. model_data.js is a
       ~45 MB render-blocking script — the largest of the three games — so over
       HTTP we fetch the .onnx and let the page paint first. The embedded
       base64 is used only when it is present, i.e. local file:// use. */
    async function loadModel() {
        aiStatusEl.textContent = "loading model…";
        aiStatusEl.hidden = false;
        try {
            ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

            let src;
            if (typeof SNAKE_MODEL_B64 !== "undefined") {
                const binaryStr = atob(SNAKE_MODEL_B64);
                src = new Uint8Array(binaryStr.length);
                for (let i = 0; i < binaryStr.length; i++) src[i] = binaryStr.charCodeAt(i);
            } else {
                src = "snake_ai.onnx";
            }

            aiSession = await ort.InferenceSession.create(src, { executionProviders: ["wasm"] });
            aiReady = true;
            aiStatusEl.hidden = true;
        } catch (err) {
            console.error("Failed to load Snake AI model:", err);
            aiStatusEl.textContent = "model failed to load";
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
                { label: "free space ahead", value: (obs[p + 8] * 100).toFixed(0) + "%" },
            ];
        },
        valueLabel: "expected score from here",
        valueHint: "the critic's estimate, in reward units",
    });

    /* Loaded only when the panel is first opened — see the note in
       watermelon/game.js. Blocked under file://, where the value readout
       simply stays hidden. */
    let criticSession = null;
    let criticPending = null;

    function loadCritic() {
        if (criticSession || criticPending) return criticPending;
        criticPending = ort.InferenceSession
            .create("snake_critic.onnx", { executionProviders: ["wasm"] })
            .then((s) => { criticSession = s; })
            .catch((err) => {
                console.warn("Snake critic unavailable — value readout hidden.", err);
            });
        return criticPending;
    }

    const inspToggle = document.getElementById("insp-toggle-ai");
    if (inspToggle) {
        inspToggle.addEventListener("click", () => {
            const on = inspToggle.getAttribute("aria-pressed") !== "true";
            inspToggle.setAttribute("aria-pressed", String(on));
            inspToggle.textContent = on ? "hide what it sees" : "what it sees";
            inspector.setOpen(on);
            if (on) loadCritic();
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

        // Feed the inspector the exact tensor the model just consumed, so
        // what it draws cannot drift from what the network actually saw.
        if (inspector.isOpen) {
            let value;
            if (criticSession) {
                const v = await criticSession.run({ observation: tensor });
                value = v.value.data[0];
            }
            inspector.update({ obs, logits, value });
        }

        let bestIdx = 0, bestVal = -Infinity;
        for (let i = 0; i < logits.length; i++) {
            if (logits[i] > bestVal) { bestVal = logits[i]; bestIdx = i; }
        }
        return bestIdx;
    }

    function applyAiAction(action) {
        if (action === STRAIGHT || action == null) return; // no-op: keep current heading
        const curIdx = dirToIndex(aiSnake.dir);
        const newIdx = action === TURN_LEFT ? (curIdx + 3) % 4 : (curIdx + 1) % 4;
        const nd = DIRS[newIdx];
        aiSnake.turn(nd.x, nd.y);
    }

    function tickAi() {
        ctxAi.fillStyle = "black";
        ctxAi.fillRect(0, 0, canvasAi.width, canvasAi.height);

        aiFood.draw(ctxAi);
        if (matchStarted && aiReady) aiSnake.update();
        aiSnake.draw(ctxAi);

        ctxAi.font = 1.5 * scl + "px Arial";
        ctxAi.fillStyle = "#fff";
        ctxAi.fillText(aiScore, canvasAi.width / 2 - ctxAi.measureText(aiScore).width / 2, 2.5 * scl);

        ctxAi.font = 0.5 * scl + "px Arial";
        const aiLabel = "High score: " + aiHighScore;
        ctxAi.fillText(aiLabel, canvasAi.width / 2 - ctxAi.measureText(aiLabel).width / 2, 3.5 * scl);

        if (aiSnake.head.collides(aiFood)) {
            aiFood.generateNew(aiSnake);
            aiSnake.appendNew();
            aiScore++;
            aiStepsSinceFood = 0;
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
            matchStarted && aiReady && !aiSnake.isDead &&
            aiSnake.frameCount % STEPS_PER_CELL === AI_DECISION_PHASE &&
            !aiInferencePending
        ) {
            aiInferencePending = true;
            aiStepsSinceFood++;
            runAiInference()
                .then(action => { applyAiAction(action); aiInferencePending = false; })
                .catch(err => { console.error("AI inference failed:", err); aiInferencePending = false; });
        }
    }

    // ── Boot ─────────────────────────────────────────────────────────────
    window.tileCount = TILE_COUNT;
    window.speed = SPEED;

    window.addEventListener("load", () => {
        food = new Food(6, Math.floor(TILE_COUNT / 2), 3);
        window.snake = new Snake(4, Math.floor(TILE_COUNT / 2), START_LENGTH, "rgb(50, 255, 50)");
        highScore = loadHighScore();

        aiFood = new Food(6, Math.floor(TILE_COUNT / 2), 3);
        aiSnake = new Snake(4, Math.floor(TILE_COUNT / 2), START_LENGTH, "rgb(50, 255, 50)");
        aiHighScore = loadHighScore(AI_HIGH_SCORE_KEY);

        if (controlsOverlay) controlsOverlay.hidden = false;

        loadModel();

        setInterval(() => { tick(); tickAi(); }, 1000 / 90);
    });

    const KEY_MAP = {
        arrowup: [0, -1], w: [0, -1],
        arrowdown: [0, 1], s: [0, 1],
        arrowleft: [-1, 0], a: [-1, 0],
        arrowright: [1, 0], d: [1, 0],
    };

    document.addEventListener("keydown", e => {
        if (window.snake && window.snake.isDead) { window.location.reload(); return; }
        const key = e.key.toLowerCase();
        if (!(key in KEY_MAP)) return;
        if (!matchStarted) matchStarted = true;
        if (controlsOverlay && !controlsOverlay.hidden) controlsOverlay.hidden = true;
        const [dx, dy] = KEY_MAP[key];
        window.snake.turn(dx, dy);
    });

    if (restartBtn) restartBtn.addEventListener("click", () => window.location.reload());
    if (restartAiBtn) restartAiBtn.addEventListener("click", () => window.location.reload());
})();

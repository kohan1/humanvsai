(() => {
    "use strict";
    /* Board colours from the theme (see --board-* in shared/themes.css). Read
       per frame, so switching theme at runtime repaints the play area too. */
    function themeVar(name, fallback) {
        var v = getComputedStyle(document.documentElement)
                    .getPropertyValue(name).trim();
        return v || fallback;
    }
    function boardBg() { return themeVar("--board-bg", "#000"); }
    function boardInk() { return themeVar("--board-ink", "#fff"); }
    /* The overlay scrim sits UNDER boardInk(), so it has to invert with it —
       a dark scrim under dark ink would be unreadable on the light themes. */
    function boardScrim(a) {
        return themeVar("--board-scrim", "0, 0, 0").replace(/^/, "rgba(") + ", " + a + ")";
    }


    function shuffle(arr) {
        for (var i = arr.length - 1; i > 0; i--) {
            var j = Math.floor(Math.random() * (i + 1));
            var tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
        }
    }

    // ─── Easing (used for AI drop animation) ───────────────────────────────────
    function easeOutQuad(t) { return t * (2 - t); }
    function easeInQuad(t)  { return t * t; }

    var CONFIG = {
        SHAPES: [
            [[1,1,1],[0,1,0],[0,0,0]],
            [[2,2],[2,2]],
            [[0,0,3,0],[0,0,3,0],[0,0,3,0],[0,0,3,0]],
            [[0,4,0],[0,4,0],[0,4,4]],
            [[0,5,0],[0,5,0],[5,5,0]],
            [[0,6,6],[6,6,0],[0,0,0]],
            [[7,7,0],[0,7,7],[0,0,0]]
        ],
        ARENA_WIDTH: 10,
        ARENA_HEIGHT: 18,
        SCALE: 30,
        DROP_INTERVAL: 800,
        DROP_KEY_INTERVAL: 44,
        LOCK_DELAY: 500,
        HORIZONTAL_MOVEMENT_INTERVAL: 76,
        FONT_FAMILY: "Arial, Helvetica, sans-serif",
        COLORS: ["#FF0D72","#0DC2FF","#0DFF72","#F538FF","#FF8E0D","#FFE138","#3877FF","#FF0000"],
        controls: {
            ROTATE:   ["w","arrowup"],
            LEFT:     ["a","arrowleft"],
            DROP:     ["s","arrowdown"],
            RIGHT:    ["d","arrowright"],
            HARDDROP: [" "]
        },
        scorePoints: { DROP: 1, HARDDROP: 3, LANDING: 10, LINECLEAR: 75 }
    };

    // ─── Matrix helpers ───────────────────────────────────────────────────────
    function cloneMatrix(m) { return m.map(function(row){ return row.slice(); }); }
    function cloneShapes(shapes) { return shapes.map(cloneMatrix); }

    function rotateMatrix(matrix, dir) {
        for (var r = 0; r < matrix.length; r++)
            for (var c = 0; c < r; c++) {
                var tmp = matrix[r][c];
                matrix[r][c] = matrix[c][r];
                matrix[c][r] = tmp;
            }
        if (dir > 0) matrix.forEach(function(row){ row.reverse(); });
        else matrix.reverse();
    }

    // ─── Bag / piece ─────────────────────────────────────────────────────────
    function refillBag(bag) {
        cloneShapes(CONFIG.SHAPES).forEach(function(s){ bag.push(s); });
        shuffle(bag);
    }

    function newPiece(bag) {
        if (bag.length === 0) refillBag(bag);
        var shape = bag.pop();
        return {
            x: Math.floor(CONFIG.ARENA_WIDTH / 2 - shape[0].length / 2),
            y: -shape.length,
            shape: shape
        };
    }

    // ─── Collision ───────────────────────────────────────────────────────────
    function hasCollision(state) {
        var shape = state.player.shape;
        for (var r = 0; r < shape.length; r++) {
            for (var c = 0; c < shape[r].length; c++) {
                if (shape[r][c] === 0) continue;
                var row = state.player.y + r;
                var col = state.player.x + c;
                if (col < 0 || col >= CONFIG.ARENA_WIDTH || row >= CONFIG.ARENA_HEIGHT) return true;
                if (row >= 0 && state.arena[row][col] > 0) return true;
            }
        }
        return false;
    }

    function lockPiece(state, addScore) {
        if (addScore === undefined) addScore = true;
        var shape = state.player.shape;
        for (var r = 0; r < shape.length; r++) {
            for (var c = 0; c < shape[r].length; c++) {
                if (shape[r][c] === 0) continue;
                var row = state.player.y + r;
                var col = state.player.x + c;
                if (row >= 0 && row < CONFIG.ARENA_HEIGHT && col >= 0 && col < CONFIG.ARENA_WIDTH)
                    state.arena[row][col] = shape[r][c];
            }
        }
        state.player = newPiece(state.bag);
        if (addScore) state.score += CONFIG.scorePoints.LANDING;
    }

    function createState(highScore) {
        var bag = [];
        refillBag(bag);
        return {
            player: newPiece(bag),
            arena: Array(CONFIG.ARENA_HEIGHT).fill(null).map(function(){ return Array(CONFIG.ARENA_WIDTH).fill(0); }),
            bag: bag,
            paused: false,
            lost: false,
            score: 0,
            highScore: highScore || 0
        };
    }

    // ─── AI speed ────────────────────────────────────────────────────────────
    // Multiplier applied to both the think interval and the drop animation, so
    // the two stay in step. Module-scope because AIPlayer.applyMove and the
    // game loop both need it and live in different closures.
    var AI_SPEED = 1;
    var AI_SPEEDS = [0.25, 0.5, 0.75, 1, 1.5, 1.75, 2];

    // ─── Match start ─────────────────────────────────────────────────────────
    // The AI used to begin playing the moment the model finished loading, so
    // it was already several pieces deep before the player had touched a key
    // and read "Press any key to start". It now waits for that first key, the
    // same way Snake's board does. Module-scope because the loop and the
    // keyboard handler live in different closures.
    var matchStarted = false;

    // ─── Persistence ─────────────────────────────────────────────────────────
    // Board state is plain data — arena and bag are arrays, player is
    // {x, y, shape} — so the whole thing round-trips through JSON with no
    // special handling. Human and AI use separate keys so one board can never
    // restore into the other.
    var Store = {
        read: function(key, fallback) {
            try {
                var raw = localStorage.getItem(key);
                return raw === null ? fallback : JSON.parse(raw);
            } catch (e) { return fallback; }
        },
        write: function(key, value) {
            try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) {}
        },
        clear: function(key) {
            try { localStorage.removeItem(key); } catch (e) {}
        }
    };

    // Guards against a saved board from an older build with different
    // dimensions being restored into the current one.
    function isValidSavedState(s) {
        if (!s || !s.arena || !s.player || !s.bag) return false;
        if (!Array.isArray(s.arena) || s.arena.length !== CONFIG.ARENA_HEIGHT) return false;
        for (var r = 0; r < s.arena.length; r++) {
            if (!Array.isArray(s.arena[r]) || s.arena[r].length !== CONFIG.ARENA_WIDTH) return false;
        }
        if (!Array.isArray(s.bag)) return false;
        if (!s.player.shape || !Array.isArray(s.player.shape)) return false;
        return true;
    }

    // Set when the user deliberately restarts. The restart buttons reload the
    // page, which fires `beforeunload` — without this flag the unload handler
    // would write the boards straight back out after clearSavedGames() wiped
    // them, and the reload would restore the game being restarted.
    var suppressSave = false;

    // Clears both boards' in-progress games but keeps high scores. Used by the
    // restart buttons, which reload the page — without this the reload would
    // restore the very game the user asked to abandon.
    function clearSavedGames() {
        suppressSave = true;
        Store.clear("tetris.human.savedGame");
        Store.clear("tetris.ai.savedGame");
    }

    // ─── Keyboard ────────────────────────────────────────────────────────────
    var KB = {
        ANY: "*",
        keys: {},
        listeners: {},
        init: function() {
            var self = this;
            window.addEventListener("keyup",   function(e){ self.keys[e.key.toLowerCase()] = false; });
            window.addEventListener("keydown", function(e){
                var key = e.key.toLowerCase();
                // Scroll suppression moved to shared/keyscroll.js, which all
                // three games load. This version fired regardless of what had
                // focus, so space on a focused button — the speed controls,
                // the checkpoint switcher — was swallowed instead of
                // activating it. The shared one skips anything with its own
                // keyboard behaviour.
                self._emit(key, e);
                self._emit(KB.ANY, e);
                self.keys[key] = true;
            });
            window.addEventListener("blur", function(){
                for (var k in self.keys) self.keys[k] = false;
            });
        },
        _emit: function(key, e) {
            if (!this.listeners[key]) return;
            var toRemove = [];
            this.listeners[key].forEach(function(fn){
                if (fn(e) === true) toRemove.push(fn);
            });
            var arr = this.listeners[key];
            toRemove.forEach(function(fn){ arr.splice(arr.indexOf(fn), 1); });
        },
        isDown: function(key) {
            if (key === this.ANY) return Object.values(this.keys).some(Boolean);
            return !!this.keys[key.toLowerCase()];
        },
        anyDown: function(keys) {
            var self = this;
            return keys.some(function(k){ return self.isDown(k); });
        },
        on: function(key, fn) {
            key = key.toLowerCase();
            if (!this.listeners[key]) this.listeners[key] = [];
            this.listeners[key].push(fn);
            var arr = this.listeners[key];
            return function(){ arr.splice(arr.indexOf(fn), 1); };
        },
        onMany: function(keys, fn) {
            var unsubs = keys.map(function(k){ return KB.on(k, fn); });
            return function(){ unsubs.forEach(function(u){ u(); }); };
        },
        onPress: function(key, fn) {
            return this.on(key, function(e){ if (!KB.isDown(key)) { fn(e); } });
        },
        once: function(key, fn) {
            return this.on(key, function(e){ fn(e); return true; });
        }
    };

    // ─── Drawing ─────────────────────────────────────────────────────────────
    function drawMatrix(ctx, matrix, ox, oy, scale, colors) {
        for (var r = 0; r < matrix.length; r++) {
            for (var c = 0; c < matrix[r].length; c++) {
                var v = matrix[r][c];
                if (v === 0) continue;
                ctx.fillStyle = colors[v - 1];
                ctx.fillRect((ox + c) * scale, (oy + r) * scale, scale, scale);
                ctx.fillStyle = "rgba(255,255,255,0.12)";
                ctx.fillRect((ox + c) * scale + 2, (oy + r) * scale + 2, scale - 4, scale - 4);
            }
        }
    }

    function drawScore(ctx, score, scale) {
        var size = 2 * scale;
        ctx.textBaseline = "middle";
        ctx.textAlign = "center";
        var text = "" + score;
        ctx.font = size + "px " + CONFIG.FONT_FAMILY;
        while (ctx.measureText(text).width > 5.5 * scale && size > 10) {
            size -= 1;
            ctx.font = size + "px " + CONFIG.FONT_FAMILY;
        }
        ctx.fillStyle = boardInk();
        ctx.fillText(text, scale * (CONFIG.ARENA_WIDTH / 2), 1.75 * scale);
    }

    function drawHighScore(ctx, highScore, scale) {
        ctx.font = (0.6 * scale) + "px " + CONFIG.FONT_FAMILY;
        ctx.fillStyle = boardInk();
        ctx.textBaseline = "middle";
        ctx.textAlign = "center";
        ctx.fillText("High score: " + highScore, scale * (CONFIG.ARENA_WIDTH / 2), 4 * scale);
    }

    // ─── AI helpers (mirrors tetris_env.py exactly) ───────────────────────────

    function aiRotateMatrix(matrix, times) {
        var m = cloneMatrix(matrix);
        for (var t = 0; t < (times % 4); t++) {
            var n = m.length;
            for (var r = 0; r < n; r++)
                for (var c = 0; c < r; c++) {
                    var tmp = m[r][c]; m[r][c] = m[c][r]; m[c][r] = tmp;
                }
            m.forEach(function(row){ row.reverse(); });
        }
        return m;
    }

    function aiHardDropY(arena, shape, px, py) {
        while (true) {
            py++;
            var hit = false;
            outer: for (var r = 0; r < shape.length; r++) {
                for (var c = 0; c < shape[r].length; c++) {
                    if (!shape[r][c]) continue;
                    var ri = py + r, ci = px + c;
                    if (ri >= CONFIG.ARENA_HEIGHT || (ri >= 0 && arena[ri][ci])) { hit = true; break outer; }
                }
            }
            if (hit) return py - 1;
        }
    }

    function aiHasCollision(arena, shape, px, py) {
        for (var r = 0; r < shape.length; r++) {
            for (var c = 0; c < shape[r].length; c++) {
                if (!shape[r][c]) continue;
                var ri = py + r, ci = px + c;
                if (ci < 0 || ci >= CONFIG.ARENA_WIDTH) return true;
                if (ri >= CONFIG.ARENA_HEIGHT) return true;
                if (ri >= 0 && arena[ri][ci]) return true;
            }
        }
        return false;
    }

    function getValidPlacements(arena, shape) {
        var seen = [], placements = [];
        for (var rot = 0; rot < 4; rot++) {
            var rotated = aiRotateMatrix(shape, rot);
            // Deduplicate
            var isDup = seen.some(function(s){
                if (s.length !== rotated.length) return false;
                for (var i = 0; i < s.length; i++)
                    for (var j = 0; j < s[i].length; j++)
                        if (s[i][j] !== rotated[i][j]) return false;
                return true;
            });
            if (isDup) continue;
            seen.push(rotated);
            var pw = rotated[0].length;
            for (var col = -1; col < CONFIG.ARENA_WIDTH - pw + 2; col++) {
                var sy = -rotated.length;
                if (aiHasCollision(arena, rotated, col, sy)) continue;
                var dy = aiHardDropY(arena, rotated, col, sy);
                // Check at least one cell is on the board
                var onBoard = false;
                outer2: for (var r2 = 0; r2 < rotated.length; r2++) {
                    for (var c2 = 0; c2 < rotated[r2].length; c2++) {
                        if (!rotated[r2][c2]) continue;
                        var ri2 = dy+r2, ci2 = col+c2;
                        if (ri2 >= 0 && ri2 < CONFIG.ARENA_HEIGHT && ci2 >= 0 && ci2 < CONFIG.ARENA_WIDTH) {
                            onBoard = true; break outer2;
                        }
                    }
                }
                if (onBoard) placements.push({ shape: rotated, col: col, dy: dy });
            }
        }
        return placements;
    }

    // ─── ONNX observation builder (mirrors tetris_env.py build_obs) ───────────

    function colHeights(arena) {
        var h = [];
        for (var c = 0; c < CONFIG.ARENA_WIDTH; c++) {
            var ht = 0;
            for (var r = 0; r < CONFIG.ARENA_HEIGHT; r++) {
                if (arena[r][c]) { ht = CONFIG.ARENA_HEIGHT - r; break; }
            }
            h.push(ht);
        }
        return h;
    }

    function countHoles(arena) {
        var holes = 0;
        for (var c = 0; c < CONFIG.ARENA_WIDTH; c++) {
            var filled = false;
            for (var r = 0; r < CONFIG.ARENA_HEIGHT; r++) {
                if (arena[r][c]) filled = true;
                else if (filled) holes++;
            }
        }
        return holes;
    }

    function getPieceId(shape) {
        for (var r = 0; r < shape.length; r++)
            for (var c = 0; c < shape[r].length; c++)
                if (shape[r][c]) return shape[r][c] - 1;
        return 0;
    }

    // Peek at the next N pieces from the bag without mutating game state.
    // Mirrors the 7-bag randomiser: pieces come off the end of the bag
    // (bag.pop() order). If the bag runs low, simulates a fresh shuffled
    // refill so the lookahead never runs out.
    function peekNextPieces(bag, n) {
        var virtualBag = bag.slice(); // shallow copy — shape refs are fine, we only read ids
        var result = [];
        for (var i = 0; i < n; i++) {
            if (virtualBag.length === 0) {
                var fresh = cloneShapes(CONFIG.SHAPES);
                shuffle(fresh);
                fresh.forEach(function(s){ virtualBag.push(s); });
            }
            result.push(virtualBag.pop());
        }
        return result;
    }

    function buildObs(arena, curPiece, nextPieces, combo) {
        var obs = new Float32Array(238);
        var idx = 0;

        // Raw board (180)
        for (var r = 0; r < CONFIG.ARENA_HEIGHT; r++)
            for (var c = 0; c < CONFIG.ARENA_WIDTH; c++)
                obs[idx++] = arena[r][c] ? 1.0 : 0.0;

        // Column heights (10)
        var h = colHeights(arena);
        for (var i = 0; i < CONFIG.ARENA_WIDTH; i++)
            obs[idx++] = h[i] / CONFIG.ARENA_HEIGHT;

        // Current piece one-hot (7)
        var curId  = getPieceId(curPiece);
        for (var i = 0; i < 7; i++) obs[idx++] = (i === curId) ? 1.0 : 0.0;

        // Next 5 pieces one-hot (35)
        for (var p = 0; p < 5; p++) {
            var pieceId = nextPieces[p] ? getPieceId(nextPieces[p]) : -1;
            for (var i = 0; i < 7; i++) obs[idx++] = (i === pieceId) ? 1.0 : 0.0;
        }

        // Scalars (4)
        var holes = countHoles(arena);
        var bumps = 0;
        for (var i = 0; i < CONFIG.ARENA_WIDTH - 1; i++) bumps += Math.abs(h[i] - h[i+1]);
        var aggH  = h.reduce(function(a,b){ return a+b; }, 0);
        var maxH  = Math.max.apply(null, h);
        obs[idx++] = Math.min(holes, CONFIG.ARENA_WIDTH * CONFIG.ARENA_HEIGHT) / (CONFIG.ARENA_WIDTH * CONFIG.ARENA_HEIGHT);
        obs[idx++] = Math.min(bumps, CONFIG.ARENA_HEIGHT * CONFIG.ARENA_WIDTH) / (CONFIG.ARENA_HEIGHT * CONFIG.ARENA_WIDTH);
        obs[idx++] = Math.min(aggH,  CONFIG.ARENA_HEIGHT * CONFIG.ARENA_WIDTH) / (CONFIG.ARENA_HEIGHT * CONFIG.ARENA_WIDTH);
        obs[idx++] = maxH / CONFIG.ARENA_HEIGHT;

        // Combo (1)
        obs[idx++] = Math.min(combo, 20) / 20.0;

        // Well depth (1)
        var sortedH = h.slice().sort(function(a,b){ return a-b; });
        obs[idx++] = Math.min(sortedH[1] - sortedH[0], CONFIG.ARENA_HEIGHT) / CONFIG.ARENA_HEIGHT;

        return obs;
    }

    /* ── Neural-network inspector ─────────────────────────────────────────
       The board occupies the first ARENA_HEIGHT * ARENA_WIDTH floats of the
       238-float observation, row-major and single-channel, so cell (row, col)
       is simply obs[row * ARENA_WIDTH + col].

       The 40 action logits are candidate PLACEMENTS (rotation x column), not
       columns — chooseMove only considers the first `placements.length` of
       them, which is why the panel labels them as placements. */
    var inspector = createInspector({
        mount: document.getElementById("insp-mount-ai"),
        grid: {
            w: CONFIG.ARENA_WIDTH,
            h: CONFIG.ARENA_HEIGHT,
            channels: [{ label: "board", hint: "1 where a cell is filled" }]
        },
        readCell: function (obs, row, col) {
            return obs[row * CONFIG.ARENA_WIDTH + col];
        },
        actions: { count: 40, orientation: "columns", label: function (i) { return String(i); } },
        scalars: function (obs) {
            var W = CONFIG.ARENA_WIDTH, H = CONFIG.ARENA_HEIGHT;
            var p = W * H + W + 7 + 35;          // board, heights, current, next five
            var cells = W * H;
            return [
                { label: "holes", value: Math.round(obs[p] * cells) },
                { label: "bumpiness", value: Math.round(obs[p + 1] * cells) },
                { label: "stack height", value: Math.round(obs[p + 2] * cells) },
                { label: "tallest column", value: Math.round(obs[p + 3] * H) },
                { label: "combo", value: Math.round(obs[p + 4] * 20) }
            ];
        },
        valueLabel: "expected score from here",
        valueHint: "the critic's estimate, in reward units",
        onReveal: loadCritic
    });

    /* Fetched only when the panel is first opened. Blocked under file://,
       where the value readout stays hidden and everything else still works. */
    var criticSession = null;
    var criticPending = null;

    function loadCritic() {
        if (criticSession || criticPending) return criticPending;
        criticPending = ort.InferenceSession
            .create("tetris_critic.onnx", { executionProviders: ["wasm"] })
            .then(function (s) { criticSession = s; })
            .catch(function (err) {
                console.warn("Tetris critic unavailable — value readout hidden.", err);
            });
        return criticPending;
    }

    // ─── AI player ───────────────────────────────────────────────────────────

    function AIPlayer(session) {
        this.session = session;
        this.combo   = 0;
        this.busy    = false;
    }

    AIPlayer.prototype.chooseMove = async function(state, nextPieces) {
        if (this.busy) return null;
        this.busy = true;

        var placements = getValidPlacements(state.arena, state.player.shape);
        if (placements.length === 0) { this.busy = false; return null; }

        var obs = buildObs(state.arena, state.player.shape, nextPieces, this.combo);

        try {
            var tensor = new ort.Tensor("float32", obs, [1, 238]);
            var results = await this.session.run({ observation: tensor });
            var logits  = results.action_logits.data;

            // The inspector sees the exact tensor the model just consumed.
            if (inspector.isOpen) {
                var value;
                if (criticSession) {
                    var v = await criticSession.run({ observation: tensor });
                    value = v.value.data[0];
                }
                inspector.update({ obs: obs, logits: logits, value: value });
            }

            // Pick a valid placement. Only the first placements.length logits
            // are legal, so the choice is capped there — at full strength this
            // is the original argmax; easing off samples. See settings.js.
            var bestIdx;
            if (typeof Settings !== "undefined") {
                bestIdx = Settings.chooseAction("tetris", logits, placements.length);
            } else {
                var bestScore = -Infinity;
                bestIdx = 0;
                for (var i = 0; i < Math.min(logits.length, placements.length); i++) {
                    if (logits[i] > bestScore) { bestScore = logits[i]; bestIdx = i; }
                }
            }

            // NOTE: busy stays true here on purpose — applyMove() now kicks off
            // a drop animation instead of locking instantly, and the game loop
            // clears busy once that animation finishes (see createGame()).
            return placements[bestIdx];
        } catch(e) {
            console.error("AI inference error:", e);
            this.busy = false;
            return null;
        }
    };

    AIPlayer.prototype.applyMove = function(state, placement) {
        // Instead of teleporting straight to the final spot, kick off a
        // scripted animation: slide toward the target column while falling
        // into view, then accelerate downward onto the landing row — like
        // a human sliding a piece over before dropping it.
        //
        // The AI pre-computes the exact rotated shape via aiRotateMatrix,
        // so we swap the shape in immediately (no visual rotation needed —
        // tweening a tetromino's rotation reads as confusing, not human).
        var shape = cloneMatrix(placement.shape);
        var fromX = state.player.x;
        var fromY = Math.min(state.player.y, -1); // ensure it starts above the board

        state.player.shape = shape;
        state.player.anim = {
            fromX: fromX,  toX: placement.col,
            fromY: fromY,  midY: 0,  toY: placement.dy,
            elapsed: 0,
            // Scaled by AI_SPEED so the animation keeps pace with the think
            // interval — otherwise at 2x the AI would want to move again
            // while the previous piece was still sliding into place.
            shiftDur: 130 / AI_SPEED, // ms — horizontal slide + fall into view
            dropDur:  170 / AI_SPEED, // ms — accelerating drop onto landing row
            curX: fromX,   curY: fromY,
            placement: placement
        };
        // Note: busy stays true and lockPiece() is deferred until the
        // animation finishes — see the loop's AI branch in createGame().
    };

    // ─── Game loop factory ────────────────────────────────────────────────────

    function createGame(canvasEl, isAI, aiPlayer, getNextPiece) {
        var dpr   = window.devicePixelRatio || 1;
        var CSS_W = 300, CSS_H = 540;
        canvasEl.width  = CSS_W * dpr;
        canvasEl.height = CSS_H * dpr;
        canvasEl.style.width  = CSS_W + "px";
        canvasEl.style.height = CSS_H + "px";
        var ctx = canvasEl.getContext("2d");
        ctx.scale(dpr, dpr);

        var SAVE_KEY = "tetris." + (isAI ? "ai" : "human") + ".savedGame";
        var HIGH_KEY = "tetris." + (isAI ? "ai" : "human") + ".highScore";

        function loadHighScore() {
            var v = Store.read(HIGH_KEY, 0);
            return typeof v === "number" && isFinite(v) ? v : 0;
        }

        function saveGame() {
            if (suppressSave) return;
            // A finished board is not worth restoring — drop it and keep only
            // the high score, so a reload starts fresh instead of reopening on
            // a game-over screen.
            if (state.lost) {
                Store.clear(SAVE_KEY);
            } else {
                Store.write(SAVE_KEY, {
                    player: state.player,
                    arena:  state.arena,
                    bag:    state.bag,
                    score:  state.score
                });
            }
            Store.write(HIGH_KEY, Math.max(state.highScore || 0, state.score || 0));
        }

        function restoreGame() {
            var saved = Store.read(SAVE_KEY, null);
            if (!isValidSavedState(saved)) return null;
            return {
                player:    saved.player,
                arena:     saved.arena,
                bag:       saved.bag,
                paused:    false,
                lost:      false,
                score:     saved.score || 0,
                highScore: Math.max(saved.score || 0, loadHighScore())
            };
        }

        var state  = restoreGame() || createState(loadHighScore());
        var timers = { lastTime: 0, dropCounter: 0, lockCounter: 0, horizCounter: 0 };
        var aiThinkTimer = 0;
        var AI_THINK_INTERVAL = 300; // ms between AI moves

        function loop(ts) {
            var dt = ts - timers.lastTime;
            timers.lastTime = ts;
            if (dt > 200) dt = 200;

            if (!state.paused && !state.lost) {
                timers.dropCounter  += dt;
                timers.horizCounter += dt;

                if (isAI && matchStarted && aiPlayer && aiPlayer.session && !state.restarting) {
                    if (state.player.anim) {
                        // ── Advance the in-flight drop animation ──────────
                        var anim = state.player.anim;
                        anim.elapsed += dt;

                        if (anim.elapsed < anim.shiftDur) {
                            var t1 = anim.elapsed / anim.shiftDur;
                            var e1 = easeOutQuad(t1);
                            anim.curX = anim.fromX + (anim.toX  - anim.fromX) * e1;
                            anim.curY = anim.fromY + (anim.midY - anim.fromY) * e1;
                        } else if (anim.elapsed < anim.shiftDur + anim.dropDur) {
                            var t2 = (anim.elapsed - anim.shiftDur) / anim.dropDur;
                            var e2 = easeInQuad(t2);
                            anim.curX = anim.toX;
                            anim.curY = anim.midY + (anim.toY - anim.midY) * e2;
                        } else {
                            // Animation finished — snap to final spot and lock
                            state.player.x = anim.toX;
                            state.player.y = anim.toY;
                            state.player.anim = null;
                            lockPiece(state, true);
                            aiPlayer.busy = false;
                        }
                    } else {
                        // AI: think and place every AI_THINK_INTERVAL ms,
                        // shortened by the speed multiplier.
                        aiThinkTimer += dt;
                        if (aiThinkTimer >= AI_THINK_INTERVAL / AI_SPEED && !aiPlayer.busy) {
                            aiThinkTimer = 0;
                            var nextPieces = peekNextPieces(state.bag, 5);
                            aiPlayer.chooseMove(state, nextPieces).then(function(placement) {
                                if (placement && !state.lost) {
                                    aiPlayer.applyMove(state, placement);
                                } else {
                                    // No valid placement (or lost mid-think) — release the lock
                                    aiPlayer.busy = false;
                                }
                            });
                        }
                    }
                } else if (!isAI) {
                    // Human: gravity + controls
                    state.player.y += 1;
                    var resting = hasCollision(state);
                    state.player.y -= 1;

                    if (resting) {
                        timers.lockCounter += dt;
                        if (timers.lockCounter >= CONFIG.LOCK_DELAY) {
                            lockPiece(state);
                            timers.lockCounter = 0;
                            timers.dropCounter = 0;
                        }
                    } else {
                        timers.lockCounter = 0;
                        if (timers.dropCounter >= CONFIG.DROP_INTERVAL) {
                            timers.dropCounter = 0;
                            state.player.y += 1;
                            if (hasCollision(state)) state.player.y -= 1;
                        }
                    }

                    if (timers.dropCounter > CONFIG.DROP_KEY_INTERVAL && KB.anyDown(CONFIG.controls.DROP)) {
                        timers.dropCounter = 0;
                        state.player.y += 1;
                        if (hasCollision(state)) state.player.y -= 1;
                        else state.score += CONFIG.scorePoints.DROP;
                    }

                    if (timers.horizCounter > CONFIG.HORIZONTAL_MOVEMENT_INTERVAL) {
                        timers.horizCounter = 0;
                        var prevX = state.player.x;
                        if      (KB.anyDown(CONFIG.controls.RIGHT)) state.player.x += 1;
                        else if (KB.anyDown(CONFIG.controls.LEFT))  state.player.x -= 1;
                        if (hasCollision(state)) state.player.x = prevX;
                    }
                }

                // Clear full lines (both human and AI)
                for (var r = CONFIG.ARENA_HEIGHT - 1; r >= 0; r--) {
                    if (state.arena[r].every(function(v){ return v > 0; })) {
                        state.arena.splice(r, 1);
                        state.arena.unshift(Array(CONFIG.ARENA_WIDTH).fill(0));
                        state.score += CONFIG.scorePoints.LINECLEAR;
                        r++;
                    }
                }

                // Detect loss
                if (state.arena.slice(0, 3).some(function(row){
                    return row.some(function(v){ return v > 0; });
                })) {
                    // Record once, on the transition into the lost state —
                    // this block runs every frame while the board sits lost.
                    if (!isAI && !state.lost && typeof MatchResults !== "undefined") {
                        var aiState = window.__aiGame ? window.__aiGame.getState() : null;
                        MatchResults.record("tetris", state.score,
                                            aiState ? aiState.score : 0, Date.now());
                    }
                    state.lost = true;
                }
            }

            // Restart human game on keypress after loss
            if (!isAI && state.lost && KB.isDown(KB.ANY)) {
                var prevHigh = Math.max(state.highScore, state.score);
                state = createState(prevHigh);
                timers = { lastTime: ts, dropCounter: 0, lockCounter: 0, horizCounter: 0 };
            }

            // AI auto-restart after loss
            if (isAI && state.lost && !state.restarting) {
                state.restarting = true;
                var prevHighScore = state.highScore;
                setTimeout(function() {
                    state = createState(Math.max(prevHighScore, state ? state.score : 0));
                    timers = { lastTime: 0, dropCounter: 0, lockCounter: 0, horizCounter: 0 };
                    aiThinkTimer = 0;
                    if (aiPlayer) { aiPlayer.combo = 0; aiPlayer.busy = false; }
                }, 1500);
            }

            // ── Render ───────────────────────────────────────────────────────
            ctx.fillStyle = boardBg();
            ctx.fillRect(0, 0, 300, 540);

            drawMatrix(ctx, state.arena, 0, 0, CONFIG.SCALE, CONFIG.COLORS);

            // Danger line
            ctx.strokeStyle = "rgba(255,40,40,0.5)";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(0, 3 * CONFIG.SCALE);
            ctx.lineTo(CONFIG.ARENA_WIDTH * CONFIG.SCALE, 3 * CONFIG.SCALE);
            ctx.stroke();

            // Active piece
            if (!state.lost) {
                var lockAlpha = timers.lockCounter > 0
                    ? 1 - (timers.lockCounter / CONFIG.LOCK_DELAY) * 0.5
                    : 1;
                var renderX = state.player.x;
                var renderY = state.player.y;
                if (isAI && state.player.anim) {
                    renderX = state.player.anim.curX;
                    renderY = state.player.anim.curY;
                }
                ctx.globalAlpha = lockAlpha;
                drawMatrix(ctx, state.player.shape, renderX, renderY, CONFIG.SCALE, CONFIG.COLORS);
                ctx.globalAlpha = 1;
            }

            // Update high score live as score increases. Written straight
            // through so a crash or force-close can't lose it — this only
            // fires on an actual increase, not every frame.
            if (state.score > state.highScore) {
                state.highScore = state.score;
                Store.write(HIGH_KEY, state.highScore);
            }
            drawScore(ctx, state.score, CONFIG.SCALE);
            drawHighScore(ctx, state.highScore, CONFIG.SCALE);

            // Game over overlay
            if (state.lost) {
                ctx.fillStyle = boardScrim(0.65);
                ctx.fillRect(0, 0, 300, 540);
                ctx.fillStyle = boardInk();
                ctx.font = "bold " + (1.2 * CONFIG.SCALE) + "px " + CONFIG.FONT_FAMILY;
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(isAI ? "AI DIED" : "GAME OVER", 150, 270 - CONFIG.SCALE);
                ctx.font = (0.55 * CONFIG.SCALE) + "px " + CONFIG.FONT_FAMILY;
                ctx.globalAlpha = 0.75; ctx.fillStyle = boardInk();
                ctx.fillText(
                    isAI ? "Restarting..." : "Press any key to restart",
                    150, 270 + CONFIG.SCALE * 0.2
                );
                ctx.globalAlpha = 1;   // persists across frames if left set
            }

            // AI waiting overlay — either the model is still loading, or it is
            // ready and holding for the player to start. Without the second
            // case the board just sits there looking broken.
            if (isAI && (!aiPlayer || !aiPlayer.session || !matchStarted)) {
                var loading = !aiPlayer || !aiPlayer.session;
                ctx.fillStyle = boardScrim(0.75);
                ctx.fillRect(0, 0, 300, 540);
                ctx.fillStyle = boardInk();
                ctx.font = "bold 18px " + CONFIG.FONT_FAMILY;
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(loading ? "Loading AI..." : "Ready", 150, 260);
                ctx.font = "13px " + CONFIG.FONT_FAMILY;
                ctx.globalAlpha = 0.6; ctx.fillStyle = boardInk();
                ctx.fillText(
                    loading ? "Run embed_model.py, then refresh"
                            : "Press any key to start",
                    150, 285
                );
                ctx.globalAlpha = 1;
            }

            requestAnimationFrame(loop);
        }

        requestAnimationFrame(function(ts) {
            timers.lastTime = ts;
            loop(ts);
        });

        // Persist on the way out. `visibilitychange` is the reliable one —
        // `beforeunload` does not always fire when a tab is closed or the
        // browser is killed, so both are wired up.
        window.addEventListener("beforeunload", saveGame);
        document.addEventListener("visibilitychange", function() {
            if (document.visibilityState === "hidden") saveGame();
        });

        return {
            getState: function() { return state; },
            save: saveGame
        };
    }

    // ─── Controls overlay ─────────────────────────────────────────────────────
    function showControls(el, visible, animated) {
        if (animated) el.classList.add("animated"); else el.classList.remove("animated");
        if (visible) el.removeAttribute("hidden"); else el.setAttribute("hidden", "");
    }

    // ─── Init ─────────────────────────────────────────────────────────────────
    window.addEventListener("DOMContentLoaded", async function() {
        KB.init();

        // ── Human board ───────────────────────────────────────────────────────
        var humanCanvas   = document.getElementById("canvas-human");
        var controlsEl    = document.querySelector("#board-human div.controls");

        var humanGame = createGame(humanCanvas, false, null, null);

        showControls(controlsEl, true, false);
        KB.once(KB.ANY, function(){
            showControls(controlsEl, false, true);
            matchStarted = true;   // releases the AI board — see the flag above
        });

        // Rotate (W / Up) — human only
        KB.onMany(CONFIG.controls.ROTATE, function() {
            var state = humanGame.getState();
            if (state.paused || state.lost) return;
            var prevX = state.player.x;
            rotateMatrix(state.player.shape, 1);
            var kick = 0, attempts = 0;
            while (hasCollision(state)) {
                state.player.x += kick;
                kick = kick > 0 ? -(kick + 1) : (1 - kick);
                attempts++;
                if (attempts > state.player.shape[0].length * 2) {
                    rotateMatrix(state.player.shape, -1);
                    state.player.x = prevX;
                    break;
                }
            }
        });

        // Hard drop (Space) — human only
        KB.onPress(" ", function() {
            var state = humanGame.getState();
            if (state.paused || state.lost) return;
            var dropped = 0;
            state.player.y += 1;
            while (!hasCollision(state)) { state.player.y++; dropped++; }
            state.player.y--;
            if (dropped > 0) state.score += dropped * CONFIG.scorePoints.HARDDROP;
            lockPiece(state, false);
        });

        // ── AI board ──────────────────────────────────────────────────────────
        var aiCanvas  = document.getElementById("canvas-ai");
        var aiPlayer  = null;
        var aiGame    = createGame(aiCanvas, true, aiPlayer, null);

        // Try to load ONNX model.
        // Prefer the embedded base64 model (model_data.js) so this works when
        // game.html is opened directly (file://) with no local server.
        // Falls back to fetch("tetris_ai.onnx") if model_data.js isn't present,
        // which still requires a local server due to browser file:// restrictions.
        try {
            ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

            var session;
            if (typeof TETRIS_MODEL_B64 !== "undefined") {
                // Decode base64 -> Uint8Array, feed straight to onnxruntime-web
                var binaryStr = atob(TETRIS_MODEL_B64);
                var bytes = new Uint8Array(binaryStr.length);
                for (var i = 0; i < binaryStr.length; i++) {
                    bytes[i] = binaryStr.charCodeAt(i);
                }
                session = await ort.InferenceSession.create(bytes, {
                    executionProviders: ["wasm"]
                });
            } else {
                session = await ort.InferenceSession.create("tetris_ai.onnx", {
                    executionProviders: ["wasm"]
                });
            }

            aiPlayer = new AIPlayer(session);
            // Patch aiPlayer into the game loop by replacing the reference
            // The loop checks aiPlayer.session so we just swap it in
            aiGame._aiPlayer = aiPlayer;
            // Re-create AI game with the loaded model
            aiGame = createGame(aiCanvas, true, aiPlayer, null);
            // The human board's loop needs the AI's score to record a match,
            // and the two live in separate createGame closures.
            window.__aiGame = aiGame;
            console.log("AI model loaded successfully.");

            /* Checkpoint switcher. Tetris holds its session on the AIPlayer
               rather than in a module variable, and the game loop reads
               aiPlayer.session on every decision — so assigning to it is all
               that a switch needs; the running game picks the new model up on
               its next move.

               No critic bookkeeping here, unlike the other two: Tetris ships no
               critic at all (the 1B-step checkpoint that produced its .onnx is
               gone), so the value readout is already hidden on every rung. */
            if (typeof CheckpointSwitcher !== "undefined") {
                var sw = CheckpointSwitcher.mount({
                    game: "tetris",
                    container: document.getElementById("board-ai"),
                    initial: session,
                    onSession: function (s) { aiPlayer.session = s; },
                });
                // Compact, as on Watermelon. The stacked variant is 89px tall
                // and sits under the speed controls, which left the AI column
                // hanging 187px below the human one; compact is 31px.
                if (sw) {
                    var el = document.querySelector("#board-ai .ckpt");
                    if (el) el.classList.add("ckpt--compact");
                }
            }
        } catch(e) {
            console.warn("AI model failed to load:", e.message);
        }

        // ── Restart buttons ───────────────────────────────────────────────────
        // Both restart buttons reload the page. Saved boards must be dropped
        // first, or the reload would restore the game being restarted. High
        // scores live under separate keys and survive.
        document.getElementById("restart-human").addEventListener("click", function() {
            clearSavedGames();
            window.location.reload();
        });
        // ── AI speed buttons ──────────────────────────────────────────────
        var speedBox = document.getElementById("speed-ai");
        if (speedBox) {
            speedBox.addEventListener("click", function(e) {
                var btn = e.target.closest("button[data-speed]");
                if (!btn) return;
                var v = parseFloat(btn.getAttribute("data-speed"));
                if (!isFinite(v) || v <= 0) return;
                AI_SPEED = v;
                var all = speedBox.querySelectorAll("button[data-speed]");
                for (var i = 0; i < all.length; i++) {
                    all[i].classList.toggle("active", all[i] === btn);
                }
            });
        }

        document.getElementById("restart-ai").addEventListener("click", function() {
            clearSavedGames();
            window.location.reload();
        });
    });
})();
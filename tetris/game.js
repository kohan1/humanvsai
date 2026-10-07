(() => {
    "use strict";
    /* Board colours from the theme (see --board-* in shared/themes.css).
       Cached, and invalidated when data-theme changes: this used to call
       getComputedStyle several times per board per frame to return the same
       strings. Switching theme at runtime still repaints the play area. */
    var _themeCache = {};
    new MutationObserver(function () { _themeCache = {}; })
        .observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    function themeVar(name, fallback) {
        if (!(name in _themeCache)) {
            _themeCache[name] = getComputedStyle(document.documentElement)
                                    .getPropertyValue(name).trim();
        }
        return _themeCache[name] || fallback;
    }
    function boardBg() { return themeVar("--board-bg", "#000"); }
    function boardInk() { return themeVar("--board-ink", "#fff"); }
    /* The overlay scrim sits UNDER boardInk(), so it has to invert with it —
       a dark scrim under dark ink would be unreadable on the light themes. */
    function boardScrim(a) {
        return themeVar("--board-scrim", "0, 0, 0").replace(/^/, "rgba(") + ", " + a + ")";
    }
    /* Piece colours, cell shading and the danger line (--tetris-* in
       style.css). The light boards swap in deeper hues and a cell edge; every
       other theme keeps CONFIG.COLORS. Parsed once per theme, not per cell. */
    function pieceColors() {
        if (!_themeCache.__colors) {
            var list = themeVar("--tetris-colors", "").split(",")
                .map(function (c) { return c.trim(); })
                .filter(Boolean);
            _themeCache.__colors = list.length >= CONFIG.COLORS.length ? list : CONFIG.COLORS;
        }
        return _themeCache.__colors;
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
        // A fresh Left/Right press moves at once; holding it starts repeating
        // at HORIZONTAL_MOVEMENT_INTERVAL after this long. See createGame().
        HORIZONTAL_REPEAT_DELAY: 150,
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

    // ─── Match start ─────────────────────────────────────────────────────────
    // The AI used to begin playing the moment the model finished loading, so
    // it was already several pieces deep before the player had touched a key
    // and read "Press any key to start". It now waits for that first key, the
    // same way Snake's board does. Module-scope because the loop and the
    // keyboard handler live in different closures.
    var matchStarted = false;

    /* The head-to-head for the current match (shared/results.js), or null
       when there is none: before the first input, when the model had not
       loaded when the human started, and for a game resumed after a reload.
       It begins on the human's first input (after load or after a restart),
       and at that moment the AI board resets so both start level.

       `matchAi` is the AI's state object for its FIRST life in this match.
       Only that life is fed to the match; the AI's auto-restarts afterwards
       are for show. The rules themselves — the AI's unfinished score is not
       a result, so a human who dies first waits until the AI passes them or
       tops out — live in MatchResults.begin(), shared with the other games.
       This used to freeze the AI's live score the moment the human died, so
       dying in the first second beat an AI that had not finished, and
       restarting the AI by hand locked its score in. */
    var match = null;
    var matchAi = null;

    // Visually hidden aria-live line under each board (see game.html).
    function announce(isAI, text) {
        var el = document.getElementById(isAI ? "live-ai" : "live-human");
        if (el) el.textContent = text;
    }

    // Set if the model cannot be fetched or parsed, so the AI board says so
    // instead of reading "Loading AI..." forever.
    var aiLoadFailed = false;

    // Download progress for the "Loading AI" overlay: {got, total} in bytes,
    // total 0 when the server sends no usable Content-Length. Null until the
    // first chunk arrives (and always under file://, which has no download).
    var aiLoadProgress = null;

    // How long GAME OVER stays up before a key or tap may restart the board.
    var RESTART_GRACE = 700;

    // Phones and tablets: no hover, coarse pointer. Only changes the start
    // prompt; the touch controls are wired up regardless.
    var TOUCH = !!(window.matchMedia && window.matchMedia("(hover: none) and (pointer: coarse)").matches);

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
                // Space and Enter on a focused button or link belong to that
                // control. Without this, Space on a speed button hard-dropped
                // the human's piece AND pressed the button. Scroll suppression
                // lives in shared/keyscroll.js.
                var t = e.target;
                if ((key === " " || key === "enter") && t && t.closest &&
                    t.closest("button, a, input, select, textarea, [role='button']")) return;
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
        // Navigation and modifier keys never start a match or restart a board:
        // a keyboard user tabbing to the controls was releasing both boards.
        isIgnoredKey: function(e) {
            var k = e.key;
            if (k === "Tab" || k === "Escape" || k === "Shift" || k === "Control" ||
                k === "Alt" || k === "Meta" || k === "CapsLock") return true;
            return !!(e.ctrlKey || e.metaKey || e.altKey);
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
    /* Board layout, in the canvas's own 300x540 units (the page scales the
       canvas; see createGame's fit()).

       The score used to be painted INSIDE the well, over rows 0-2, so every
       new piece spawned across the big digits and a tall stack ran under
       "High score". The header is now a band of its own above the well and is
       painted last, opaque, so a piece entering from above slides out from
       under it instead of across it. The well keeps its 10x18 cells — they
       are 26 units instead of 30 to make room — and the board keeps its
       300x540 size and 10:18 aspect, so nothing outside the canvas moved. */
    var VIEW_W = 300, VIEW_H = 540;
    var HEADER_H = 72;
    var CELL = (VIEW_H - HEADER_H) / CONFIG.ARENA_HEIGHT;          // 26
    var WELL_W = CELL * CONFIG.ARENA_WIDTH;                         // 260
    var WELL_X = (VIEW_W - WELL_W) / 2;                             // 20
    var WELL_Y = HEADER_H;
    // Overlays (scrims, cards) cover the well from just under the danger
    // line down, as the human board's HTML start overlay does — so the score
    // band and the red line read identically on both boards in every state.
    var SCRIM_Y = WELL_Y + 3 * CELL + 2;                            // 152
    var WELL_MID = SCRIM_Y + (VIEW_H - SCRIM_Y) / 2;                // overlay centre

    function drawMatrix(ctx, matrix, ox, oy, colors) {
        var shine = themeVar("--tetris-cell-shine", "rgba(255,255,255,0.12)");
        var edge  = themeVar("--tetris-cell-edge", "transparent");
        var hasEdge = edge !== "transparent";
        if (hasEdge) { ctx.strokeStyle = edge; ctx.lineWidth = 1; }
        for (var r = 0; r < matrix.length; r++) {
            for (var c = 0; c < matrix[r].length; c++) {
                var v = matrix[r][c];
                if (v === 0) continue;
                var x = WELL_X + (ox + c) * CELL, y = WELL_Y + (oy + r) * CELL;
                ctx.fillStyle = colors[v - 1];
                ctx.fillRect(x, y, CELL, CELL);
                ctx.fillStyle = shine;
                ctx.fillRect(x + 2, y + 2, CELL - 4, CELL - 4);
                if (hasEdge) ctx.strokeRect(x + 0.5, y + 0.5, CELL - 1, CELL - 1);
            }
        }
    }

    /* Largest font no bigger than `size`, and no smaller than `floor`, that
       fits `text` in `maxW` (both in canvas units). Measured once per
       text/size and remembered on the canvas: the score changes a few times a
       second and the loop runs at 60. fit() clears the memo, because resizing
       a canvas resets its context.

       It used to shrink all the way to 6 units, which on a phone-sized board
       drew the AI's loading detail at 7.4 CSS px. It now stops at the floor
       (the caller's minimum, converted to canvas units) and the caller wraps
       whatever still does not fit: see wrapLines() and drawCard(). */
    function fitFont(ctx, text, size, maxW, weight, floor) {
        var memo = ctx._fontMemo || (ctx._fontMemo = {});
        floor = Math.min(size, floor || size);
        var key = text + "|" + size + "|" + maxW + "|" + weight + "|" + floor;
        if (!memo[key]) {
            var px = size;
            ctx.font = weight + px + "px " + CONFIG.FONT_FAMILY;
            while (ctx.measureText(text).width > maxW && px - 1 >= floor) {
                px -= 1;
                ctx.font = weight + px + "px " + CONFIG.FONT_FAMILY;
            }
            memo[key] = weight + px + "px " + CONFIG.FONT_FAMILY;
        }
        return memo[key];
    }

    /* `text` broken at spaces into lines no wider than maxW in the current
       ctx.font, BALANCED: the narrowest width that still needs no more lines
       than maxW does, so "Couldn't load the AI" becomes "Couldn't load / the
       AI" rather than leaving "AI" alone on the second line. A single word
       longer than maxW keeps a line to itself rather than being cut.
       Memoised alongside fitFont. */
    function greedyLines(ctx, words, maxW) {
        var lines = [], cur = "";
        for (var i = 0; i < words.length; i++) {
            var next = cur ? cur + " " + words[i] : words[i];
            if (cur && ctx.measureText(next).width > maxW) { lines.push(cur); cur = words[i]; }
            else cur = next;
        }
        if (cur) lines.push(cur);
        return lines;
    }
    function wrapLines(ctx, text, maxW) {
        var memo = ctx._fontMemo || (ctx._fontMemo = {});
        var key = "wrap|" + ctx.font + "|" + maxW + "|" + text;
        if (memo[key]) return memo[key];
        var words = text.split(" ");
        var lines = greedyLines(ctx, words, maxW);
        if (lines.length > 1) {
            var lo = 0, hi = maxW;
            for (var k = 0; k < 12; k++) {
                var mid = (lo + hi) / 2;
                if (greedyLines(ctx, words, mid).length > lines.length) lo = mid; else hi = mid;
            }
            lines = greedyLines(ctx, words, hi);
        }
        return (memo[key] = lines);
    }

    /* A size in canvas units that never renders smaller than minCss CSS
       pixels. On a phone the board is ~170px wide, a 0.57x scale, and the
       desktop sizes came out at 7px. */
    function textSize(base, minCss, s) { return Math.max(base, minCss / (s || 1)); }

    // Text a visitor needs is never drawn below this many CSS pixels.
    var TEXT_FLOOR = 11;

    // Score, high score and the separator. Opaque, and drawn after the pieces.
    function drawHeader(ctx, state, s) {
        ctx.fillStyle = boardBg();
        ctx.fillRect(0, 0, VIEW_W, HEADER_H);
        ctx.strokeStyle = themeVar("--board-edge", "#2a2a2a");
        ctx.lineWidth = Math.max(1, 1 / s);
        ctx.beginPath();
        ctx.moveTo(0, HEADER_H - ctx.lineWidth / 2);
        ctx.lineTo(VIEW_W, HEADER_H - ctx.lineWidth / 2);
        ctx.stroke();

        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle = boardInk();
        // Kept clear of the restart button on the left, symmetrically.
        ctx.font = fitFont(ctx, "" + state.score, textSize(40, 20, s), 160, "", 20 / s);
        ctx.fillText("" + state.score, VIEW_W / 2, 29);
        /* At the 11px floor a long "High score: 123456" no longer fits beside
           the restart button on a phone-sized board, and the band has no room
           for a second line, so it falls back to the shorter label rather
           than shrinking. It used to shrink to 10px. */
        var size = textSize(13, TEXT_FLOOR, s), floor = TEXT_FLOOR / s, maxW = 180;
        var hs = "High score: " + state.highScore;
        ctx.font = fitFont(ctx, hs, size, maxW, "", floor);
        if (ctx.measureText(hs).width > maxW) {
            hs = "Best: " + state.highScore;
            ctx.font = fitFont(ctx, hs, size, maxW, "", floor);
        }
        ctx.globalAlpha = 0.75;
        ctx.fillText(hs, VIEW_W / 2, 60);
        ctx.globalAlpha = 1;
    }

    function drawDangerLine(ctx) {
        ctx.strokeStyle = themeVar("--tetris-danger", "rgba(255,40,40,0.5)");
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(WELL_X, WELL_Y + 3 * CELL);
        ctx.lineTo(WELL_X + WELL_W, WELL_Y + 3 * CELL);
        ctx.stroke();
    }

    function roundRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        if (ctx.roundRect) { ctx.roundRect(x, y, w, h, r); return; }
        ctx.moveTo(x + r, y);
        ctx.arcTo(x + w, y, x + w, y + h, r);
        ctx.arcTo(x + w, y + h, x, y + h, r);
        ctx.arcTo(x, y + h, x, y, r);
        ctx.arcTo(x, y, x + w, y, r);
        ctx.closePath();
    }

    /* Overlay text (start prompt, loading, game over) on a backing pill, so it
       never prints straight over pieces. lines: [{text, size, min, bold,
       alpha}], centred on the well at cy. `min` is in CSS pixels: a line
       shrinks towards it and then WRAPS, so nothing renders below it. */
    function drawCard(ctx, lines, cy, s) {
        var maxW = WELL_W - 28, w = 0, h = 0, laid = [];
        lines.forEach(function (l) {
            var weight = l.bold ? "bold " : "";
            var min = Math.max(l.min || TEXT_FLOOR, TEXT_FLOOR);
            var size = textSize(l.size, min, s);
            var font = fitFont(ctx, l.text, size, maxW, weight, min / s);
            ctx.font = font;
            /* A match line too long even at the floor breaks after its dash
               first ("You 255 · AI 50 —" / "AI still playing, needs 206 to
               win"), sized for the longer half, rather than leaving one word
               on a line of its own. */
            var parts = [l.text], cut = l.text.indexOf(" \u2014 ");
            if (cut > 0 && ctx.measureText(l.text).width > maxW) {
                parts = [l.text.slice(0, cut + 2), l.text.slice(cut + 3)];
                ctx.font = weight + size + "px " + CONFIG.FONT_FAMILY;
                var longer = ctx.measureText(parts[0]).width > ctx.measureText(parts[1]).width
                    ? parts[0] : parts[1];
                font = fitFont(ctx, longer, size, maxW, weight, min / s);
                ctx.font = font;
            }
            var px = parseFloat(font.replace(/^bold /, ""));
            var wrapped = [];
            parts.forEach(function (part) { wrapped = wrapped.concat(wrapLines(ctx, part, maxW)); });
            wrapped.forEach(function (t, i) {
                // Wrapped continuation lines sit closer than separate entries.
                var lh = px * (i === 0 ? 1.45 : 1.2);
                w = Math.max(w, ctx.measureText(t).width);
                laid.push({ l: l, text: t, font: font, h: lh });
                h += lh;
            });
        });
        var padX = 14, padY = 10;
        ctx.fillStyle = boardScrim(0.82);
        roundRect(ctx, VIEW_W / 2 - w / 2 - padX, cy - h / 2 - padY, w + 2 * padX, h + 2 * padY, 10);
        ctx.fill();
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle = boardInk();
        var y = cy - h / 2;
        laid.forEach(function (it) {
            ctx.font = it.font;
            ctx.globalAlpha = it.l.alpha || 1;
            ctx.fillText(it.text, VIEW_W / 2, y + it.h / 2);
            y += it.h;
        });
        ctx.globalAlpha = 1;
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

    /* The next N pieces, in the order they will actually arrive.

       tetris_env.py keeps a real five-piece queue, so the model was trained on
       the true upcoming pieces. This used to copy the bag and, whenever it ran
       short (five placements in every seven), invent a freshly shuffled refill
       for the tail of the preview. The real refill in newPiece() shuffles
       independently, so up to four of the five "next" pieces the AI saw were
       fiction.

       Topping the real bag up from the FRONT keeps it deep enough to read
       directly. Pieces come off the end with pop(), so everything already in
       the bag is still drawn first and the fresh bag queues behind it: the
       same 7-bag order newPiece() would have produced. */
    function peekNextPieces(bag, n) {
        while (bag.length < n) {
            var fresh = cloneShapes(CONFIG.SHAPES);
            shuffle(fresh);
            Array.prototype.unshift.apply(bag, fresh);
        }
        var result = [];
        for (var i = 0; i < n; i++) result.push(bag[bag.length - 1 - i]);
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
        /* Tetris ships no critic: the 1B-step checkpoint that produced its
           .onnx is gone, and deploy_pages.sh publishes critics for Snake and
           Watermelon only. This used to request tetris_critic.onnx on reveal
           anyway — a 404, a console warning and an empty "expected score from
           here —" row. noValue leaves the row out entirely. */
        noValue: true
    });

    // ─── AI player ───────────────────────────────────────────────────────────

    /* `busy` is the claim of the board that asked for the current move — an
       object token, or null. A restart used to clear a plain flag while the
       old board's inference was still in flight; that stale promise then
       cleared it again under the NEW board, so two decisions could overlap.
       Now only the token's owner can release it (see the loop). `inflight`
       tracks the session itself, so a new board's first run never overlaps
       an abandoned one on the same session. */
    function AIPlayer(session) {
        this.session  = session;
        this.combo    = 0;
        this.busy     = null;
        this.inflight = false;
    }

    AIPlayer.prototype.chooseMove = async function(state, nextPieces) {
        var placements = getValidPlacements(state.arena, state.player.shape);
        if (placements.length === 0) return null;

        var obs = buildObs(state.arena, state.player.shape, nextPieces, this.combo);

        this.inflight = true;
        try {
            var tensor = new ort.Tensor("float32", obs, [1, 238]);
            var results = await this.session.run({ observation: tensor });
            var logits  = results.action_logits.data;

            // The inspector sees the exact tensor the model just consumed.
            // Only the first placements.length logits are legal moves. The
            // softmax used to include the rest, so confidence and the
            // highlighted "best" could show a move the AI can never make.
            if (inspector.isOpen) {
                inspector.update({ obs: obs, logits: logits.subarray(0, placements.length) });
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

            // NOTE: busy stays claimed on purpose — applyMove() kicks off a
            // drop animation instead of locking instantly, and the game loop
            // releases busy once that animation finishes (see createGame()).
            return placements[bestIdx];
        } catch(e) {
            console.error("AI inference error:", e);
            return null;
        } finally {
            this.inflight = false;
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

    /* Remove full rows and return how many went, which the AI's combo needs.
       Plain loops: this runs every frame on both boards. */
    function clearLines(state) {
        var cleared = 0;
        for (var r = CONFIG.ARENA_HEIGHT - 1; r >= 0; r--) {
            var row = state.arena[r], full = true;
            for (var c = 0; c < CONFIG.ARENA_WIDTH; c++) {
                if (!(row[c] > 0)) { full = false; break; }
            }
            if (!full) continue;
            state.arena.splice(r, 1);
            state.arena.unshift(new Array(CONFIG.ARENA_WIDTH).fill(0));
            state.score += CONFIG.scorePoints.LINECLEAR;
            cleared++;
            r++;
        }
        return cleared;
    }

    function toppedOut(state) {
        for (var r = 0; r < 3; r++)
            for (var c = 0; c < CONFIG.ARENA_WIDTH; c++)
                if (state.arena[r][c] > 0) return true;
        return false;
    }

    // ─── Game loop factory ────────────────────────────────────────────────────

    function createGame(canvasEl, isAI, aiPlayer, hooks) {
        hooks = hooks || {};
        // No inline width/height: style.css sizes the canvas (it scales with
        // the viewport), and inline sizes overrode its narrow-screen rule.
        var ctx = canvasEl.getContext("2d");

        /* The backing store follows the canvas's DISPLAYED size times
           devicePixelRatio, and everything is drawn in fixed 300x540 units
           scaled onto it. It used to be fixed at 300x540 x dpr, which was only
           sharp at exactly that size; the board now grows on large screens
           and shrinks on phones. `textScale` is CSS px per unit, for the
           minimum text sizes. Browser zoom and moving to another monitor
           change dpr, so that is checked every frame too. */
        var dpr = 0, cssW = 0, textScale = 1, needFit = true;
        function fit() {
            needFit = false;
            dpr = window.devicePixelRatio || 1;
            cssW = canvasEl.getBoundingClientRect().width || VIEW_W;  // 0 while hidden
            textScale = cssW / VIEW_W;
            canvasEl.width  = Math.round(cssW * dpr);
            canvasEl.height = Math.round(cssW * VIEW_H / VIEW_W * dpr);
            var k = canvasEl.width / VIEW_W;
            ctx.setTransform(k, 0, 0, k, 0, 0);
            ctx._fontMemo = null;    // resizing resets the context's font
        }
        if (window.ResizeObserver) {
            new ResizeObserver(function () { needFit = true; }).observe(canvasEl);
        }

        var SAVE_KEY = "tetris." + (isAI ? "ai" : "human") + ".savedGame";
        var HIGH_KEY = "tetris." + (isAI ? "ai" : "human") + ".highScore";

        function loadHighScore() {
            var v = Store.read(HIGH_KEY, 0);
            return typeof v === "number" && isFinite(v) ? v : 0;
        }

        /* Only the human's board is saved. The AI's used to be too, but the
           AI board resets at the start of every match, so a restored AI game
           was thrown away on the first key; it only ever showed for the
           moment before the player started. Its high score is still kept.
           Any save left by an older build is dropped. */
        if (isAI) Store.clear(SAVE_KEY);

        function saveGame() {
            // A finished board is not worth restoring — drop it and keep only
            // the high score, so a reload starts fresh instead of reopening on
            // a game-over screen.
            if (isAI) {
                // High score only; see above.
            } else if (state.lost || !(state.started || state.resumed)) {
                // Nor is a board nobody has played: saving the untouched
                // board meant a plain reload came back as a "resumed" game
                // that could never be scored.
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

        /* `resumed` marks a game carried over a reload. It is played to the
           end but never scored: the AI board starts afresh when it resumes,
           so the two games did not start level (see startMatch). */
        function restoreGame() {
            if (isAI) return null;
            var saved = Store.read(SAVE_KEY, null);
            if (!isValidSavedState(saved)) return null;
            return {
                player:    saved.player,
                arena:     saved.arena,
                bag:       saved.bag,
                paused:    false,
                lost:      false,
                resumed:   true,
                score:     saved.score || 0,
                highScore: Math.max(saved.score || 0, loadHighScore())
            };
        }

        var state  = restoreGame() || createState(loadHighScore());
        var timers = { lastTime: 0, dropCounter: 0, lockCounter: 0, horizCounter: 0 };
        var aiThinkTimer = 0;
        var AI_THINK_INTERVAL = 300; // ms between AI moves
        var lastLabelScore = null;

        /* A fresh board in place, keeping the high score. Used by the restart
           buttons, restart-after-loss and the AI's auto-restart, none of which
           reload the page any more. The buttons used to reload, so restarting
           the AI also threw away the human's game. */
        function resetBoard() {
            state = createState(Math.max(state.highScore || 0, state.score || 0, loadHighScore()));
            timers = { lastTime: performance.now(), dropCounter: 0, lockCounter: 0, horizCounter: 0 };
            aiThinkTimer = 0;
            // Releases the old board's claim; its in-flight decision, if any,
            // finds the token changed and stands down.
            if (aiPlayer) { aiPlayer.combo = 0; aiPlayer.busy = null; }
            Store.clear(SAVE_KEY);
            if (hooks.onReset) hooks.onReset();
        }

        if (!isAI) {
            /* Left/Right move on the keydown itself, then auto-repeat after
               HORIZONTAL_REPEAT_DELAY. Movement used to come only from the
               loop polling held keys every 76ms, so a press could wait up to
               76ms to register and a quick tap that went up between two polls
               was dropped entirely. These run before the ANY handler that
               starts the match, so the starting key still does nothing. */
            var tapShift = function (dir) {
                return function () {
                    if (!matchStarted || state.lost || state.paused) return;
                    state.player.x += dir;
                    if (hasCollision(state)) state.player.x -= dir;
                    timers.horizCounter = CONFIG.HORIZONTAL_MOVEMENT_INTERVAL
                                        - CONFIG.HORIZONTAL_REPEAT_DELAY;
                };
            };
            CONFIG.controls.LEFT.forEach(function (k) { KB.onPress(k, tapShift(-1)); });
            CONFIG.controls.RIGHT.forEach(function (k) { KB.onPress(k, tapShift(1)); });

            /* Restart after GAME OVER on a deliberate, FRESH keypress once the
               screen has been up RESTART_GRACE ms. OS auto-repeat (e.repeat)
               never counts: holding soft drop through the loss, or mashing
               Space, used to dismiss the screen within half a second — two
               losses were recorded 0.9 s apart. The board then waits for a
               first input again (hooks.onReset), so the restarting key does
               not also start the next game; it is marked on the event so the
               start handler, which runs next, leaves it alone.

               Space or Enter only, as the card says (the wording all three
               games share): arrows still held or mashed from the last piece
               no longer throw the result card away. */
            KB.on(KB.ANY, function (e) {
                if (!state.lost || e.repeat || KB.isIgnoredKey(e)) return;
                if (e.key !== " " && e.key !== "Enter") return;
                if (performance.now() - (state.lostAt || 0) < RESTART_GRACE) return;
                resetBoard();
                e.tetrisRestarted = true;
            });
        }

        function loop(ts) {
            if (needFit || (window.devicePixelRatio || 1) !== dpr) fit();
            var dt = ts - timers.lastTime;
            timers.lastTime = ts;
            if (dt > 200) dt = 200;
            var lockedThisFrame = false;

            if (!state.paused && !state.lost) {
                timers.dropCounter  += dt;
                timers.horizCounter += dt;
                // Do not bank gravity while waiting for the first key, or the
                // piece jumps a row the instant the match starts.
                if (!isAI && !matchStarted) { timers.dropCounter = 0; timers.horizCounter = 0; }

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
                            lockedThisFrame = true;
                            aiPlayer.busy = null;
                        }
                    } else {
                        // AI: think and place every AI_THINK_INTERVAL ms,
                        // shortened by the speed multiplier.
                        aiThinkTimer += dt;
                        if (aiThinkTimer >= AI_THINK_INTERVAL / AI_SPEED &&
                                !aiPlayer.busy && !aiPlayer.inflight) {
                            aiThinkTimer = 0;
                            var nextPieces = peekNextPieces(state.bag, 5);
                            var thinkState = state;
                            var token = aiPlayer.busy = {};
                            aiPlayer.chooseMove(state, nextPieces).then(function(placement) {
                                // The board was reset while the model was
                                // thinking, and the new board owns `busy` now.
                                if (aiPlayer.busy !== token) return;
                                // `state` may also have been replaced. A move
                                // worked out for the old board must not land on
                                // the new one.
                                if (placement && !state.lost && state === thinkState) {
                                    aiPlayer.applyMove(state, placement);
                                } else {
                                    aiPlayer.busy = null;
                                }
                            });
                        }
                    }
                } else if (!isAI && matchStarted) {
                    // Human: gravity + controls. Gated on matchStarted: pieces
                    // used to fall under the "Press any key" overlay, so an
                    // idle tab topped out on its own and recorded a result.
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

                var cleared = clearLines(state);

                // tetris_env.py: combo += 1 on a lock that clears, else 0. The
                // browser never updated it, so the model's combo input sat at
                // zero for every move it ever made here.
                if (isAI && lockedThisFrame && aiPlayer) {
                    aiPlayer.combo = cleared > 0 ? aiPlayer.combo + 1 : 0;
                }
                if (!isAI && cleared > 0) {
                    announce(false, (cleared === 1 ? "Line cleared" : cleared + " lines cleared") +
                                    ". Score " + state.score + ".");
                }

                // Loss. Runs once, on the transition: this block is skipped
                // while state.lost is set.
                if (toppedOut(state)) {
                    state.lost = true;
                    state.lostAt = performance.now();
                    if (state.score > state.highScore) state.highScore = state.score;
                    Store.write(HIGH_KEY, state.highScore);

                    if (isAI) {
                        if (match && state === matchAi) match.aiDied(state.score);
                        announce(true, "The AI topped out with " + state.score + ".");
                    } else {
                        // Only a real match is a result: not a game started
                        // before the AI had loaded, or with no AI at all. The
                        // card reads the match's line every frame, so "AI
                        // still playing, needs N" counts down live.
                        state.match = match;
                        if (match) match.humanDied(state.score);
                        var line = resultLine(state);
                        announce(false, "Game over. Score " + state.score + "." +
                                        (line ? " " + line + "." : ""));
                    }
                }
            }

            // The AI's first life in the match feeds the match its score on
            // every change. Cheap: aiScore() returns at once when unchanged.
            if (isAI && match && state === matchAi && !state.lost) match.aiScore(state.score);

            // AI auto-restart after loss
            if (isAI && state.lost && !state.restarting) {
                state.restarting = true;
                var lostState = state;
                setTimeout(function() {
                    // Skip if the board was already restarted by hand meanwhile.
                    if (state === lostState) resetBoard();
                }, 1500);
            }

            // ── Render ───────────────────────────────────────────────────────
            var s = textScale;
            ctx.fillStyle = boardBg();
            ctx.fillRect(0, 0, VIEW_W, VIEW_H);

            // The well's side walls: the board is wider than the 10 columns.
            ctx.strokeStyle = themeVar("--board-edge", "#2a2a2a");
            ctx.lineWidth = Math.max(1, 1 / s);
            ctx.strokeRect(WELL_X - ctx.lineWidth / 2, WELL_Y - 1,
                           WELL_W + ctx.lineWidth, VIEW_H - WELL_Y + 2);

            // Clipped to the well, so a piece still above it (spawning, or the
            // AI's drop animation) cannot bleed a cell edge under the header.
            ctx.save();
            ctx.beginPath();
            ctx.rect(WELL_X, WELL_Y, WELL_W, VIEW_H - WELL_Y);
            ctx.clip();
            drawMatrix(ctx, state.arena, 0, 0, pieceColors());
            drawDangerLine(ctx);

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
                drawMatrix(ctx, state.player.shape, renderX, renderY, pieceColors());
                ctx.globalAlpha = 1;
            }
            ctx.restore();

            // Tracked live; persisted on loss and in saveGame(), which runs on
            // beforeunload and visibilitychange. Writing on every increase was
            // ~23 synchronous localStorage writes a second during soft drop.
            if (state.score > state.highScore) state.highScore = state.score;

            // Overlays stop short of the danger line. The scrim used to cover
            // it, so it was strong red on the human board and pale pink on the
            // AI board while that one waited to start.
            var waiting = isAI && (!aiPlayer || !aiPlayer.session || !matchStarted);
            if (state.lost || waiting) {
                ctx.fillStyle = boardScrim(state.lost ? 0.65 : 0.75);
                ctx.fillRect(0, SCRIM_Y, VIEW_W, VIEW_H - SCRIM_Y);
            }

            // Game over
            if (state.lost && !waiting) {
                var over = [{ text: isAI ? "AI GAME OVER" : "GAME OVER", size: isAI ? 30 : 36, min: 18, bold: true }];
                // The head-to-head, at full ink like the score. Re-read every
                // frame: a match waiting on the AI changes as it plays.
                var result = isAI ? "" : resultLine(state);
                if (result) over.push({ text: result, size: 16, min: 12 });
                over.push({
                    text: isAI ? "Restarting\u2026" : (TOUCH ? "Tap to play again" : "Press Space or Enter to play again"),
                    size: 15, min: 11, alpha: 0.75
                });
                drawCard(ctx, over, WELL_MID, s);
            }

            // AI waiting overlay — either the model is still loading, or it is
            // ready and holding for the player to start. Without the second
            // case the board just sits there looking broken.
            if (waiting) {
                var loading = !aiPlayer || !aiPlayer.session;
                // Was "Run embed_model.py, then refresh": developer instructions
                // shown to every visitor during every normal load.
                drawCard(ctx, [
                    { text: loading ? (aiLoadFailed ? "Couldn't load the AI" : "Loading AI\u2026") : "AI ready",
                      size: 20, min: 14, bold: true },
                    { text: loading ? (aiLoadFailed ? "Check your connection and refresh" : loadingDetail())
                                    : "Starts with your first move",
                      size: 14, min: 11, alpha: 0.75 }
                ], WELL_MID, s);
            }

            // Last, and opaque: a piece entering the well slides out from
            // under the header rather than across the score.
            drawHeader(ctx, state, s);

            // The canvas's accessible name carries the live score; the aria-live
            // line below the board announces game over and line clears.
            if (state.score !== lastLabelScore) {
                lastLabelScore = state.score;
                canvasEl.setAttribute("aria-label", (isAI ? "AI board" : "Your board") +
                                      ", score " + state.score);
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
            save: saveGame,
            reset: resetBoard,
            setAIPlayer: function(p) { aiPlayer = p; }
        };
    }

    /* The line under the human's GAME OVER: the match's own line (a result,
       "AI still playing, needs N to win", or why it was not scored), or a
       note for a game that was never a match. Empty when the AI never
       loaded, as in the other games. */
    function resultLine(st) {
        if (st.match) return st.match.line();
        if (st.resumed) return "Resumed game, not scored";
        if (st.aiJoinedLate) return "Not scored: the AI loaded after you started";
        return "";
    }

    /* A static "Downloading the model" gave no sign of life for the whole
       download, which on a slow connection reads as stuck. */
    function loadingDetail() {
        var p = aiLoadProgress;
        if (!p) return "Downloading the model";
        var mb = function (n) { return (n / 1048576).toFixed(1); };
        if (p.total > 0 && p.got >= p.total) return "Starting the model\u2026";
        if (p.total > 0) return "Downloading the model \u00b7 " + Math.min(99, Math.floor(100 * p.got / p.total)) + "%";
        return "Downloading the model \u00b7 " + mb(p.got) + " MB";
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

        // After any restart the board waits for a first input again, exactly
        // like the first game — and the AI board pauses with it.
        var humanGame = createGame(humanCanvas, false, null, {
            onReset: function () {
                // A match still running (restart button) or waiting on the AI
                // (play again before it finished) is void: the AI board
                // restarts with the next game.
                if (match) match.cancel("a new game started");
                match = null;
                matchAi = null;
                matchStarted = false;
                setStartHint(false);
                showControls(controlsEl, true, false);
                announce(false, "New game. " + (TOUCH ? "Tap" : "Press any key") + " to start.");
            }
        });

        // Hold the switcher's space from first paint — it mounts only after the
        // model loads, and the page is vertically centred, so a late insert
        // shifts everything. See CheckpointSwitcher.reserve.
        if (typeof CheckpointSwitcher !== "undefined") {
            CheckpointSwitcher.reserve("tetris", document.getElementById("board-ai"), false);
        }

        /* Read by shared/confirm-exit.js: in progress once the first key has
           released the boards and while the human has not lost. The human's
           board is saved on beforeunload/visibilitychange and restored on
           return, so the message says that rather than warning the run will
           be lost. */
        window.gameInProgress = function () {
            try { return matchStarted && !humanGame.getState().lost; }
            catch (e) { return false; }
        };
        window.gameExitMessage = "Leave the game? Your board is saved and will be here when you come back, but a resumed game is not scored.";

        /* The start prompt. A game restored after a reload says it will
           resume, and that it is not scored (see startMatch). */
        function setStartHint(resumed) {
            var keys = controlsEl.querySelector(".hint-keys");
            var touch = controlsEl.querySelector(".hint-touch");
            if (keys) keys.textContent = resumed ? "Press any key to resume \u00b7 not scored"
                                                 : "Press any key to start";
            if (touch) touch.textContent = (resumed ? "Tap to resume (not scored)"
                                                    : "Tap to start") +
                " \u00b7 tap rotates, swipe moves, swipe down drops";
        }
        setStartHint(!!humanGame.getState().resumed);
        showControls(controlsEl, true, false);

        // ── AI board ──────────────────────────────────────────────────────────
        var aiCanvas  = document.getElementById("canvas-ai");
        var aiPlayer  = null;
        // Created ONCE. It used to be created here with no player and then
        // again on the same canvas after the model loaded, leaving the first
        // loop repainting underneath the second forever, with a second set of
        // save handlers attached.
        var aiGame    = createGame(aiCanvas, true, null);

        /* A match starts here, on the human's first input after load or
           after a restart. The AI board restarts with it so both begin level;
           see `match`. No match begins if the model is not loaded yet (the
           human just plays), or for a game resumed after a reload: the AI
           board starts afresh while the human's game is half played, so the
           two would not start level. The AI still plays alongside it. */
        function startMatch() {
            if (matchStarted) return;
            showControls(controlsEl, false, true);
            var resumed = !!humanGame.getState().resumed;
            humanGame.getState().started = true;    // worth saving now; see saveGame
            if (aiPlayer) aiGame.reset();
            if (match) match.cancel("a new game started");
            match = null;
            matchAi = null;
            if (aiPlayer && !resumed && typeof MatchResults !== "undefined") {
                matchAi = aiGame.getState();
                match = MatchResults.begin("tetris", { speed: AI_SPEED });
                // Screen readers hear the outcome once it is settled; the
                // "needs N" countdown in between would be noise.
                match.onChange(function (m) {
                    if (m.state === "done" || m.state === "void") announce(false, m.line() + ".");
                });
            }
            matchStarted = true;   // releases the AI board — see the flag above
            announce(false, resumed ? "Game resumed. It will not be scored." : "Game started.");
        }
        // Not once: every restart needs a fresh start. A held key's OS repeat
        // and the key that just restarted the board do not count.
        KB.on(KB.ANY, function(e){
            if (KB.isIgnoredKey(e) || e.repeat || e.tetrisRestarted) return;
            startMatch();
        });

        // Game actions do nothing until the match has started: key handlers run
        // before the ANY handler, so the key that starts the match (often
        // Space) used to perform its action as well.
        function humanPlayable() {
            var st = humanGame.getState();
            return matchStarted && !st.paused && !st.lost ? st : null;
        }

        // Rotate (W / Up, or tap) — human only
        function humanRotate() {
            var state = humanPlayable();
            if (!state) return;
            var prevX = state.player.x;
            rotateMatrix(state.player.shape, 1);
            var kick = 0, attempts = 0;
            while (hasCollision(state)) {
                state.player.x += kick;
                kick = kick > 0 ? -(kick + 1) : (1 - kick);
                attempts++;
                // Offsets 0, +1, -1, +2, -2, then give up. The old limit scaled
                // with piece width and let an I piece hop four columns through
                // part of the stack.
                if (attempts > 5) {
                    rotateMatrix(state.player.shape, -1);
                    state.player.x = prevX;
                    break;
                }
            }
        }

        // Hard drop (Space, or swipe down) — human only
        function humanHardDrop() {
            var state = humanPlayable();
            if (!state) return;
            var dropped = 0;
            state.player.y += 1;
            while (!hasCollision(state)) { state.player.y++; dropped++; }
            state.player.y--;
            if (dropped > 0) state.score += dropped * CONFIG.scorePoints.HARDDROP;
            lockPiece(state, false);
        }

        // Sideways by n columns (swipe), stopping at the first collision.
        function humanShift(n) {
            var state = humanPlayable();
            if (!state) return;
            var step = n > 0 ? 1 : -1;
            for (var i = 0; i < Math.abs(n); i++) {
                state.player.x += step;
                if (hasCollision(state)) { state.player.x -= step; break; }
            }
        }

        // onPress, not on: holding Up used to spin the piece at the OS key
        // repeat rate, so a slightly long press over-rotated it.
        CONFIG.controls.ROTATE.forEach(function (k) { KB.onPress(k, humanRotate); });
        KB.onPress(" ", humanHardDrop);

        /* Touch. The game was keyboard-only and the match waits for a first
           key, so on a phone both boards sat behind "Press any key to start"
           forever. Tap to rotate, swipe sideways to move (one column per cell
           of travel), swipe down to hard drop; the first touch starts the
           match. touch-action: none keeps a swipe from scrolling the page. */
        humanCanvas.style.touchAction = "none";
        var touchStart = null;
        humanCanvas.addEventListener("pointerdown", function (e) {
            if (e.pointerType === "mouse") return;
            touchStart = { x: e.clientX, y: e.clientY };
        });
        humanCanvas.addEventListener("pointercancel", function () { touchStart = null; });
        humanCanvas.addEventListener("pointerup", function (e) {
            if (e.pointerType === "mouse" || !touchStart) return;
            var dx = e.clientX - touchStart.x, dy = e.clientY - touchStart.y;
            touchStart = null;
            if (!matchStarted) { startMatch(); return; }
            var st = humanGame.getState();
            if (st.lost) {
                if (performance.now() - (st.lostAt || 0) >= RESTART_GRACE) humanGame.reset();
                return;
            }
            var cell = humanCanvas.getBoundingClientRect().width * CELL / VIEW_W;
            if (Math.abs(dx) < 12 && Math.abs(dy) < 12) humanRotate();
            else if (Math.abs(dx) > Math.abs(dy)) humanShift(Math.round(dx / cell) || (dx > 0 ? 1 : -1));
            else if (dy > 0) humanHardDrop();
            else humanRotate();
        });

        // Tapping the AI board ("Tap to start") starts the match too.
        aiCanvas.addEventListener("pointerup", function (e) {
            if (e.pointerType !== "mouse") startMatch();
        });

        /* ── Buttons ───────────────────────────────────────────────────────
           Wired BEFORE the model is awaited. They used to be wired after it,
           so on a slow connection Restart and the speed buttons did nothing
           at all for the length of the download. None of them needs the
           model: restarting a board works without one, and the speed is just
           a number the AI reads once it plays.

           Restart in place. No blur here: a pointer click is blurred by the
           delegated handler below, and keyboard activation keeps focus where
           the keyboard user put it. */
        document.getElementById("restart-human").addEventListener("click", function () {
            humanGame.reset();
        });
        document.getElementById("restart-ai").addEventListener("click", function () {
            /* Restarting the AI during its first life in a match would let
               whoever pressed it pick the AI's score, so the match is void.
               Once that life has ended the AI only plays for show, and a
               restart changes nothing. */
            if (match && !match.aiDone) match.cancel("the AI was restarted");
            aiGame.reset();
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
                // Any speed but 1x voids a match in progress (results.js).
                if (match) match.speed(v);
                var all = speedBox.querySelectorAll("button[data-speed]");
                for (var i = 0; i < all.length; i++) {
                    all[i].classList.toggle("active", all[i] === btn);
                    all[i].setAttribute("aria-pressed", all[i] === btn ? "true" : "false");
                }
            });
        }

        /* A mouse click leaves the button focused, and Space on a focused
           button belongs to the button (see KB.init) — so after picking a
           speed or a model version, Space re-pressed that button instead of
           hard-dropping, until you clicked somewhere else. Drop focus after a
           pointer click; keyboard activation (detail 0) keeps it, so tabbing
           through the controls still works. */
        document.getElementById("arena").addEventListener("click", function(e) {
            if (e.detail === 0) return;
            var b = e.target.closest && e.target.closest("button");
            if (b) b.blur();
        });

        // Try to load ONNX model. The .onnx is preferred over HTTP; the
        // base64 in model_data.js is for opening game.html straight from disk,
        // where fetch() cannot read a sibling file.
        try {
            ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/";

            // .onnx over HTTP, base64 only under file:// — see
            // shared/model-source.js.
            // The progress callback goes fourth, with dataUrl left undefined,
            // so an older model-source.js that takes three arguments just
            // ignores it.
            var src = await modelSource("tetris_ai.onnx", "TETRIS_MODEL_B64", undefined,
                function (got, total) { aiLoadProgress = { got: got, total: total }; });
            var session = await ort.InferenceSession.create(src, {
                executionProviders: ["wasm"]
            });

            aiPlayer = new AIPlayer(session);
            aiGame.setAIPlayer(aiPlayer);
            // The model arrived while the human was already playing: the AI
            // joins in for show, and the game says why it is not scored.
            var hs = humanGame.getState();
            if (matchStarted && !hs.lost && !hs.resumed) hs.aiJoinedLate = true;
            // A handle for tests and the console; the page itself no longer
            // reads the AI board from outside its closure.
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
                CheckpointSwitcher.mount({
                    game: "tetris",
                    container: document.getElementById("board-ai"),
                    initial: session,
                    onSession: function (s) { aiPlayer.session = s; },
                });
                /* The stacked variant, not the compact one. Compact clipped
                   the note to "full strength · av…" at every width, and the
                   board's height now budgets for the stacked control (see
                   --tetris-h in style.css), so there is no column to save. */
            }
        } catch(e) {
            aiLoadFailed = true;
            console.warn("AI model failed to load:", e.message);
        }
    });
})();
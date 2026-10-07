/* The in-game checkpoint switcher: play against an earlier version of the AI.
 *
 * Reads the generated manifest in checkpoints.js (built by
 * tools/build_checkpoints.py) and renders a control under the AI board.
 *
 * WHY THIS EXISTS ALONGSIDE THE DIFFICULTY SETTING
 *
 * Difficulty (settings.js) weakens the SHIPPED model by sampling its moves at a
 * temperature — a good network playing carelessly. This is the other axis: a
 * genuinely earlier, genuinely weaker network from a real point in training.
 * settings.js already records why both are needed — Watermelon has a floor of
 * about 42% of full strength no matter how high the temperature goes, so
 * "make it actually easy" was only ever going to work with an earlier
 * checkpoint. The two dials compose.
 *
 * WHY THE RUNGS ARE FETCHED ON DEMAND
 *
 * A Snake rung is 34 MB. Putting four in the page would mean 136 MB before the
 * first frame — the exact mistake that made these games "stuck on loading" when
 * model_data.js was a render-blocking script. So the page ships only the
 * strongest rung (the model it downloads anyway) and fetches another when it is
 * chosen.
 *
 * Only the shipped session and the active one are kept. Caching every rung
 * meant a visitor who clicked along a ladder held all of them at once — four
 * Snake rungs is ~140 MB of weights in the WebAssembly heap, which never
 * shrinks. Switching back to a released rung re-reads it from the HTTP cache.
 */
(function (global) {
    'use strict';

    /* Its own storage key rather than a field in Settings: the settings picker
     * iterates Settings.DEFAULTS to build its UI, and a nested per-game object
     * would surface there as a stray control. */
    var KEY = 'humanvsai.checkpoint';

    function readChoice(game) {
        try {
            var raw = localStorage.getItem(KEY);
            return (raw ? JSON.parse(raw) : {})[game] || null;
        } catch (e) { return null; }
    }

    /* The chosen rung's FILE goes under a second key, for the model preload
     * in each game.html: that runs in <head>, before this manifest has
     * loaded, and needs a URL rather than an id. model-source.js keeps it in
     * step too, for choices saved before it existed. */
    var FILE_KEY = 'humanvsai.checkpointFile';

    function writeChoice(game, id, file) {
        try {
            var raw = localStorage.getItem(KEY);
            var all = raw ? JSON.parse(raw) : {};
            all[game] = id;
            localStorage.setItem(KEY, JSON.stringify(all));
            if (file) {
                var files = JSON.parse(localStorage.getItem(FILE_KEY) || '{}') || {};
                files[game] = file;
                localStorage.setItem(FILE_KEY, JSON.stringify(files));
            }
        } catch (e) { /* private browsing — the choice just will not persist */ }
    }

    function ladder(game) {
        var g = (global.CHECKPOINTS || {})[game];
        // Fewer than two rungs is nothing to switch between, so show no control
        // at all rather than a single dead button.
        if (!g || !g.rungs || g.rungs.length < 2) return null;
        return g;
    }

    /* Scores span 70 to 119,000 across the three games, so one format will not
     * serve all of them. */
    function fmtScore(n) {
        if (n >= 10000) return Math.round(n / 1000) + 'k';
        if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
        return String(Math.round(n));
    }

    function fmtSteps(n) {
        if (n >= 1e9) return (n / 1e9).toFixed(2).replace(/\.?0+$/, '') + 'B';
        return Math.round(n / 1e6) + 'M';
    }

    function fmtBytes(n) { return (n / 1048576).toFixed(0) + ' MB'; }

    /* The difficulty level in force, as its picker label ("Gentle"). The
     * switcher is the only place on a game page that names the opponent, so
     * it is where the second dial has to show — otherwise a visitor on Gentle
     * reads "full strength" and believes it. */
    function difficulty() {
        var S = global.Settings;
        if (!S || !S.get) return null;
        var id = S.get().difficulty;
        var d = (S.DIFFICULTY || []).filter(function (x) { return x.id === id; })[0];
        return d ? { id: d.id, label: d.label, hint: d.hint } : null;
    }

    /* game -> the rung currently playing, for MatchResults to record. Only
     * mounted ladders appear; a game without one is on its shipped model. */
    var playing = {};

    /* Fetch with progress, then hand the bytes to onnxruntime.
     *
     * InferenceSession.create(url) would download it itself, but reports
     * nothing while it does — for 34 MB that is a page which looks hung.
     * Reading the stream costs one extra copy and buys an honest progress
     * line. Some proxies strip Content-Length, so with no total we count bytes
     * instead of showing a bar frozen at zero. */
    async function fetchWithProgress(url, onProgress) {
        var res = await fetch(url);
        if (!res.ok) throw new Error('HTTP ' + res.status + ' for ' + url);
        if (!res.body) return new Uint8Array(await res.arrayBuffer());

        var total = parseInt(res.headers.get('Content-Length') || '0', 10);
        var reader = res.body.getReader();
        var chunks = [], received = 0;
        for (;;) {
            var r = await reader.read();
            if (r.done) break;
            chunks.push(r.value);
            received += r.value.length;
            onProgress(received, total);
        }
        var out = new Uint8Array(received), at = 0;
        for (var i = 0; i < chunks.length; i++) {
            out.set(chunks[i], at);
            at += chunks[i].length;
        }
        return out;
    }

    /* Mount the switcher.
     *
     *   game        'snake' | 'tetris' | 'watermelon'
     *   container   element to append the control to
     *   onSession   async (session, rung) => void, called whenever the active
     *               model changes, with the live onnxruntime session
     *   initial     the session the page already built for the shipped model,
     *               so selecting the top rung never re-downloads it
     */
    function mount(opts) {
        var game = opts.game;
        /* A game with a single rung (Watermelon, today) still gets the
         * control, minus the buttons: there is nothing to switch between, but
         * the note is the only place on the page that says which model is
         * playing and at what difficulty. */
        var g = (global.CHECKPOINTS || {})[game];
        if (!g || !g.rungs || !g.rungs.length) return null;
        var single = g.rungs.length < 2;

        // Drop the placeholder reserve() left holding this space, now that the
        // real control is about to occupy it.
        if (opts.container) {
            var slot = opts.container.querySelector('.ckpt-reserved');
            if (slot) slot.remove();
        }

        var rungs = g.rungs;
        var top = rungs[rungs.length - 1];   // strongest; the model the site ships
        var sessions = {};                   // id -> session, cached
        if (opts.initial) sessions[top.id] = opts.initial;

        var root = document.createElement('div');
        root.className = 'ckpt' + (single ? ' ckpt--single' : '');
        root.innerHTML = '<div class="ckpt-head"><span class="ckpt-title">' +
            (single ? 'AI OPPONENT' : 'MODEL VERSION') + '</span></div>';

        var row = document.createElement('div');
        row.className = 'ckpt-row';
        root.append(row);

        var note = document.createElement('div');
        note.className = 'ckpt-note';
        // Announced, so a screen reader hears the download progress and the result.
        note.setAttribute('aria-live', 'polite');
        root.append(note);

        var buttons = {};
        var active = null;
        var busy = false;

        if (!single) rungs.forEach(function (r) {
            var b = document.createElement('button');
            b.type = 'button';
            b.className = 'ckpt-btn';
            b.dataset.id = r.id;
            b.innerHTML = '<span class="ckpt-score">' + fmtScore(r.score) + '</span>';
            b.title = r.shipped
                ? 'The model the site ships — already loaded'
                : 'Averages ' + fmtScore(r.score) + ' · ' + fmtBytes(r.bytes)
                  + ' to download';
            b.addEventListener('click', function (e) {
                /* A pointer click leaves focus on the rung, and the game's
                   Space/Enter would then press it again instead of reaching
                   the board. Keyboard activation (detail 0) keeps focus. */
                if (e.detail > 0) b.blur();
                select(r.id);
            });
            row.append(b);
            buttons[r.id] = b;
        });

        function describe(r, extra) {
            var d = difficulty();
            var bits = [r.shipped ? 'shipped model' : 'earlier network'];
            if (d) bits.push(d.label);
            /* The average was measured at argmax. Under any other difficulty
               that is the network's ceiling, not what this opponent will
               score, so say which it is. */
            bits.push('averages ' + fmtScore(r.score) + ' over ' + r.episodes +
                      (r.episodes === 1 ? ' game' : ' games') +
                      (d && d.id !== 'full' ? ' at full strength' : ''));
            /* Steps are shown only for a ladder whose runs form one continuous
             * lineage. For Snake and Watermelon they do not, and a WEAKER
             * checkpoint can honestly carry a larger number — see the note in
             * tools/build_checkpoints.py. */
            if (g.stepsMeaning === 'lineage') bits.push(fmtSteps(r.steps) + ' steps');
            if (r.capped) bits.push('some games hit the length cap');
            return bits.join(' · ') + (extra ? ' — ' + extra : '');
        }

        function paint(id, extra) {
            Object.keys(buttons).forEach(function (k) {
                buttons[k].classList.toggle('is-active', k === id);
            });
            var r = rungs.filter(function (x) { return x.id === id; })[0];
            if (!r) return;
            var prev = playing[game];
            playing[game] = r;
            // MatchResults voids a match whose opponent changes part-way.
            if (prev && prev.id !== r.id) {
                window.dispatchEvent(new CustomEvent('rungchange',
                    { detail: { game: game, id: r.id } }));
            }
            /* The whole sentence is always visible — it wraps rather than
             * truncating, even in the compact variant, because a tooltip is
             * unreachable on touch. */
            note.textContent = describe(r, extra);
            var d = difficulty();
            note.title = d ? 'Difficulty: ' + d.label + ' — ' + d.hint +
                             '. Change it on the game-select screen.' : '';
        }

        /* Difficulty is set on select.html, so on a game page it changes
         * only from another tab (settings.js relays the storage event as
         * 'settingschange') or when the page is restored from the
         * back/forward cache after a visit to the picker. */
        function repaint() { if (!busy && active) paint(active); }
        global.addEventListener('settingschange', repaint);
        global.addEventListener('pageshow', function (e) { if (e.persisted) repaint(); });

        /* Free a rung's session once it is no longer playing. Deferred, because
         * the game may still be awaiting a run() on it when the swap happens;
         * a few seconds is many decisions' worth of slack. */
        function retire(id) {
            if (!id || id === top.id || id === active || !sessions[id]) return;
            var s = sessions[id];
            delete sessions[id];
            setTimeout(function () {
                try { if (s.release) s.release(); } catch (e) { /* already gone */ }
            }, 3000);
        }

        async function select(id) {
            if (busy || id === active) return;
            var previous = active;
            var r = rungs.filter(function (x) { return x.id === id; })[0];
            if (!r) return;

            if (sessions[id]) {
                active = id;
                writeChoice(game, id, r.file);
                paint(id);
                await opts.onSession(sessions[id], r);
                retire(previous);
                return;
            }

            busy = true;
            root.classList.add('is-busy');
            buttons[id].classList.add('is-loading');
            try {
                var bytes = await fetchWithProgress(r.file, function (got, total) {
                    note.textContent = total
                        ? 'downloading ' + Math.min(100, Math.round(100 * got / total)) + '%  (' +
                          fmtBytes(total) + ')'
                        : 'downloading ' + fmtBytes(got) + '…';
                });
                note.textContent = 'starting up…';
                sessions[id] = await ort.InferenceSession.create(bytes, {
                    executionProviders: ['wasm'],
                });
                active = id;
                writeChoice(game, id, r.file);
                paint(id);
                await opts.onSession(sessions[id], r);
                retire(previous);
            } catch (err) {
                console.error('Checkpoint ' + id + ' failed to load:', err);
                /* Leave the previous model playing. A switcher that breaks the
                 * game when a download fails is worse than one that refuses. */
                paint(active, 'that version failed to load — still on the previous one');
            } finally {
                busy = false;
                root.classList.remove('is-busy');
                buttons[id].classList.remove('is-loading');
            }
        }

        (opts.container || document.body).append(root);

        /* Start on whichever rung the page actually loaded. model-source.js
         * honours a remembered choice BEFORE downloading anything, so a
         * returning visitor who picked an earlier version gets that one model
         * and nothing else — this used to start on the shipped model and then
         * switch, which downloaded both (68 MB on Snake) and spent the second
         * download playing the wrong opponent.
         *
         * outcome() is null only when model-source did not choose (file://,
         * where a rung cannot be fetched, or an older model-source.js); then
         * the remembered choice is restored afterwards, as before. When it
         * tried the saved rung and had to fall back, say so instead of
         * retrying the same failing download. */
        var ms = global.modelSource && global.modelSource.outcome &&
                 global.modelSource.outcome(game);
        var start = ms && ms.got && buttons[ms.got] ? ms.got : top.id;
        if (start !== top.id) {
            delete sessions[top.id];
            sessions[start] = opts.initial;
        }
        active = start;
        paint(start, ms && ms.wanted !== ms.got
            ? 'your saved version failed to load — playing the shipped model' : null);
        var startRung = rungs.filter(function (x) { return x.id === start; })[0];
        // Lets the game drop state tied to the shipped model (its critic).
        if (start !== top.id && opts.initial) opts.onSession(opts.initial, startRung);
        var saved = readChoice(game);
        if (!ms && saved && saved !== top.id && buttons[saved]) {
            setTimeout(function () { select(saved); }, 0);
        }

        return {
            select: select,
            current: function () { return active; },
            rung: function () { return playing[game] || null; },
            isShipped: function () {
                var r = rungs.filter(function (x) { return x.id === active; })[0];
                return !r || r.shipped;
            },
        };
    }

    /* Hold the switcher's space before it exists.
     *
     * mount() runs only once the model has loaded, and the control is ~89px
     * tall. The game screen is centred vertically, so that block appearing
     * mid-load grew the AI column and shoved the whole page up by about half
     * its height — the page visibly jerking under the cursor just as someone
     * starts playing. Reserving the space up front costs nothing and removes
     * the shift entirely.
     *
     * Does nothing when the ladder has fewer than two rungs, because then
     * mount() renders no control and there is no space to hold. */
    function reserve(game, container, compact) {
        if (!container || !ladder(game)) return;
        if (container.querySelector('.ckpt, .ckpt-reserved')) return;
        var slot = document.createElement('div');
        // The compact variant is 28px against the stacked one's 89px, so the
        // placeholder has to match the variant the game will actually mount or
        // it trades one layout shift for a smaller one in the other direction.
        slot.className = 'ckpt-reserved' + (compact ? ' ckpt-reserved--compact' : '');
        container.appendChild(slot);
    }

    /* What MatchResults records: the rung playing in `game` right now, or
     * null when no switcher is mounted (no ladder, or the model never
     * loaded) — which means the shipped model, if anything. */
    function playingRung(game) {
        var r = playing[game];
        return r ? { id: r.id, shipped: !!r.shipped } : null;
    }

    global.CheckpointSwitcher = {
        mount: mount, ladder: ladder, reserve: reserve,
        playing: playingRung, difficulty: difficulty,
    };
})(window);

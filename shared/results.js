/* Records finished matches so the Inside page can show a head-to-head tally.
 *
 * Everything stays in this browser's localStorage. Nothing is sent anywhere,
 * there is no account, and clearing site data clears it — which is worth
 * saying plainly on a page that otherwise talks about training runs.
 *
 * Results are recorded by a match object (begin(), below): it settles when
 * both sides' first lives are decided, not at the instant the human dies.
 */
(function (global) {
    'use strict';

    var KEY = 'humanvsai.results';
    var LIMIT = 500;        // plenty for a tally; keeps localStorage small

    function read() {
        try {
            var raw = localStorage.getItem(KEY);
            var parsed = raw ? JSON.parse(raw) : [];
            return Array.isArray(parsed) ? parsed : [];
        } catch (e) {
            return [];
        }
    }

    /* Guards against double-recording. Snake's death animation and Tetris's
     * lost state both persist for several frames, so the call site can fire
     * repeatedly for one match; the game passes a token that changes per
     * match and repeats are ignored. */
    var lastToken = null;

    /* record(game, humanScore, aiScore, token, meta)
     *
     *   meta    optional { difficulty, rung } describing the opponent. Either
     *           field left out is filled in from what is in force right now:
     *           the difficulty from Settings, the rung from the checkpoint
     *           switcher. So existing four-argument calls record both.
     *
     * Each entry gains `difficulty` ('full' | 'strong' | 'fair' | 'gentle'),
     * `rung` (the switcher's rung id, or null for a game with no ladder) and
     * `shipped` (false only when an earlier network played). Readers that
     * only look at game/human/ai, like the Inside tally, are unaffected; a
     * win on Gentle against an early checkpoint is still a win, but it is no
     * longer indistinguishable from one against the full model. */
    function record(game, humanScore, aiScore, token, meta) {
        if (token !== undefined && token === lastToken) return;
        lastToken = token;
        meta = meta || {};

        var difficulty = meta.difficulty;
        if (!difficulty) {
            try { difficulty = global.Settings ? global.Settings.get().difficulty : null; }
            catch (e) { difficulty = null; }
        }
        var rung = meta.rung;
        var shipped = true;
        if (rung === undefined) {
            var p = global.CheckpointSwitcher && global.CheckpointSwitcher.playing
                ? global.CheckpointSwitcher.playing(game) : null;
            rung = p ? p.id : null;
            shipped = p ? p.shipped : true;
        } else if (meta.shipped !== undefined) {
            shipped = !!meta.shipped;
        }

        var all = read();
        all.push({
            game: game,
            human: Math.round(humanScore) || 0,
            ai: Math.round(aiScore) || 0,
            difficulty: difficulty || 'full',
            rung: rung === undefined ? null : rung,
            shipped: shipped,
            at: new Date().toISOString(),
        });
        if (all.length > LIMIT) all = all.slice(all.length - LIMIT);

        try {
            localStorage.setItem(KEY, JSON.stringify(all));
        } catch (e) {
            /* private browsing or quota — a missing tally must never break a game */
        }
    }

    /* ── A match ────────────────────────────────────────────────────────────
     *
     * One object per match, shared by all three games so the rules cannot
     * drift between them (they did: each game froze the AI's UNFINISHED score
     * when the human died, so dying in the first second beat an AI that
     * averages 107k, and restarting the AI locked in a 0).
     *
     *   var m = MatchResults.begin(game, { speed: 1 });   // human's first input
     *   m.aiScore(n)      the AI's live score in its first life (every change,
     *                     or every frame — cheap)
     *   m.aiDied(n)       the AI's first life ended at n
     *   m.humanDied(n)    the human's game ended at n
     *   m.cancel(reason)  void it: "the AI was restarted", "a new game started"
     *   m.state           'playing' | 'waiting' | 'done' | 'void'
     *   m.line()          one line for the human's game-over card
     *   m.onChange(fn)    called whenever state or line() changes
     *
     * The rules. The AI's score can only go up, so a human who dies first has
     * not won yet: the match WAITS until the AI either passes the human (AI
     * wins at once) or ends its first life (compare). The AI dying first just
     * fixes its score. Changing the opponent mid-match (model version,
     * difficulty, AI speed) voids it, and so does any speed but 1x, because
     * both make the result about something other than this model.
     */
    function begin(game, opts) {
        opts = opts || {};
        var p = global.CheckpointSwitcher && global.CheckpointSwitcher.playing
            ? global.CheckpointSwitcher.playing(game) : null;
        var diff = null;
        try { diff = global.Settings ? global.Settings.get().difficulty : null; }
        catch (e) { diff = null; }

        var m = {
            state: 'playing', human: null, ai: 0, aiDone: false,
            winner: null, reason: '', listeners: [],
            meta: { difficulty: diff || 'full', rung: p ? p.id : null,
                    shipped: p ? p.shipped : true },
        };
        var token = {};

        function changed() {
            m.listeners.forEach(function (fn) { try { fn(m); } catch (e) {} });
        }
        function finish(aiFinal) {
            m.ai = aiFinal;
            m.state = 'done';
            m.winner = m.human > aiFinal ? 'human' : m.human < aiFinal ? 'ai' : 'draw';
            record(game, m.human, aiFinal, token, m.meta);
            detach();
            changed();
        }
        function cancel(reason) {
            if (m.state === 'done' || m.state === 'void') return;
            m.state = 'void';
            m.reason = reason || 'the opponent changed';
            detach();
            changed();
        }
        function onSettings(e) {
            var d = e && e.detail && e.detail.difficulty;
            if (d && d !== m.meta.difficulty) cancel('the difficulty changed');
        }
        function onRung(e) {
            if (e.detail && e.detail.game === game) cancel('the model version changed');
        }
        function detach() {
            global.removeEventListener('settingschange', onSettings);
            global.removeEventListener('rungchange', onRung);
        }
        global.addEventListener('settingschange', onSettings);
        global.addEventListener('rungchange', onRung);

        m.aiScore = function (n) {
            n = Math.round(n) || 0;
            if (m.aiDone || m.state === 'done' || m.state === 'void' || n === m.ai) return;
            m.ai = n;
            if (m.state === 'waiting' && n > m.human) finish(n);
            else if (m.state === 'waiting') changed();     // "needs N" moved
        };
        m.aiDied = function (n) {
            if (m.aiDone || m.state === 'done' || m.state === 'void') return;
            m.aiDone = true;
            m.ai = Math.round(n) || 0;
            if (m.state === 'waiting') finish(m.ai);
        };
        m.humanDied = function (n) {
            if (m.state !== 'playing') return;
            m.human = Math.round(n) || 0;
            if (m.aiDone || m.ai > m.human) finish(m.ai);
            else { m.state = 'waiting'; changed(); }
        };
        m.cancel = cancel;
        // Once the AI's first life is over its match score is fixed, so a
        // speed change or an AI restart after that cannot affect the result.
        // Every game calls these rather than cancel(), so the rule is one
        // rule; it had already drifted between games twice.
        m.speed = function (s) {
            if (s !== 1 && !m.aiDone) cancel('the AI speed changed');
        };
        m.aiRestarted = function () {
            if (!m.aiDone) cancel('the AI was restarted');
        };
        /* The human pressed restart or "play again". Returns true while the
         * match is still WAITING on the AI: the game must keep the AI board
         * running until it settles (a waiting match only exists when the
         * human is ahead, so cancelling it would throw away exactly their
         * wins). A match still being played is void. */
        m.humanRestarted = function () {
            if (m.state === 'playing') cancel('you restarted');
            return m.state === 'waiting';
        };
        /* The score the human has to beat once the AI's first life is over,
         * for the AI board to show while its later lives play for show. */
        m.target = function () {
            return m.aiDone && m.state === 'playing' ? m.ai : null;
        };
        m.onChange = function (fn) { m.listeners.push(fn); };
        m.line = function () {
            if (m.state === 'void') return 'Not scored: ' + m.reason;
            if (m.human === null) return '';
            var head = 'You ' + m.human + ' · AI ' + m.ai + ' — ';
            if (m.state === 'waiting') {
                return head + 'AI still playing, needs ' + (m.human - m.ai + 1) + ' to win';
            }
            return head + (m.winner === 'human' ? 'You win'
                         : m.winner === 'ai' ? 'AI wins' : 'Draw');
        };

        if (opts.speed !== undefined && opts.speed !== 1) cancel('AI speed is not 1x');
        return m;
    }

    /* A match that was never scored, with the same face as begin()'s, so a
     * game can hold one in `match` and show its line(): a game resumed after
     * a reload, or one the model arrived for after the human had started. */
    var REASONS = {
        resumed: 'resumed game',
        late: 'the AI loaded after you started',
    };
    function unscored(reason) {
        var noop = function () {};
        var why = REASONS[reason] || reason;
        return {
            state: 'void', reason: why, human: null, ai: 0, aiDone: true,
            aiScore: noop, aiDied: noop, humanDied: noop, cancel: noop,
            speed: noop, aiRestarted: noop, onChange: noop,
            humanRestarted: function () { return false; },
            target: function () { return null; },
            line: function () { return 'Not scored: ' + why; },
        };
    }

    global.MatchResults = { record: record, read: read, begin: begin, unscored: unscored };
})(window);

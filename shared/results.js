/* Records finished matches so the Inside page can show a head-to-head tally.
 *
 * Everything stays in this browser's localStorage. Nothing is sent anywhere,
 * there is no account, and clearing site data clears it — which is worth
 * saying plainly on a page that otherwise talks about training runs.
 *
 * A match is recorded when the HUMAN's game ends, capturing both scores at
 * that instant. The AI usually plays on afterwards (Snake and Tetris restart
 * theirs automatically), so waiting for both to finish would either never fire
 * or compare a finished human against an AI on its third life.
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

    function record(game, humanScore, aiScore, token) {
        if (token !== undefined && token === lastToken) return;
        lastToken = token;

        var all = read();
        all.push({
            game: game,
            human: Math.round(humanScore) || 0,
            ai: Math.round(aiScore) || 0,
            at: new Date().toISOString(),
        });
        if (all.length > LIMIT) all = all.slice(all.length - LIMIT);

        try {
            localStorage.setItem(KEY, JSON.stringify(all));
        } catch (e) {
            /* private browsing or quota — a missing tally must never break a game */
        }
    }

    global.MatchResults = { record: record, read: read };
})(window);

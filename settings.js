/* Site-wide settings, shared by the select page and all three games.
 *
 * Stored in localStorage under one key, so a choice made on the select screen
 * is already in effect by the time a game page loads. Nothing is sent
 * anywhere.
 *
 * DIFFICULTY IS A SAMPLING TEMPERATURE, not a speed or a handicap.
 *
 * Each model outputs one score (a "logit") per possible move, and the games
 * have always taken the highest — so the AI plays its single best move every
 * time and never errs. Dividing those scores by a temperature T before the
 * softmax stretches or squashes the gaps between them:
 *
 *     T -> 0   the gaps become infinite; the best move gets all the
 *              probability. Identical to what the site did before.
 *     T = 1    the distribution the policy actually learned.
 *     T > 1    gaps shrink; weaker moves start winning the draw.
 *
 * The reason this beats the obvious alternatives: slowing the AI down does not
 * make it play worse, and picking a random move x% of the time produces
 * nonsense rather than a weaker opponent. Temperature makes it choose moves it
 * rated SECOND best — still reasonable, just not optimal. It plays like a
 * weaker player instead of a broken one, from the same model file, with no
 * extra download.
 */
(function (global) {
    'use strict';

    var KEY = 'humanvsai.settings';

    /* Levels rather than a raw number: "T = 1.5" means nothing to a player.
     * The temperatures are per game because the games punish a mistake very
     * differently — one bad turn ends Snake outright, while a poor Watermelon
     * drop is usually survivable, so the same T does not cost them the same. */
    var DIFFICULTY = [
        { id: 'full',   label: 'Full strength',
          hint: 'Always its best move — the model exactly as trained' },
        { id: 'strong', label: 'Strong',
          hint: 'Occasionally takes its second choice' },
        { id: 'fair',   label: 'Fair',
          hint: 'Plays its learned distribution rather than always the best' },
        { id: 'gentle', label: 'Gentle',
          hint: 'Makes real mistakes — beatable' },
    ];

    /* Snake's are MEASURED — tools/temperature_sweep.py, 12 fixed-seed games
     * at each value, scored as a percentage of full strength:
     *
     *     T 0.00  154.6  100%     T 1.00  108.0   70%
     *     T 0.50  149.8   97%     T 1.50   85.2   55%
     *     T 0.75  121.4   79%     T 3.00   30.5   20%
     *
     * The curve falls smoothly with no cliff, which is what makes temperature
     * usable as a difficulty dial at all — worth checking rather than
     * assuming, since a game you die in instantly could easily have gone from
     * unbeatable to hopeless with nothing in between.
     *
     * Note how little it takes: changing 8% of moves halves the score. That is
     * also why "just make it play worse" is hard to do by hand.
     *
     * Tetris and Watermelon are still ESTIMATES pending their own sweep. Both
     * survive a bad move far better than Snake, so they should tolerate higher
     * temperatures for the same drop in strength — the values below assume
     * that and have not yet been verified.
     */
    var TEMPERATURES = {
        snake:      { full: 0, strong: 0.75, fair: 1.50, gentle: 3.00 },
        tetris:     { full: 0, strong: 1.00, fair: 2.00, gentle: 3.50 },
        watermelon: { full: 0, strong: 1.00, fair: 2.00, gentle: 3.50 },
    };

    var DEFAULTS = {
        // Full strength by default: the site has always played this way, and
        // quietly weakening the opponent would misrepresent the model the rest
        // of the site is busy explaining. The control is one click away.
        difficulty: 'full',
        inspector: true,
    };

    function read() {
        try {
            var raw = localStorage.getItem(KEY);
            var parsed = raw ? JSON.parse(raw) : {};
            var out = {};
            for (var k in DEFAULTS) {
                out[k] = Object.prototype.hasOwnProperty.call(parsed, k)
                    ? parsed[k] : DEFAULTS[k];
            }
            return out;
        } catch (e) {
            return JSON.parse(JSON.stringify(DEFAULTS));
        }
    }

    function write(patch) {
        var next = read();
        for (var k in patch) next[k] = patch[k];
        try {
            localStorage.setItem(KEY, JSON.stringify(next));
        } catch (e) { /* private browsing — settings just will not persist */ }
        return next;
    }

    /* The temperature this game should sample at right now. 0 means "take the
     * best move", which callers can treat as the fast path. */
    function temperature(game) {
        var level = read().difficulty;
        var table = TEMPERATURES[game] || TEMPERATURES.watermelon;
        var t = table[level];
        return typeof t === 'number' ? t : 0;
    }

    /* Pick an action from raw logits, honouring the current difficulty.
     * `allowed` optionally limits the choice to the first N actions, which
     * Tetris needs — it only considers as many logits as there are legal
     * placements. */
    function chooseAction(game, logits, allowed) {
        var n = allowed === undefined ? logits.length
                                      : Math.min(allowed, logits.length);
        if (n <= 0) return 0;

        var t = temperature(game);
        var best = 0;
        for (var i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        if (t <= 0) return best;

        // Subtract the max before exponentiating — logits can be large enough
        // that exp() overflows to Infinity and every probability becomes NaN.
        var max = logits[best];
        var probs = new Array(n);
        var sum = 0;
        for (var j = 0; j < n; j++) {
            probs[j] = Math.exp((logits[j] - max) / t);
            sum += probs[j];
        }
        if (!(sum > 0) || !isFinite(sum)) return best;   // degenerate: play safe

        var r = Math.random() * sum;
        for (var k = 0; k < n; k++) {
            r -= probs[k];
            if (r <= 0) return k;
        }
        return best;
    }

    global.Settings = {
        DIFFICULTY: DIFFICULTY,
        TEMPERATURES: TEMPERATURES,
        DEFAULTS: DEFAULTS,
        get: read,
        set: write,
        temperature: temperature,
        chooseAction: chooseAction,
    };
})(window);

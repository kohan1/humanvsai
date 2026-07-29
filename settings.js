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
     * Watermelon is MEASURED too, and needs FAR higher temperatures for the
     * same effect:
     *
     *     T 0   1019  100%      T 6    624   61%
     *     T 3    807   76%      T 9    444   44%
     *     T 4    652   64%      T 20   432   42%
     *
     * At T=2 it was still at 96% with a quarter of its moves randomised. Suika
     * forgives a bad drop in a way Snake does not — column 11 instead of 12
     * rarely matters — so randomising the choice costs much less.
     *
     * It also has a FLOOR at about 42%: even at T=20, with 92% of moves
     * effectively random, fruit still merge by accident and the score still
     * accumulates. Watermelon therefore cannot be made as weak as Snake this
     * way, and "Gentle" here means noticeably worse rather than bad. Making it
     * genuinely easy needs an early, actually-weaker checkpoint.
     *
     * Tetris is measured too, through onnxruntime — its training checkpoint no
     * longer exists, so the exported .onnx is the only copy left:
     *
     *     T 0     119.0k  100%     T 0.75   40.0k   34%
     *     T 0.25   98.8k   83%     T 1.50   13.8k   12%
     *     T 0.50   86.9k   73%     T 3.00    2.9k    2%
     *
     * It is the most fragile of the three — 16% of moves changed costs it 88%
     * of its score. A tetris board punishes one bad placement for the rest of
     * the game, where Snake at least dies cleanly and Watermelon shrugs.
     * Its scores are also wildly skewed (mean 119k against a median of 57k),
     * so treat these percentages as a shape, not as precise figures.
     *
     * Three games, three completely different curves — which is exactly why
     * one shared temperature ladder would have been wrong.
     */
    var TEMPERATURES = {
        snake:      { full: 0, strong: 0.75, fair: 1.50, gentle: 3.00 },
        tetris:     { full: 0, strong: 0.35, fair: 0.70, gentle: 1.50 },
        watermelon: { full: 0, strong: 3.00, fair: 6.00, gentle: 10.00 },
    };

    /* Each theme is a palette (themes.css) plus a background renderer
     * (mesh.js). The swatch colours here are only for the picker. */
    var THEMES = [
        { id: 'mesh',   label: 'Mesh',
          swatch: 'linear-gradient(135deg,#080808 0%,#141a24 60%,#cfe4ff 190%)' },
        { id: 'paper',  label: 'Paper',
          swatch: 'linear-gradient(90deg,#ffffff 0%,#ffffff 72%,#000000 72%)' },
        { id: 'carbon', label: 'Carbon',
          swatch: 'linear-gradient(90deg,#000000 0%,#000000 72%,#ffffff 72%)' },
    ];

    var DEFAULTS = {
        // Full strength by default: the site has always played this way, and
        // quietly weakening the opponent would misrepresent the model the rest
        // of the site is busy explaining. The control is one click away.
        difficulty: 'full',
        inspector: true,
        theme: 'mesh',
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

    /* Applied to <html> so CSS and mesh.js both see it. Called on every page
     * as early as possible — see the inline snippet in each <head>, which runs
     * before first paint so a non-default theme never flashes the dark one. */
    function applyTheme(id) {
        var t = id || read().theme;
        document.documentElement.setAttribute('data-theme', t);
        return t;
    }

    global.Settings = {
        THEMES: THEMES,
        applyTheme: applyTheme,
        DIFFICULTY: DIFFICULTY,
        TEMPERATURES: TEMPERATURES,
        DEFAULTS: DEFAULTS,
        get: read,
        set: write,
        temperature: temperature,
        chooseAction: chooseAction,
    };
})(window);

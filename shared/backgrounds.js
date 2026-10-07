/* Cursor-reactive backgrounds.
 *
 * One renderer per theme. Each takes the canvas and an options object, draws
 * until torn down, and returns a teardown function — the same contract
 * initLattice in mesh.js uses, so mesh.js can dispatch between all of them.
 *
 * Shared rules, learned the hard way over three attempts:
 *
 *   - ONE colour, read from --particle-rgb, so the palette carries the theme
 *     and the motion carries the interest. Mixing several colours into the
 *     motion is what made the earlier attempts read as decoration.
 *   - No library. Pages serves under a strict CSP and the games already run
 *     WASM inference; a bundled framework for a background is not worth it.
 *   - Particle counts scale with viewport area, so a laptop does not render
 *     the same 2000 particles a 4K monitor needs.
 */

/* Cheap 2D value noise. A real simplex implementation is ~200 lines and this
 * is a background — smoothstep-interpolated value noise is visually
 * indistinguishable here and fits in twenty. Deterministic, so the field is
 * the same every load. */
function makeNoise(seed) {
    const P = new Uint8Array(512);
    let s = seed || 1;
    const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
    const perm = Array.from({ length: 256 }, (_, i) => i);
    for (let i = 255; i > 0; i--) {
        const j = Math.floor(rnd() * (i + 1));
        const t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    for (let i = 0; i < 512; i++) P[i] = perm[i & 255];

    const fade = (t) => t * t * (3 - 2 * t);
    const lerp = (a, b, t) => a + (b - a) * t;
    const grad = (h) => (h & 255) / 255 * 2 - 1;

    return function (x, y) {
        const xi = Math.floor(x) & 255, yi = Math.floor(y) & 255;
        const xf = x - Math.floor(x), yf = y - Math.floor(y);
        const u = fade(xf), v = fade(yf);
        const aa = grad(P[P[xi] + yi]);
        const ba = grad(P[P[xi + 1] + yi]);
        const ab = grad(P[P[xi] + yi + 1]);
        const bb = grad(P[P[xi + 1] + yi + 1]);
        return lerp(lerp(aa, ba, u), lerp(ab, bb, u), v);
    };
}

/* Read once per theme, not every frame. Each renderer called these inside its
 * draw loop: a style resolution per frame for values that only change with
 * data-theme, and a theme change remounts the renderer anyway. */
let _bgColours = null;
new MutationObserver(() => { _bgColours = null; })
    .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
function bgColours() {
    if (!_bgColours) {
        const s = getComputedStyle(document.documentElement);
        _bgColours = {
            particle: s.getPropertyValue('--particle-rgb').trim() || '235, 235, 235',
            bg: s.getPropertyValue('--bg-rgb').trim() || '8, 8, 8',
        };
    }
    return _bgColours;
}
function particleRgb() { return bgColours().particle; }
function themeBgRgb() { return bgColours().bg; }

/* Batching. Every renderer below used to set a fresh rgba() fillStyle or
 * strokeStyle and issue a separate fill/stroke PER PARTICLE — up to 5000 a
 * frame, each one a colour-string parse and a draw call, on the same thread
 * as the games' physics and inference. Alpha is instead quantised into a
 * handful of levels and every particle of a level goes into one Path2D, so a
 * frame is ~16 draw calls. 16 levels is a step of ~0.05 alpha at most, which
 * a 1-3px mark cannot show. */
const BG_LEVELS = 16;
function bgBuckets() { return new Array(BG_LEVELS + 1); }
function bgLevel(a, max) {
    return Math.max(0, Math.min(BG_LEVELS, Math.round(a / max * BG_LEVELS)));
}
function bgPath(buckets, k) { return buckets[k] || (buckets[k] = new Path2D()); }
function bgFlush(ctx, buckets, rgb, max, stroke) {
    for (let k = 1; k <= BG_LEVELS; k++) {
        if (!buckets[k]) continue;
        const style = 'rgba(' + rgb + ',' + (k / BG_LEVELS * max).toFixed(3) + ')';
        if (stroke) { ctx.strokeStyle = style; ctx.stroke(buckets[k]); }
        else { ctx.fillStyle = style; ctx.fill(buckets[k]); }
    }
}

/* Boilerplate every renderer needs: sizing, pointer tracking, teardown.
 *
 * opts.maxFps caps the PAINT rate (unset: every frame). Behind a game board
 * the backgrounds are ambient, and 30 fps halves their share of the main
 * thread at no visible cost. The simulation still steps on every animation
 * frame — each renderer's draw(now, paint) integrates its motion always and
 * touches the canvas only when `paint` is true — because all of them advance
 * a fixed amount per call, and skipping calls would slow the motion down
 * rather than just make it coarser. */
function bgHarness(cvs, setup, opts) {
    const ctx = cvs.getContext('2d');
    const state = { W: 0, H: 0, mx: -9999, my: -9999, raf: 0, down: false };

    /* prefers-reduced-motion: settle for a moment, then hold a still frame.
     * The pointer still wakes it for a short burst, because that motion is the
     * visitor's own doing rather than something moving by itself (WCAG 2.2.2
     * is about content that animates unprompted and cannot be paused). */
    const reduce = !!(window.matchMedia &&
                      window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    let budget = reduce ? 90 : Infinity;
    const wake = () => {
        if (!reduce) return;
        budget = Math.max(budget, 45);
        if (!state.raf) state.raf = requestAnimationFrame(frame);
    };
    const onMove = (e) => { state.mx = e.clientX; state.my = e.clientY; wake(); };
    const onTouch = (e) => {
        if (!e.touches.length) return;
        state.mx = e.touches[0].clientX; state.my = e.touches[0].clientY;
        wake();
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('touchmove', onTouch, { passive: true });

    // Renderers that move with something other than the pointer (dispersion
    // follows the page's scroll) need to restart a reduced-motion loop too.
    state.wake = wake;
    const api = setup(ctx, state);

    function resize() {
        // The canvas box, not innerWidth — innerWidth includes the scrollbar
        // while this fixed inset:0 canvas does not, and the mismatch made the
        // browser rescale every background by the scrollbar's width.
        // CSS pixels, not device pixels: see the measurement above
        // initLattice in mesh.js.
        state.W = cvs.width = cvs.clientWidth || window.innerWidth;
        state.H = cvs.height = cvs.clientHeight || window.innerHeight;
        if (api.resize) api.resize();
        wake();
    }
    window.addEventListener('resize', resize);
    resize();

    // bgPaceMs lives in mesh.js (it also paces the lattice); fall back to the
    // plain cap if that file did not load.
    const pace = () => typeof bgPaceMs === 'function' ? bgPaceMs(opts)
        : (opts && opts.maxFps ? 1000 / opts.maxFps - 2 : 0);
    let last = -Infinity;
    function frame(now) {
        const minDt = pace();
        const paint = !minDt || now - last >= minDt;
        if (paint) last = now;
        api.draw(now, paint);
        // Only painted frames spend the reduced-motion budget — see mesh.js.
        if (paint) budget--;
        state.raf = budget > 0 ? requestAnimationFrame(frame) : 0;
    }
    if (!state.raf) state.raf = requestAnimationFrame(frame);

    return function () {
        cancelAnimationFrame(state.raf);
        if (api.destroy) api.destroy();
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('touchmove', onTouch);
        window.removeEventListener('resize', resize);
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.globalAlpha = 1;
        ctx.clearRect(0, 0, cvs.width, cvs.height);
    };
}

/* ── Flow ───────────────────────────────────────────────────────────────────
 * Particles drift along a noise field, leaving trails. The cursor ROTATES the
 * field locally, so the whole current bends around it and straightens out
 * again after you pass — the disturbance outlives the pointer, which is what
 * separates it from a spotlight.
 *
 * Trails come from painting a translucent background each frame instead of
 * clearing, so old positions fade rather than vanish. */
function initFlow(cvs, opts) {
    const o = opts || {};
    const INTENSITY = o.intensity === undefined ? 1 : o.intensity;
    const SCALE = 0.0016;      // noise zoom — smaller is smoother
    const SPEED = 34;
    const SWIRL = 240;         // how far the cursor bends the field
    /* Trail persistence. See the note in draw(): this is an ERASE strength, not
       a blend, so it converges to nothing instead of stalling. 0.05 at 60fps
       leaves a trail visible for roughly two seconds. */
    const FADE = 0.05;

    return bgHarness(cvs, (ctx, st) => {
        const noise = makeNoise(20260729);
        let parts = [];
        let t = 0;
        let sweep = 0;
        let pending = new Path2D();   // segments since the last paint
        let skipped = 0;              // frames since the last paint

        function seed(n) {
            parts = Array.from({ length: n }, () => ({
                x: Math.random() * st.W,
                y: Math.random() * st.H,
                life: Math.random() * 200,
            }));
        }

        return {
            resize() {
                const n = Math.round(
                    Math.min(1600, (st.W * st.H) / 1400) * INTENSITY);
                seed(n);
                ctx.clearRect(0, 0, st.W, st.H);
            },
            draw(now, paint) {
                t += 0.0016;
                skipped++;

                for (const p of parts) {
                    let a = noise(p.x * SCALE, p.y * SCALE + t) * Math.PI * 3;

                    const dx = p.x - st.mx, dy = p.y - st.my;
                    const d = Math.sqrt(dx * dx + dy * dy);
                    if (d < SWIRL) {
                        // Rotate the field toward tangential near the pointer.
                        const w = (1 - d / SWIRL) * (1 - d / SWIRL);
                        a += Math.atan2(dy, dx) * w * 1.6 + w * 1.9;
                    }

                    const nx = p.x + Math.cos(a) * SPEED * 0.06;
                    const ny = p.y + Math.sin(a) * SPEED * 0.06;
                    pending.moveTo(p.x, p.y);
                    pending.lineTo(nx, ny);
                    p.x = nx; p.y = ny;

                    if (++p.life > 260 || p.x < -10 || p.x > st.W + 10 ||
                        p.y < -10 || p.y > st.H + 10) {
                        p.x = Math.random() * st.W;
                        p.y = Math.random() * st.H;
                        p.life = 0;
                    }
                }
                if (!paint) return;

                /* ERASE the old frame instead of painting the background over
                   it at low alpha.
                   Blending toward the background looks equivalent but does not
                   converge: colour is 8-bit, so once a trail is within about
                   18 levels of the background the per-frame change rounds to
                   zero and the trail stops fading. The residue never clears,
                   and because the particles keep moving it accumulates until
                   the whole page is a haze — which is exactly what was
                   happening.
                   destination-out multiplies the existing ALPHA by (1 - FADE)
                   instead, and that does reach zero, so trails genuinely
                   disappear. The canvas is transparent, so the page's own
                   background shows through rather than a painted copy of it. */
                ctx.globalCompositeOperation = 'destination-out';
                // One erase standing in for every frame since the last paint,
                // so trails last as long in time at any paint rate.
                const fade = 1 - Math.pow(1 - FADE, skipped);
                ctx.fillStyle = 'rgba(0, 0, 0, ' + fade.toFixed(4) + ')';
                ctx.fillRect(0, 0, st.W, st.H);

                /* A multiply still stalls at the bottom of the 8-bit range:
                   alpha 9 * 0.95 = 8.55, which rounds back to 9 and sticks, so
                   every pixel the particles ever crossed kept a permanent
                   veil — measured mean alpha 11 with a floor of 9 everywhere.
                   A harder erase every 8th frame drags that floor down to ~2
                   (under 1% — invisible) while costing live trails only a few
                   percent of their length. */
                if ((sweep += skipped) >= 8) {
                    sweep %= 8;
                    ctx.fillStyle = 'rgba(0, 0, 0, 0.22)';
                    ctx.fillRect(0, 0, st.W, st.H);
                }
                ctx.globalCompositeOperation = 'source-over';

                const rgb = particleRgb();
                ctx.strokeStyle = 'rgba(' + rgb + ',' + (0.5 * INTENSITY).toFixed(3) + ')';
                ctx.lineWidth = 1;
                ctx.stroke(pending);
                pending = new Path2D();
                skipped = 0;
            },
        };
    }, o);
}

/* ── Filings ────────────────────────────────────────────────────────────────
 * A dense grid of short segments, each aligned to a slowly-turning field. The
 * cursor is a magnetic pole: segments within reach swing to point at it, then
 * relax back. Iron filings around a magnet. */
function initFilings(cvs, opts) {
    const o = opts || {};
    const INTENSITY = o.intensity === undefined ? 1 : o.intensity;
    const GAP = 26;
    const LEN = 9;
    const REACH = 230;
    const EASE = 0.14;

    return bgHarness(cvs, (ctx, st) => {
        const noise = makeNoise(77);
        let cells = [];
        let t = 0;

        return {
            resize() {
                cells = [];
                // Capped at ~4000 segments. Density used to scale with the
                // screen with no ceiling: ~12k per frame at 4K, on the same
                // thread as the games' physics and inference.
                const gap = Math.max(GAP / Math.max(0.6, INTENSITY), Math.sqrt((st.W * st.H) / 4000));
                for (let y = gap / 2; y < st.H + gap; y += gap) {
                    for (let x = gap / 2; x < st.W + gap; x += gap) {
                        cells.push({ x, y, a: 0, cur: 0 });
                    }
                }
            },
            draw(now, paint) {
                t += 0.0009;
                const rgb = particleRgb();
                const maxA = 0.97 * INTENSITY;
                const buckets = bgBuckets();

                for (const c of cells) {
                    // Resting orientation: the noise field, turning slowly.
                    let target = noise(c.x * 0.0022, c.y * 0.0022 + t) * Math.PI * 2;
                    let strength = 0.22;

                    const dx = st.mx - c.x, dy = st.my - c.y;
                    const d = Math.sqrt(dx * dx + dy * dy);
                    if (d < REACH) {
                        const w = 1 - d / REACH;
                        // Point AT the pointer, weighted by proximity.
                        const toward = Math.atan2(dy, dx);
                        let diff = toward - target;
                        while (diff > Math.PI) diff -= Math.PI * 2;
                        while (diff < -Math.PI) diff += Math.PI * 2;
                        target += diff * w;
                        strength = 0.22 + w * w * 0.75;
                    }

                    let diff = target - c.cur;
                    while (diff > Math.PI) diff -= Math.PI * 2;
                    while (diff < -Math.PI) diff += Math.PI * 2;
                    c.cur += diff * EASE;
                    if (!paint) continue;

                    const half = LEN * (0.55 + strength * 0.8) / 2;
                    const cos = Math.cos(c.cur) * half, sin = Math.sin(c.cur) * half;

                    const k = bgLevel(strength * INTENSITY, maxA);
                    if (!k) continue;
                    const path = bgPath(buckets, k);
                    path.moveTo(c.x - cos, c.y - sin);
                    path.lineTo(c.x + cos, c.y + sin);
                }
                if (!paint) return;
                ctx.clearRect(0, 0, st.W, st.H);
                ctx.lineWidth = 1;
                bgFlush(ctx, buckets, rgb, maxA, true);
            },
        };
    }, o);
}

/* ── Sand ───────────────────────────────────────────────────────────────────
 * Dots resting in a loose grid. The cursor shoves them aside and they spring
 * back with damping, overshooting very slightly. The most tactile of the set —
 * it is the one that feels like touching the page. */
function initSand(cvs, opts) {
    const o = opts || {};
    const INTENSITY = o.intensity === undefined ? 1 : o.intensity;
    const GAP = 22;
    const PUSH = 165;
    const SPRING = 0.055;
    const DAMP = 0.86;

    return bgHarness(cvs, (ctx, st) => {
        let parts = [];

        return {
            resize() {
                parts = [];
                // Capped at ~5000 grains, for the same reason as Filings.
                const gap = Math.max(GAP / Math.max(0.6, INTENSITY), Math.sqrt((st.W * st.H) / 5000));
                for (let y = gap / 2; y < st.H + gap; y += gap) {
                    for (let x = gap / 2; x < st.W + gap; x += gap) {
                        const jx = (Math.random() - 0.5) * gap * 0.45;
                        const jy = (Math.random() - 0.5) * gap * 0.45;
                        parts.push({ hx: x + jx, hy: y + jy, x: x + jx, y: y + jy,
                                     vx: 0, vy: 0, r: 0.7 + Math.random() * 1.1 });
                    }
                }
            },
            draw(now, paint) {
                const rgb = particleRgb();
                const maxA = 0.85 * INTENSITY;
                const buckets = bgBuckets();

                for (const p of parts) {
                    const dx = p.x - st.mx, dy = p.y - st.my;
                    const d = Math.sqrt(dx * dx + dy * dy);
                    if (d < PUSH && d > 0.01) {
                        const f = (1 - d / PUSH) * (1 - d / PUSH) * 9;
                        p.vx += (dx / d) * f;
                        p.vy += (dy / d) * f;
                    }
                    p.vx += (p.hx - p.x) * SPRING;
                    p.vy += (p.hy - p.y) * SPRING;
                    p.vx *= DAMP; p.vy *= DAMP;
                    p.x += p.vx; p.y += p.vy;
                    if (!paint) continue;

                    // Displaced dots brighten, so the disturbance is visible
                    // as light as well as position.
                    const off = Math.min(1, Math.hypot(p.x - p.hx, p.y - p.hy) / 26);
                    const a = (0.2 + off * 0.65) * INTENSITY;
                    // A 2-5px square is indistinguishable from a circle at this
                    // size, and is far cheaper to add to a path than an arc.
                    const rr = p.r + off * 1.1;
                    const k = bgLevel(a, maxA);
                    if (k) bgPath(buckets, k).rect(p.x - rr, p.y - rr, rr * 2, rr * 2);
                }
                if (!paint) return;
                ctx.clearRect(0, 0, st.W, st.H);
                bgFlush(ctx, buckets, rgb, maxA, false);
            },
        };
    }, o);
}

/* ── Constellation ──────────────────────────────────────────────────────────
 * Drifting points joined by a line whenever two come close enough, with the
 * cursor dragging its neighbours along.
 *
 * This is the most familiar background on the web and was included on request
 * after being argued against. Two changes keep it from looking like the stock
 * plugin: the links are hairline and very faint, and the cursor ATTRACTS
 * rather than repels, so the mesh gathers instead of scattering. */
function initConstellation(cvs, opts) {
    const o = opts || {};
    const INTENSITY = o.intensity === undefined ? 1 : o.intensity;
    const LINK = 116;
    const PULL = 220;

    return bgHarness(cvs, (ctx, st) => {
        let parts = [];

        return {
            resize() {
                const n = Math.round(
                    Math.min(220, (st.W * st.H) / 11000) * INTENSITY);
                parts = Array.from({ length: n }, () => ({
                    x: Math.random() * st.W, y: Math.random() * st.H,
                    vx: (Math.random() - 0.5) * 0.32,
                    vy: (Math.random() - 0.5) * 0.32,
                }));
            },
            draw(now, paint) {
                const rgb = particleRgb();

                for (const p of parts) {
                    const dx = st.mx - p.x, dy = st.my - p.y;
                    const d = Math.sqrt(dx * dx + dy * dy);
                    if (d < PULL && d > 1) {
                        const f = (1 - d / PULL) * 0.045;
                        p.vx += (dx / d) * f;
                        p.vy += (dy / d) * f;
                    }
                    p.vx *= 0.995; p.vy *= 0.995;
                    p.x += p.vx; p.y += p.vy;
                    if (p.x < 0) p.x += st.W; else if (p.x > st.W) p.x -= st.W;
                    if (p.y < 0) p.y += st.H; else if (p.y > st.H) p.y -= st.H;
                }
                if (!paint) return;
                ctx.clearRect(0, 0, st.W, st.H);

                const maxL = 0.16 * INTENSITY;
                const links = bgBuckets();
                const dots = new Path2D();
                for (let i = 0; i < parts.length; i++) {
                    const a = parts[i];
                    for (let j = i + 1; j < parts.length; j++) {
                        const b = parts[j];
                        const dx = a.x - b.x, dy = a.y - b.y;
                        if (Math.abs(dx) > LINK || Math.abs(dy) > LINK) continue;
                        const d = Math.sqrt(dx * dx + dy * dy);
                        if (d > LINK) continue;
                        const k = bgLevel((1 - d / LINK) * maxL, maxL);
                        if (!k) continue;
                        const path = bgPath(links, k);
                        path.moveTo(a.x, a.y); path.lineTo(b.x, b.y);
                    }
                    dots.moveTo(a.x + 1.2, a.y);
                    dots.arc(a.x, a.y, 1.2, 0, Math.PI * 2);
                }
                ctx.lineWidth = 1;
                bgFlush(ctx, links, rgb, maxL, true);
                ctx.fillStyle = 'rgba(' + rgb + ',' + (0.45 * INTENSITY).toFixed(3) + ')';
                ctx.fill(dots);
            },
        };
    }, o);
}

/* ── Dispersion ─────────────────────────────────────────────────────────────
 * Particles hold the shape of the page's own heading, scatter when the cursor
 * passes through, and reassemble.
 *
 * The target positions are sampled from the heading rendered to an offscreen
 * canvas, so it adapts to whatever each page's title actually says instead of
 * being hardcoded. They outline the heading's own letters, in register with
 * them (see sampleTargets). A page with no visible heading — or a screen
 * narrower than NARROW — gets a drifting field instead of rendering nothing. */
function initDispersion(cvs, opts) {
    const o = opts || {};
    const INTENSITY = o.intensity === undefined ? 1 : o.intensity;
    const PUSH = 130;
    const SPRING = 0.045;
    const DAMP = 0.9;
    const GAP = 6;              // clearance kept from the text above and below
    const NARROW = 600;         // below this width, no ghost at all

    return bgHarness(cvs, (ctx, st) => {
        let parts = [];
        let el = null;          // the heading being traced, if any
        let ax = 0, ay = 0;     // where the ghost is anchored right now
        let half = 0;           // how far the halo reaches above/below its centre
        let fade = 1, cleared = false;

        /* A heading that is actually on the page. The game pages keep an h1
           for screen readers that is clipped to 1px, and tracing that drew a
           ghost of "Snake — Human vs AI" hanging off the top-left corner. */
        function findHeading() {
            const h = document.querySelector('.title, .header h1, h1');
            if (!h) return null;
            const r = h.getBoundingClientRect();
            return r.width > 2 && r.height > 2 ? h : null;
        }

        /* The TEXT's box, not the element's. On inside.html the h1 is a
           full-width block with left-aligned text, so centring on the element
           floated the ghost far to the right of the visible word. A Range over
           the contents measures the glyphs. */
        function textBox(h) {
            let box = null;
            try {
                const range = document.createRange();
                range.selectNodeContents(h);
                box = range.getBoundingClientRect();
            } catch (e) { /* fall through to the element box */ }
            return box && box.width ? box : h.getBoundingClientRect();
        }

        /* The nearest rendered element above (dir -1) or below (dir +1) the
           heading in document order: its sibling, or its ancestor's sibling.
           These are what the ghost must not print over — on select.html the
           "SELECT A GAME" eyebrow and the card grid. */
        function neighbour(h, dir) {
            for (let n = h; n && n !== document.body; n = n.parentElement) {
                let s = dir < 0 ? n.previousElementSibling : n.nextElementSibling;
                while (s) {
                    const r = s.getBoundingClientRect();
                    /* Out-of-flow boxes are not "the text above": on
                       index.html the previous sibling is this very canvas,
                       fixed over the whole viewport, and treating it as a
                       neighbour left no room at all, so the landing page
                       never got its ghost. */
                    const pos = getComputedStyle(s).position;
                    if (r.height > 0 && r.width > 0 && s.tagName !== 'CANVAS' &&
                        pos !== 'fixed' && pos !== 'absolute') return r;
                    s = dir < 0 ? s.previousElementSibling : s.nextElementSibling;
                }
            }
            return null;
        }

        /* Sample the ghost as offsets from the heading's centre, so it can
           follow the heading when the page scrolls instead of staying pinned
           to where the heading was at load.

           REGISTRATION. The ghost used to be a separate wordmark ~1.9x the
           heading's size, in a different font, centred on it — so the real
           title landed on different letters of the ghost ("human vs ai" over
           the ghost's "man vs"), which read as a misprint, and on inside.html
           it ran past the content margin. Now every glyph is drawn exactly
           where the browser drew it: each character's own box comes from a
           Range, and it is rendered in that text's computed font, size,
           weight, style and case. The particles then form a HALO — a ring
           RING px wide, starting INSET px outside each letter's edge — so the
           echo hugs the real letters instead of hiding under them (particles
           at the heading's own size and position would just vanish beneath
           the solid ink). A cursor still scatters them; they spring back to
           the outline.

           It stays clear of the text above and below (select.html's eyebrow
           and card grid): any halo point that would cross a neighbour is
           dropped rather than the whole ghost. */
        function glyphBoxes(h) {
            const out = [];
            const walker = document.createTreeWalker(h, NodeFilter.SHOW_TEXT);
            const range = document.createRange();
            for (let n = walker.nextNode(); n; n = walker.nextNode()) {
                const cs = getComputedStyle(n.parentElement);
                if (cs.visibility === 'hidden' || cs.display === 'none') continue;
                const font = [cs.fontStyle, cs.fontVariant === 'small-caps' ? 'small-caps' : '',
                              cs.fontWeight, cs.fontSize, cs.fontFamily].filter(Boolean).join(' ');
                const tt = cs.textTransform;
                const txt = n.textContent;
                for (let i = 0; i < txt.length; i++) {
                    let ch = txt[i];
                    if (/\s/.test(ch)) continue;
                    range.setStart(n, i); range.setEnd(n, i + 1);
                    const r = range.getBoundingClientRect();
                    if (!r.width || !r.height) continue;
                    if (tt === 'uppercase') ch = ch.toUpperCase();
                    else if (tt === 'lowercase') ch = ch.toLowerCase();
                    out.push({ ch, font, r });
                }
            }
            return out;
        }

        function sampleTargets() {
            el = st.W >= NARROW ? findHeading() : null;
            if (!el) return [];
            const glyphs = glyphBoxes(el);
            if (!glyphs.length) return [];

            const box = textBox(el);
            const cx = box.left + box.width / 2, cy = box.top + box.height / 2;
            const headSize = parseFloat(getComputedStyle(el).fontSize) || 40;
            const INSET = Math.max(2, Math.round(headSize * 0.05));
            const RING = Math.max(3, Math.round(headSize * 0.05));
            const PAD = INSET + RING + 2;

            // An offscreen canvas covering the glyphs' union box plus the halo.
            let L = Infinity, T = Infinity, R = -Infinity, B = -Infinity;
            for (const g of glyphs) {
                L = Math.min(L, g.r.left); T = Math.min(T, g.r.top);
                R = Math.max(R, g.r.right); B = Math.max(B, g.r.bottom);
            }
            const ox = Math.floor(L) - PAD, oy = Math.floor(T) - PAD;
            const w = Math.ceil(R) + PAD - ox, h = Math.ceil(B) + PAD - oy;
            if (!(w > 0 && h > 0) || w * h > 4e6) return [];
            const off = document.createElement('canvas');
            off.width = w; off.height = h;
            const g = off.getContext('2d');
            g.textBaseline = 'alphabetic';
            g.lineJoin = 'round';

            /* Each character at its own box. The baseline sits where the
               browser puts it inside an inline box: the font's ascent below
               the top of the content area, which is centred in the box. */
            const draw = (stroke) => {
                for (const q of glyphs) {
                    g.font = q.font;
                    const m = g.measureText(q.ch);
                    const asc = m.fontBoundingBoxAscent || parseFloat(q.font) * 0.8;
                    const desc = m.fontBoundingBoxDescent || parseFloat(q.font) * 0.2;
                    const x = q.r.left - ox;
                    const y = q.r.top - oy + (q.r.height - asc - desc) / 2 + asc;
                    if (stroke) g.strokeText(q.ch, x, y);
                    g.fillText(q.ch, x, y);
                }
            };
            // The letters grown by INSET + RING...
            g.fillStyle = g.strokeStyle = '#fff';
            g.lineWidth = 2 * (INSET + RING);
            draw(true);
            // ...minus the letters grown by INSET, leaves the ring.
            g.globalCompositeOperation = 'destination-out';
            g.lineWidth = 2 * INSET;
            draw(true);
            g.globalCompositeOperation = 'source-over';

            const above = neighbour(el, -1), below = neighbour(el, 1);
            const minY = above ? above.bottom + GAP : -Infinity;
            const maxY = below ? below.top - GAP : Infinity;
            const minX = 8, maxX = st.W - 8;

            const data = g.getImageData(0, 0, w, h).data;
            /* The sampling step must stay finer than the ring, or a thin
               ring aliases into a dotted rectangle (inside.html's low
               intensity used to stretch it to 6px over a 4px ring). Intensity
               already scales the particles' alpha. */
            const step = RING >= 6 ? 3 : 2;
            const pts = [];
            let top = Infinity, bottom = -Infinity;
            for (let y = 0; y < h; y += step) {
                for (let x = 0; x < w; x += step) {
                    if (data[(y * w + x) * 4 + 3] <= 128) continue;
                    const px = ox + x, py = oy + y;
                    if (py < minY || py > maxY || px < minX || px > maxX) continue;
                    pts.push({ x: px - cx, y: py - cy });
                    top = Math.min(top, py); bottom = Math.max(bottom, py);
                }
            }
            half = pts.length ? Math.max(cy - top, bottom - cy) : 0;
            ax = cx;
            ay = cy;
            return pts;
        }

        /* Follow the heading as the page scrolls. The particles are moved
           rigidly with it, not left to spring after it, so the ghost scrolls
           with the text it belongs to instead of smearing behind it. */
        function track() {
            if (!el) return;
            const box = textBox(el);
            const nx = box.left + box.width / 2, ny = box.top + box.height / 2;
            const dx = nx - ax, dy = ny - ay;
            if (!dx && !dy) return;
            ax = nx; ay = ny;
            for (const p of parts) { p.x += dx; p.y += dy; }
            if (st.wake) st.wake();
        }
        window.addEventListener('scroll', track, { passive: true });

        /* Is the ghost on screen? Once the heading scrolls away the ghost
           fades out, rather than dotting over whatever section text has
           scrolled under it, and fades back in when the heading returns. */
        function onScreen() {
            return !el || (ay + half > 0 && ay - half < st.H);
        }

        /* The glyphs are measured from the live layout, so a web font that
           arrives after the first sample would leave the halo tracing the
           fallback font's letters. Re-sample once fonts settle. */
        let gone = false;
        if (document.fonts && document.fonts.ready) {
            document.fonts.ready.then(() => { if (!gone && el) api.resize(); });
        }

        const api = {
            resize() {
                const pts = sampleTargets();
                if (!pts.length) {
                    el = null;
                    // No heading to trace — drift instead of showing nothing.
                    parts = Array.from({ length: 500 }, () => {
                        const x = Math.random() * st.W, y = Math.random() * st.H;
                        return { ox: 0, oy: 0, hx: x, hy: y, x, y, vx: 0, vy: 0 };
                    });
                    return;
                }
                parts = pts.map((t) => ({
                    ox: t.x, oy: t.y,
                    x: Math.random() * st.W, y: Math.random() * st.H,
                    vx: 0, vy: 0,
                }));
            },
            destroy() { gone = true; window.removeEventListener('scroll', track); },
            draw(now, paint) {
                // Also catches the heading moving WITHOUT a scroll — a web
                // font arriving, or content loading in above it.
                if (paint) track();
                const want = onScreen() ? 1 : 0;
                fade += Math.max(-0.08, Math.min(0.08, want - fade));
                if (fade <= 0) {
                    // Fully faded: nothing to integrate or draw until it returns.
                    if (paint && !cleared) { ctx.clearRect(0, 0, st.W, st.H); cleared = true; }
                    return;
                }
                cleared = false;

                const rgb = particleRgb();
                const maxA = 0.5 * INTENSITY;
                const buckets = bgBuckets();

                for (const p of parts) {
                    const hx = el ? ax + p.ox : p.hx, hy = el ? ay + p.oy : p.hy;
                    const dx = p.x - st.mx, dy = p.y - st.my;
                    const d = Math.sqrt(dx * dx + dy * dy);
                    if (d < PUSH && d > 0.01) {
                        const f = (1 - d / PUSH) * (1 - d / PUSH) * 13;
                        p.vx += (dx / d) * f;
                        p.vy += (dy / d) * f;
                    }
                    p.vx += (hx - p.x) * SPRING;
                    p.vy += (hy - p.y) * SPRING;
                    p.vx *= DAMP; p.vy *= DAMP;
                    p.x += p.vx; p.y += p.vy;
                    if (!paint) continue;

                    const off = Math.min(1, Math.hypot(p.x - hx, p.y - hy) / 40);
                    const a = (0.5 - off * 0.3) * INTENSITY * fade;
                    if (a < 0.02) continue;
                    const k = bgLevel(a, maxA);
                    if (k) bgPath(buckets, k).rect(p.x, p.y, 1.6, 1.6);
                }
                if (!paint) return;
                ctx.clearRect(0, 0, st.W, st.H);
                bgFlush(ctx, buckets, rgb, maxA, false);
            },
        };
        return api;
    }, o);
}

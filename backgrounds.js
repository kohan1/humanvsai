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

function particleRgb() {
    const s = getComputedStyle(document.documentElement);
    return (s.getPropertyValue('--particle-rgb').trim() || '235, 235, 235');
}
function themeBgRgb() {
    const s = getComputedStyle(document.documentElement);
    return (s.getPropertyValue('--bg-rgb').trim() || '8, 8, 8');
}

/* Boilerplate every renderer needs: sizing, pointer tracking, teardown. */
function bgHarness(cvs, setup) {
    const ctx = cvs.getContext('2d');
    const state = { W: 0, H: 0, mx: -9999, my: -9999, raf: 0, down: false };

    const onMove = (e) => { state.mx = e.clientX; state.my = e.clientY; };
    const onTouch = (e) => {
        if (!e.touches.length) return;
        state.mx = e.touches[0].clientX; state.my = e.touches[0].clientY;
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('touchmove', onTouch, { passive: true });

    const api = setup(ctx, state);

    function resize() {
        state.W = cvs.width = window.innerWidth;
        state.H = cvs.height = window.innerHeight;
        if (api.resize) api.resize();
    }
    window.addEventListener('resize', resize);
    resize();

    function frame(now) {
        api.draw(now);
        state.raf = requestAnimationFrame(frame);
    }
    state.raf = requestAnimationFrame(frame);

    return function () {
        cancelAnimationFrame(state.raf);
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
    const FADE = 0.055;        // trail persistence

    return bgHarness(cvs, (ctx, st) => {
        const noise = makeNoise(20260729);
        let parts = [];
        let t = 0;

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
            draw() {
                t += 0.0016;

                // Fade toward the theme background rather than clearing.
                ctx.globalCompositeOperation = 'source-over';
                ctx.fillStyle = 'rgba(' + themeBgRgb() + ',' + FADE + ')';
                ctx.fillRect(0, 0, st.W, st.H);

                const rgb = particleRgb();
                ctx.strokeStyle = 'rgba(' + rgb + ',' + (0.5 * INTENSITY).toFixed(3) + ')';
                ctx.lineWidth = 1;
                ctx.beginPath();

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
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(nx, ny);
                    p.x = nx; p.y = ny;

                    if (++p.life > 260 || p.x < -10 || p.x > st.W + 10 ||
                        p.y < -10 || p.y > st.H + 10) {
                        p.x = Math.random() * st.W;
                        p.y = Math.random() * st.H;
                        p.life = 0;
                    }
                }
                ctx.stroke();
            },
        };
    });
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
                const gap = GAP / Math.max(0.6, INTENSITY);
                for (let y = gap / 2; y < st.H + gap; y += gap) {
                    for (let x = gap / 2; x < st.W + gap; x += gap) {
                        cells.push({ x, y, a: 0, cur: 0 });
                    }
                }
            },
            draw() {
                t += 0.0009;
                ctx.clearRect(0, 0, st.W, st.H);
                const rgb = particleRgb();

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

                    const half = LEN * (0.55 + strength * 0.8) / 2;
                    const cos = Math.cos(c.cur) * half, sin = Math.sin(c.cur) * half;

                    ctx.strokeStyle = 'rgba(' + rgb + ',' + (strength * INTENSITY).toFixed(3) + ')';
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(c.x - cos, c.y - sin);
                    ctx.lineTo(c.x + cos, c.y + sin);
                    ctx.stroke();
                }
            },
        };
    });
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
                const gap = GAP / Math.max(0.6, INTENSITY);
                for (let y = gap / 2; y < st.H + gap; y += gap) {
                    for (let x = gap / 2; x < st.W + gap; x += gap) {
                        const jx = (Math.random() - 0.5) * gap * 0.45;
                        const jy = (Math.random() - 0.5) * gap * 0.45;
                        parts.push({ hx: x + jx, hy: y + jy, x: x + jx, y: y + jy,
                                     vx: 0, vy: 0, r: 0.7 + Math.random() * 1.1 });
                    }
                }
            },
            draw() {
                ctx.clearRect(0, 0, st.W, st.H);
                const rgb = particleRgb();

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

                    // Displaced dots brighten, so the disturbance is visible
                    // as light as well as position.
                    const off = Math.min(1, Math.hypot(p.x - p.hx, p.y - p.hy) / 26);
                    const a = (0.2 + off * 0.65) * INTENSITY;
                    ctx.fillStyle = 'rgba(' + rgb + ',' + a.toFixed(3) + ')';
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, p.r + off * 1.1, 0, Math.PI * 2);
                    ctx.fill();
                }
            },
        };
    });
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
            draw() {
                ctx.clearRect(0, 0, st.W, st.H);
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

                ctx.lineWidth = 1;
                for (let i = 0; i < parts.length; i++) {
                    const a = parts[i];
                    for (let j = i + 1; j < parts.length; j++) {
                        const b = parts[j];
                        const dx = a.x - b.x, dy = a.y - b.y;
                        if (Math.abs(dx) > LINK || Math.abs(dy) > LINK) continue;
                        const d = Math.sqrt(dx * dx + dy * dy);
                        if (d > LINK) continue;
                        const al = (1 - d / LINK) * 0.16 * INTENSITY;
                        ctx.strokeStyle = 'rgba(' + rgb + ',' + al.toFixed(3) + ')';
                        ctx.beginPath();
                        ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
                        ctx.stroke();
                    }
                    ctx.fillStyle = 'rgba(' + rgb + ',' + (0.45 * INTENSITY).toFixed(3) + ')';
                    ctx.beginPath();
                    ctx.arc(a.x, a.y, 1.2, 0, Math.PI * 2);
                    ctx.fill();
                }
            },
        };
    });
}

/* ── Dispersion ─────────────────────────────────────────────────────────────
 * Particles hold the shape of the page's own heading, scatter when the cursor
 * passes through, and reassemble.
 *
 * The target positions are sampled from the heading rendered to an offscreen
 * canvas, so it adapts to whatever each page's title actually says instead of
 * being hardcoded. A page with no heading falls back to a drifting field
 * rather than rendering nothing. */
function initDispersion(cvs, opts) {
    const o = opts || {};
    const INTENSITY = o.intensity === undefined ? 1 : o.intensity;
    const PUSH = 130;
    const SPRING = 0.045;
    const DAMP = 0.9;

    return bgHarness(cvs, (ctx, st) => {
        let parts = [];

        function headingText() {
            const el = document.querySelector('.title, .header h1, h1');
            const t = el ? el.textContent.trim().replace(/\s+/g, ' ') : '';
            return t.slice(0, 22) || 'human vs ai';
        }

        function sampleTargets() {
            const off = document.createElement('canvas');
            const w = off.width = Math.min(1100, st.W);
            const h = off.height = 260;
            const g = off.getContext('2d');
            const size = Math.min(150, w / (headingText().length * 0.52));
            g.fillStyle = '#fff';
            g.font = '600 ' + size + 'px "Helvetica Neue", Helvetica, Arial, sans-serif';
            g.textAlign = 'center';
            g.textBaseline = 'middle';
            g.fillText(headingText(), w / 2, h / 2);

            const data = g.getImageData(0, 0, w, h).data;
            const step = Math.max(2, Math.round(4 / Math.max(0.5, INTENSITY)));
            const pts = [];
            for (let y = 0; y < h; y += step) {
                for (let x = 0; x < w; x += step) {
                    if (data[(y * w + x) * 4 + 3] > 128) {
                        pts.push({ x: x + (st.W - w) / 2,
                                   y: y + (st.H - h) / 2 });
                    }
                }
            }
            return pts;
        }

        return {
            resize() {
                const pts = sampleTargets();
                if (!pts.length) {
                    // No heading to trace — drift instead of showing nothing.
                    parts = Array.from({ length: 500 }, () => {
                        const x = Math.random() * st.W, y = Math.random() * st.H;
                        return { hx: x, hy: y, x, y, vx: 0, vy: 0 };
                    });
                    return;
                }
                parts = pts.map((t) => ({
                    hx: t.x, hy: t.y,
                    x: Math.random() * st.W, y: Math.random() * st.H,
                    vx: 0, vy: 0,
                }));
            },
            draw() {
                ctx.clearRect(0, 0, st.W, st.H);
                const rgb = particleRgb();

                for (const p of parts) {
                    const dx = p.x - st.mx, dy = p.y - st.my;
                    const d = Math.sqrt(dx * dx + dy * dy);
                    if (d < PUSH && d > 0.01) {
                        const f = (1 - d / PUSH) * (1 - d / PUSH) * 13;
                        p.vx += (dx / d) * f;
                        p.vy += (dy / d) * f;
                    }
                    p.vx += (p.hx - p.x) * SPRING;
                    p.vy += (p.hy - p.y) * SPRING;
                    p.vx *= DAMP; p.vy *= DAMP;
                    p.x += p.vx; p.y += p.vy;

                    const off = Math.min(1, Math.hypot(p.x - p.hx, p.y - p.hy) / 40);
                    const a = (0.5 - off * 0.3) * INTENSITY;
                    if (a < 0.02) continue;
                    ctx.fillStyle = 'rgba(' + rgb + ',' + a.toFixed(3) + ')';
                    ctx.fillRect(p.x, p.y, 1.6, 1.6);
                }
            },
        };
    });
}

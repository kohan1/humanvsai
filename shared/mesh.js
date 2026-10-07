/* Animated triangular mesh background.
 *
 * Extracted from select.html so inside.html can use the same one rather than
 * carrying a second copy. Both pages call initMesh() with a canvas id.
 *
 * The look: a grid of points that drift, ripple on a slow wave, and push away
 * from the cursor. Triangles between them are drawn brighter where the wave
 * lifts them and where the cursor is, which reads as depth without any 3D.
 *
 * opts.intensity scales every alpha. select.html is a landing page and runs at
 * 1; inside.html sits behind dense charts and runs lower, so the data stays
 * legible.
 *
 * opts.maxFps caps the background's frame rate (motion keeps its speed).
 * The game pages should pass 30: there the background shares the main thread
 * with physics, rendering and WASM inference, and is behind the boards. While
 * a game is being played it drops further, to opts.busyFps — see bgPaceMs.
 */
/* Which background a theme draws. Each one reacts to the cursor, because that
 * is the whole point — the page should feel like a surface being disturbed,
 * not a picture with a mouse over it.
 *
 *   mesh      the original: a triangular lattice that flexes and pushes away
 *   reveal    used by both minimal themes: a hairline grid that is invisible
 *             until the pointer nears it, and fades again behind you. Nothing
 *             animates on its own — if the cursor never moves, nothing draws
 *
 * initMesh() dispatches on document.documentElement.dataset.theme, and
 * re-dispatches when that changes, so switching theme swaps the renderer
 * without a reload.
 */
/* How long to leave between background frames, in ms (0 = every frame).
 *
 * opts.maxFps is the cap a page asks for (the game pages pass 30). On top of
 * that, while someone is actually PLAYING the background drops to
 * opts.busyFps (default 15): the boards cover the middle of the screen, the
 * player is looking at them, and the background shares the main thread with
 * the game loop, the physics and the AI. "Playing" is detected rather than
 * reported, so no game has to call anything: a key or a press anywhere in the
 * last BUSY_MS, or a `playing` class on <html> if a page ever wants to say so
 * itself. Pages without maxFps (landing, picker, Inside) are never throttled.
 * Every renderer's motion is paced by the clock or integrated per frame, so
 * the frame rate changes how smooth the background is, never how fast. */
const BG_BUSY_MS = 8000;
let bgLastInput = -Infinity;
(function () {
    const note = () => { bgLastInput = performance.now(); };
    window.addEventListener('keydown', note, true);
    window.addEventListener('pointerdown', note, true);
})();
function bgPaceMs(o) {
    if (!o || !o.maxFps) return 0;
    const busy = performance.now() - bgLastInput < BG_BUSY_MS ||
                 document.documentElement.classList.contains('playing');
    const fps = busy ? Math.min(o.maxFps, o.busyFps || 15) : o.maxFps;
    return 1000 / fps - 2;
}

function initMesh(canvasId, opts) {
    const cvs = document.getElementById(canvasId);
    if (!cvs) return;

    const themeOf = () => document.documentElement.getAttribute('data-theme') || 'mesh';
    let stop = null;
    let mounted = null;

    function mount() {
        const t = themeOf();
        // setAttribute with an unchanged value still produces a mutation
        // record, and every one used to tear the background down and reseed
        // it — a visible restart for a call that changed nothing.
        if (stop && t === mounted) return;
        if (stop) { stop(); stop = null; }
        mounted = t;
        // Renderers live in backgrounds.js; the lattice stays here as the
        // default and as the fallback if that file fails to load.
        const R = {
            flow:          typeof initFlow          === 'function' && initFlow,
            filings:       typeof initFilings       === 'function' && initFilings,
            sand:          typeof initSand          === 'function' && initSand,
            constellation: typeof initConstellation === 'function' && initConstellation,
            dispersion:    typeof initDispersion    === 'function' && initDispersion,
        };
        stop = (R[t] || initLattice)(cvs, opts);
    }

    new MutationObserver(mount).observe(document.documentElement,
        { attributes: true, attributeFilter: ['data-theme'] });
    mount();
}

/* ── Reveal ─────────────────────────────────────────────────────────────────
 * The interaction for both minimal themes: a hairline grid that is invisible
 * until the pointer is near it, and fades out again behind you.
 *
 * Deliberately the least it can do and still respond. The previous attempts
 * added particles, glow and grain, which is decoration — the opposite of what
 * minimalism is. Here the page looks completely empty until you move, and what
 * appears is the same square grid the rest of the site is built on rather than
 * an unrelated effect.
 *
 * One colour, one shape, no gradients, nothing animating on its own. If the
 * cursor never moves, nothing ever draws. */
function initReveal(cvs, opts) {
    const ctx = cvs.getContext('2d');
    const o = opts || {};
    const INTENSITY = o.intensity === undefined ? 1 : o.intensity;
    const CELL = 44;
    const R = 190;          // how far the reveal reaches
    const EASE = 0.12;      // how closely the reveal follows the pointer

    let W, H, raf;
    let mx = -9999, my = -9999, cx = -9999, cy = -9999;

    const onMove = (e) => { mx = e.clientX; my = e.clientY; };
    const onTouch = (e) => { mx = e.touches[0].clientX; my = e.touches[0].clientY; };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('touchmove', onTouch, { passive: true });

    function resize() {
        // The canvas's OWN box, not window.innerWidth. innerWidth includes the
        // scrollbar but the fixed, inset:0 canvas does not span it — measured
        // 1280 against a 1265px box on inside.html — so sizing the bitmap from
        // innerWidth made the browser rescale the whole background by 0.988.
        W = cvs.width = cvs.clientWidth || window.innerWidth;
        H = cvs.height = cvs.clientHeight || window.innerHeight;
    }
    window.addEventListener('resize', resize);
    resize();

    // Read once: a theme change remounts this renderer, so a per-frame style
    // lookup was only ever returning the same value.
    const INK = getComputedStyle(document.documentElement).getPropertyValue('--grid-ink').trim() || '17, 17, 17';
    function inkRgb() { return INK; }

    function draw() {
        cx += (mx - cx) * EASE;
        cy += (my - cy) * EASE;

        ctx.clearRect(0, 0, W, H);
        const rgb = inkRgb();

        // Only the segments inside the reveal radius are drawn at all, so the
        // cost is bounded by the radius rather than the viewport.
        const x0 = Math.max(0, Math.floor((cx - R) / CELL) * CELL);
        const x1 = Math.min(W, cx + R);
        const y0 = Math.max(0, Math.floor((cy - R) / CELL) * CELL);
        const y1 = Math.min(H, cy + R);

        ctx.lineWidth = 1;

        for (let x = x0; x <= x1; x += CELL) {
            for (let y = y0; y <= y1; y += CELL) {
                const dx = x - cx, dy = y - cy;
                const d = Math.sqrt(dx * dx + dy * dy);
                if (d > R) continue;
                const a = (1 - d / R) * (1 - d / R) * 0.5 * INTENSITY;
                if (a < 0.008) continue;

                ctx.strokeStyle = 'rgba(' + rgb + ',' + a.toFixed(3) + ')';
                ctx.beginPath();
                ctx.moveTo(Math.round(x) + 0.5, Math.round(y) + 0.5);
                ctx.lineTo(Math.round(Math.min(x + CELL, x1)) + 0.5, Math.round(y) + 0.5);
                ctx.moveTo(Math.round(x) + 0.5, Math.round(y) + 0.5);
                ctx.lineTo(Math.round(x) + 0.5, Math.round(Math.min(y + CELL, y1)) + 0.5);
                ctx.stroke();
            }
        }

        // A single dot exactly on the pointer. The only ornament in either
        // theme, and it is 3px wide.
        if (mx > -9000) {
            ctx.fillStyle = 'rgba(' + rgb + ',' + (0.55 * INTENSITY).toFixed(3) + ')';
            ctx.beginPath();
            ctx.arc(cx, cy, 1.6, 0, Math.PI * 2);
            ctx.fill();
        }

        raf = requestAnimationFrame(draw);
    }
    raf = requestAnimationFrame(draw);

    return function () {
        cancelAnimationFrame(raf);
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('touchmove', onTouch);
        // Was `onResize`, which this renderer never defines: teardown threw a
        // ReferenceError and left the resize listener attached.
        window.removeEventListener('resize', resize);
        ctx.clearRect(0, 0, cvs.width, cvs.height);
    };
}

/* CSS-PIXEL BACKING STORE, deliberately, even on a 2x screen — where the
 * lattice's hairlines therefore come out a little soft. Rendering at device
 * resolution was measured on 2026-10-07 (headless Chromium, DPR 2, 1440x900,
 * software raster): main-thread time on index.html went from ~0.5 s to ~1.0 s
 * per second and the frame rate fell from 60 to 38-46; flow was the same
 * shape. Path building is resolution-independent, so all of that is fill and
 * raster, which is 4x the pixels. A GPU-composited canvas pays far less, but
 * the machines that fall back to software raster are the same ones where the
 * games already struggle, and this canvas sits behind them. Softness is the
 * cheaper failure. */
function initLattice(cvs, opts) {
    const ctx = cvs.getContext('2d');

    const o = opts || {};
    const COLS = o.cols || 22;
    const ROWS = o.rows || 14;
    const INTENSITY = o.intensity === undefined ? 1 : o.intensity;

    const CURSOR_RADIUS = 160;
    const REPEL_FORCE   = 0.32;
    const DRIFT_AMP     = 12;
    const RETURN_SPEED  = 0.055;
    const WAVE_AMP      = 18;

    /* OVERSCAN — rings of cells built OUTSIDE the viewport on every side.
     *
     * Every point moves: drift (up to ~26px), the wave (18px) and the cursor
     * push (REPEL_FORCE * CURSOR_RADIUS, ~51px). Without overscan the outermost
     * row and column are the edge of the mesh, so any of that motion pulling
     * them inward opens a bare strip along the border — most visible when the
     * cursor is near an edge and shoves the boundary points away.
     *
     * Two rings covers the worst case (~95px) at any sensible cell size. The
     * cost is the extra triangles: 26x18 instead of 22x14. */
    const OVER = 2;
    const gridCols = COLS + OVER * 2;
    const gridRows = ROWS + OVER * 2;
    const stride = gridCols + 1;

    let W, H, pts, raf;
    let driftT = 0, waveT = 0;
    let mouseX = -9999, mouseY = -9999;

    // Named rather than inline, so switching theme can remove them — an
    // anonymous handler cannot be detached and would keep firing forever.
    /* prefers-reduced-motion: the lattice drifts and waves on its own, so it
     * settles for a moment and then holds still; moving the pointer wakes it
     * briefly, since that motion is the visitor's own doing. */
    const reduce = !!(window.matchMedia &&
                      window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    let budget = reduce ? 90 : Infinity;
    const wake = () => {
        if (!reduce) return;
        budget = Math.max(budget, 45);
        if (!raf) raf = requestAnimationFrame(draw);
    };
    const onMove = (e) => { mouseX = e.clientX; mouseY = e.clientY; wake(); };
    const onTouch = (e) => {
        if (!e.touches.length) return;
        mouseX = e.touches[0].clientX; mouseY = e.touches[0].clientY; wake();
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('touchmove', onTouch, { passive: true });

    function resize() {
        // See the note in the other resize(): the bitmap must match the canvas
        // box, or the background is rescaled by the scrollbar's width.
        W = cvs.width  = cvs.clientWidth || window.innerWidth;
        H = cvs.height = cvs.clientHeight || window.innerHeight;
        build();
    }

    function build() {
        pts = [];
        // Cell size still comes from the VISIBLE area, so the mesh looks the
        // same density as before; the overscan rings simply extend past it.
        const cw = W / COLS, ch = H / ROWS;
        for (let r = 0; r <= gridRows; r++) {
            for (let c = 0; c <= gridCols; c++) {
                const x = (c - OVER) * cw, y = (r - OVER) * ch;
                pts.push({
                    bx: x, by: y,
                    cx: x, cy: y,
                    ox: (Math.random()-0.5) * DRIFT_AMP * 2.2,
                    oy: (Math.random()-0.5) * DRIFT_AMP * 2.2,
                    f:  0.28 + Math.random() * 0.55,
                    ph: Math.random() * Math.PI * 2,
                });
            }
        }
    }

    function waveDisplacement(bx, by) {
        const nx = bx / W, ny = by / H;
        const w1 = Math.sin(nx * 4.5 + ny * 2.2 + waveT);
        const w2 = Math.sin(nx * 2.6 - ny * 3.8 + waveT * 0.62 + 1.8);
        const w3 = Math.sin(nx * 1.4 + ny * 1.4 - waveT * 0.4 + 3.1);
        return (w1 * 0.45 + w2 * 0.35 + w3 * 0.2) * WAVE_AMP;
    }

    /* k is how many 60 Hz frames' worth of motion to advance. Paced by the
     * clock rather than by calls, so capping the frame rate (or a 120 Hz
     * screen) changes how often the mesh is stepped, not how fast it moves. */
    function update(k) {
        driftT += 0.00028 * k;
        waveT  += 0.0062 * k;
        const ease = 1 - Math.pow(1 - RETURN_SPEED, k);
        for (let i = 0; i < pts.length; i++) {
            const p = pts[i];
            const wob = waveDisplacement(p.bx, p.by);
            const tx = p.bx + p.ox * Math.sin(driftT * p.f + p.ph);
            const ty = p.by + p.oy * Math.cos(driftT * p.f + p.ph + 1.3) + wob;
            const dx = p.cx - mouseX, dy = p.cy - mouseY;
            const dist = Math.sqrt(dx*dx + dy*dy);
            let rx = 0, ry = 0;
            if (dist < CURSOR_RADIUS && dist > 0.1) {
                const s = (1 - dist / CURSOR_RADIUS) * REPEL_FORCE * CURSOR_RADIUS;
                rx = (dx / dist) * s; ry = (dy / dist) * s;
            }
            p.cx += (tx + rx - p.cx) * ease;
            p.cy += (ty + ry - p.cy) * ease;
            p.wob = wob;
        }
    }

    /* BATCHED DRAWING.
     *
     * This used to stroke every triangle separately — 936 beginPath/stroke
     * calls a frame, each with its own lineWidth and a freshly formatted
     * rgba() string, plus one arc/fill per node. Profiled on the game pages it
     * was the single largest per-frame cost, ~15x the Tetris game loop, and it
     * churned ~20 MB of short-lived strings through the heap.
     *
     * Now it draws EDGES, not triangles, and groups them into a few dozen
     * buckets by quantised depth and cursor glow, so a frame is one stroke
     * per bucket instead of one per triangle. Two details keep it looking the
     * same:
     *   - Every visible edge is shared by two triangles, so it used to be
     *     painted twice. One pass at 1-(1-a)^2 composites to exactly what two
     *     passes at alpha a did.
     *   - An edge's depth and glow are taken from its own endpoints and
     *     midpoint, which sit between the two triangles it used to inherit
     *     from — so the shading is the average of what was drawn before.
     * Quantisation steps are ~0.014 alpha in depth and ~0.05 near the cursor,
     * below what a hairline on a dark page can show. */
    const DQ = 8, CQ = 6;                       // depth / glow levels
    const NB = (DQ + 1) * (CQ + 1);
    const styles = new Array(NB);               // bucket -> [strokeStyle, lineWidth]
    for (let qd = 0; qd <= DQ; qd++) {
        for (let qc = 0; qc <= CQ; qc++) {
            const depth = qd / DQ, cg = qc / CQ;
            const r = Math.round(200 + depth * 55);
            const g = Math.round(210 + depth * 45);
            const a1 = Math.min((0.08 + cg * 0.28 + depth * 0.11) * INTENSITY, 0.68 * INTENSITY);
            const a2 = 1 - (1 - a1) * (1 - a1);
            styles[qd * (CQ + 1) + qc] = [
                'rgba(' + r + ',' + g + ',255,' + a2.toFixed(3) + ')',
                0.3 + cg * 0.55 + depth * 0.32,
            ];
        }
    }
    let paths = new Array(NB);

    function edge(i, j) {
        const p = pts[i], q = pts[j];
        const mx = (p.cx + q.cx) * 0.5, my = (p.cy + q.cy) * 0.5;
        const dx = mx - mouseX, dy = my - mouseY;
        const cg = Math.max(0, 1 - Math.sqrt(dx * dx + dy * dy) / CURSOR_RADIUS);
        const depth = ((p.wob + q.wob) * 0.5 / WAVE_AMP + 1) * 0.5;
        const qd = Math.max(0, Math.min(DQ, Math.round(depth * DQ)));
        const k = qd * (CQ + 1) + Math.round(cg * CQ);
        const path = paths[k] || (paths[k] = new Path2D());
        path.moveTo(p.cx, p.cy);
        path.lineTo(q.cx, q.cy);
    }

    /* Nodes, batched the same way: one fill per alpha level. */
    const NQ = 12;
    let nodePaths = new Array(NQ + 1);
    const NODE_MAX = 0.75 * INTENSITY;

    function drawNodes() {
        for (let i = 0; i < pts.length; i++) {
            const p = pts[i];
            const depth = (p.wob / WAVE_AMP + 1) / 2;
            if (depth < 0.72) continue;
            const dx = p.cx - mouseX, dy = p.cy - mouseY;
            const cg = Math.max(0, 1 - Math.sqrt(dx * dx + dy * dy) / CURSOR_RADIUS);
            const size = 1 + depth * 1.3 + cg * 1.1;
            const alpha = Math.min(((depth - 0.72) * 2.4 + cg * 0.28) * INTENSITY, NODE_MAX);
            const k = Math.round(alpha / NODE_MAX * NQ);
            if (k <= 0) continue;
            const path = nodePaths[k] || (nodePaths[k] = new Path2D());
            path.moveTo(p.cx + size, p.cy);
            path.arc(p.cx, p.cy, size, 0, Math.PI * 2);
        }
        for (let k = 1; k <= NQ; k++) {
            if (!nodePaths[k]) continue;
            ctx.fillStyle = 'rgba(220,232,255,' + (k / NQ * NODE_MAX).toFixed(3) + ')';
            ctx.fill(nodePaths[k]);
        }
        nodePaths = new Array(NQ + 1);
    }

    /* opts.maxFps caps the frame rate — motion AND paint. The motion is slow
     * drift, so 30 fps is indistinguishable from 60 behind a game, and halves
     * the cost on the thread the game's physics and inference share. Unset
     * means every frame. A skipped frame now does nothing at all: update()
     * used to run on every frame regardless, because it advanced a fixed
     * amount per call, so the capped pages still paid for the wave on
     * frames they never drew. It is time-scaled now (see update), so the
     * speed is unchanged. bgPaceMs() also slows it further while a game is
     * being played. */
    let lastDraw = -Infinity;

    function draw(now) {
        const minDt = bgPaceMs(o);
        if (minDt && now - lastDraw < minDt) {
            /* A skipped frame costs nothing from the reduced-motion budget:
               that budget counts frames the visitor SEES, and charging the
               skipped ones too halved it on every page that sets maxFps. */
            raf = budget > 0 ? requestAnimationFrame(draw) : 0;
            return;
        }
        // Clamped, so a frame after a hidden tab or a long stall resumes
        // the drift instead of jumping it.
        update(lastDraw < 0 ? 1 : Math.min(4, Math.max(0, (now - lastDraw) / (1000 / 60))));
        lastDraw = now;
        ctx.clearRect(0, 0, W, H);
        for (let r = 0; r < gridRows; r++) {
            for (let c = 0; c < gridCols; c++) {
                const a = r * stride + c;
                edge(a, a + 1);                          // top
                edge(a, a + stride);                     // left
                edge(a + 1, a + stride);                 // diagonal
                if (c === gridCols - 1) edge(a + 1, a + stride + 1);          // right
                if (r === gridRows - 1) edge(a + stride, a + stride + 1);     // bottom
            }
        }
        for (let k = 0; k < NB; k++) {
            if (!paths[k]) continue;
            ctx.strokeStyle = styles[k][0];
            ctx.lineWidth = styles[k][1];
            ctx.stroke(paths[k]);
        }
        paths = new Array(NB);

        drawNodes();

        if (mouseX > 0) {
            const g = ctx.createRadialGradient(mouseX, mouseY, 0, mouseX, mouseY, CURSOR_RADIUS);
            g.addColorStop(0,   `rgba(210,225,255,${0.045 * INTENSITY})`);
            g.addColorStop(0.5, `rgba(210,225,255,${0.015 * INTENSITY})`);
            g.addColorStop(1,   'rgba(210,225,255,0)');
            ctx.fillStyle = g;
            ctx.beginPath();
            ctx.arc(mouseX, mouseY, CURSOR_RADIUS, 0, Math.PI*2);
            ctx.fill();
        }

        raf = --budget > 0 ? requestAnimationFrame(draw) : 0;
    }

    const onResize = () => { resize(); wake(); };
    window.addEventListener('resize', onResize);
    resize();
    raf = requestAnimationFrame(draw);

    return () => {
        cancelAnimationFrame(raf);
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('touchmove', onTouch);
        window.removeEventListener('resize', onResize);
        ctx.clearRect(0, 0, cvs.width, cvs.height);
    };
}

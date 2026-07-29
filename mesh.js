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
 */
/* Which background a theme draws. Each one reacts to the cursor, because that
 * is the whole point — the page should feel like a surface being disturbed,
 * not a picture with a mouse over it.
 *
 *   mesh      the original: a triangular lattice that flexes and pushes away
 *   embers    sparks rising through the dark, pulled into a vortex by the
 *             pointer — tangential force, so they orbit rather than scatter
 *   lens      a hard grid with an inverting disc around the pointer, drawn
 *             with difference blending so it punches through rather than glows
 *
 * initMesh() dispatches on document.documentElement.dataset.theme, and
 * re-dispatches when that changes, so switching theme swaps the renderer
 * without a reload.
 */
function initMesh(canvasId, opts) {
    const cvs = document.getElementById(canvasId);
    if (!cvs) return;

    const themeOf = () => document.documentElement.getAttribute('data-theme') || 'mesh';
    let stop = null;

    function mount() {
        if (stop) { stop(); stop = null; }
        const t = themeOf();
        stop = t === 'ember'  ? initEmbers(cvs, opts)
             : t === 'halide' ? initLens(cvs, opts)
             :                  initLattice(cvs, opts);
    }

    new MutationObserver(mount).observe(document.documentElement,
        { attributes: true, attributeFilter: ['data-theme'] });
    mount();
}

/* ── Embers ─────────────────────────────────────────────────────────────────
 * Sparks rising through the dark, dragged into a vortex around the pointer.
 *
 * The cursor does not merely push them away — it applies a TANGENTIAL force,
 * so they orbit rather than scatter. That is the difference between "particles
 * avoiding the mouse", which everyone has seen, and something that reads as
 * heat actually moving. */
function initEmbers(cvs, opts) {
    const ctx = cvs.getContext('2d');
    const o = opts || {};
    const INTENSITY = o.intensity === undefined ? 1 : o.intensity;
    const COUNT = Math.round(150 * (o.intensity === undefined ? 1 : o.intensity));
    const SWIRL = 190;

    let W, H, raf, parts = [];
    let mx = -9999, my = -9999;
    let last = performance.now();

    const onMove = (e) => { mx = e.clientX; my = e.clientY; };
    const onTouch = (e) => { mx = e.touches[0].clientX; my = e.touches[0].clientY; };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('touchmove', onTouch, { passive: true });

    function spawn(anywhere) {
        return {
            x: Math.random() * W,
            y: anywhere ? Math.random() * H : H + Math.random() * 60,
            vx: (Math.random() - 0.5) * 6,
            vy: -(8 + Math.random() * 26),
            r: 0.6 + Math.random() * 1.9,
            life: Math.random(),
            hot: Math.random(),          // how far toward cream this one burns
        };
    }

    function resize() {
        W = cvs.width = window.innerWidth;
        H = cvs.height = window.innerHeight;
        parts = Array.from({ length: COUNT }, () => spawn(true));
    }
    window.addEventListener('resize', resize);
    resize();

    function hexToRgb(h) {
        const v = h.replace('#', '').trim();
        const full = v.length === 3 ? v[0] + v[0] + v[1] + v[1] + v[2] + v[2] : v;
        const n = parseInt(full, 16);
        return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }

    function palette() {
        const s = getComputedStyle(document.documentElement);
        const a = s.getPropertyValue('--accent').trim() || '#c5283d';
        const b = s.getPropertyValue('--accent-2').trim() || '#f5e6c8';
        return [hexToRgb(a), hexToRgb(b)];
    }

    function draw(now) {
        const dt = Math.min(0.05, (now - last) / 1000);
        last = now;
        ctx.clearRect(0, 0, W, H);

        const pal = palette();
        const a = pal[0], b = pal[1];

        for (let i = 0; i < parts.length; i++) {
            const p = parts[i];
            const dx = p.x - mx, dy = p.y - my;
            const d = Math.sqrt(dx * dx + dy * dy);

            if (d < SWIRL && d > 0.5) {
                const f = (1 - d / SWIRL) * (1 - d / SWIRL);
                // Tangential first — this is what makes them orbit rather
                // than simply flee.
                p.vx += (-dy / d) * f * 260 * dt;
                p.vy += (dx / d) * f * 260 * dt;
                p.vx += (dx / d) * f * 60 * dt;
                p.vy += (dy / d) * f * 60 * dt;
            }

            p.vy -= 9 * dt;                    // buoyancy
            p.vx *= 0.985;
            p.vy *= 0.985;
            p.x += p.vx * dt * 6;
            p.y += p.vy * dt * 6;
            p.life += dt * 0.22;

            if (p.y < -40 || p.life > 1.6 || p.x < -60 || p.x > W + 60) {
                const fresh = spawn(false);
                p.x = fresh.x; p.y = fresh.y; p.vx = fresh.vx; p.vy = fresh.vy;
                p.r = fresh.r; p.life = 0; p.hot = fresh.hot;
            }

            const fade = Math.max(0, 1 - Math.abs(p.life - 0.5) * 1.4);
            const mix = p.hot;
            const cr = Math.round(a[0] + (b[0] - a[0]) * mix);
            const cg = Math.round(a[1] + (b[1] - a[1]) * mix);
            const cb = Math.round(a[2] + (b[2] - a[2]) * mix);
            const alpha = fade * (0.35 + p.hot * 0.5) * INTENSITY;
            if (alpha < 0.01) continue;

            const rad = p.r * 5;
            const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, rad);
            g.addColorStop(0, 'rgba(' + cr + ',' + cg + ',' + cb + ',' + alpha.toFixed(3) + ')');
            g.addColorStop(1, 'rgba(' + cr + ',' + cg + ',' + cb + ',0)');
            ctx.fillStyle = g;
            ctx.beginPath();
            ctx.arc(p.x, p.y, rad, 0, Math.PI * 2);
            ctx.fill();
        }
        raf = requestAnimationFrame(draw);
    }
    raf = requestAnimationFrame(draw);

    return function () {
        cancelAnimationFrame(raf);
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('touchmove', onTouch);
        window.removeEventListener('resize', resize);
        ctx.clearRect(0, 0, cvs.width, cvs.height);
    };
}

/* ── Lens ───────────────────────────────────────────────────────────────────
 * A hard engineering grid, and a disc around the pointer where everything
 * INVERTS. No glow, no falloff — a clean edge, composited with difference
 * blending, so it reads as a hole punched through the page rather than a light
 * shone on it. Deliberately the opposite of Embers. */
function initLens(cvs, opts) {
    const ctx = cvs.getContext('2d');
    const o = opts || {};
    const INTENSITY = o.intensity === undefined ? 1 : o.intensity;
    const CELL = 46;
    const R = 120;

    let W, H, raf;
    let mx = -9999, my = -9999, cx = -9999, cy = -9999;
    const t0 = performance.now();

    const onMove = (e) => { mx = e.clientX; my = e.clientY; };
    const onTouch = (e) => { mx = e.touches[0].clientX; my = e.touches[0].clientY; };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('touchmove', onTouch, { passive: true });

    function resize() {
        W = cvs.width = window.innerWidth;
        H = cvs.height = window.innerHeight;
    }
    window.addEventListener('resize', resize);
    resize();

    function ink() {
        const s = getComputedStyle(document.documentElement);
        return s.getPropertyValue('--fg').trim() || '#0d0d0c';
    }

    function grid(alpha, phase) {
        ctx.globalAlpha = alpha;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let x = phase % CELL; x < W; x += CELL) {
            ctx.moveTo(Math.round(x) + 0.5, 0);
            ctx.lineTo(Math.round(x) + 0.5, H);
        }
        for (let y = phase % CELL; y < H; y += CELL) {
            ctx.moveTo(0, Math.round(y) + 0.5);
            ctx.lineTo(W, Math.round(y) + 0.5);
        }
        ctx.stroke();
    }

    function draw(now) {
        const time = (now - t0) / 1000;
        // Ease toward the pointer so the disc trails very slightly behind it.
        cx += (mx - cx) * 0.16;
        cy += (my - cy) * 0.16;

        ctx.clearRect(0, 0, W, H);
        ctx.strokeStyle = ink();

        grid(0.10 * INTENSITY, time * 6);

        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, cy, R, 0, Math.PI * 2);
        ctx.clip();
        ctx.globalCompositeOperation = 'difference';
        ctx.globalAlpha = 1;
        ctx.fillStyle = ink();
        ctx.fillRect(cx - R, cy - R, R * 2, R * 2);
        ctx.strokeStyle = ink();
        grid(1, time * 6);
        for (let r = 18; r < R; r += 18) {
            ctx.globalAlpha = 0.9;
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0, Math.PI * 2);
            ctx.stroke();
        }
        ctx.restore();

        ctx.globalAlpha = 0.55 * INTENSITY;
        ctx.lineWidth = 1;
        ctx.strokeStyle = ink();
        ctx.beginPath();
        ctx.arc(cx, cy, R, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 1;

        raf = requestAnimationFrame(draw);
    }
    raf = requestAnimationFrame(draw);

    return function () {
        cancelAnimationFrame(raf);
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('touchmove', onTouch);
        window.removeEventListener('resize', resize);
        ctx.clearRect(0, 0, cvs.width, cvs.height);
    };
}

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
    const onMove = (e) => { mouseX = e.clientX; mouseY = e.clientY; };
    const onTouch = (e) => { mouseX = e.touches[0].clientX; mouseY = e.touches[0].clientY; };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('touchmove', onTouch, { passive: true });

    function resize() {
        W = cvs.width  = window.innerWidth;
        H = cvs.height = window.innerHeight;
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

    function update() {
        driftT += 0.00028;
        waveT  += 0.0062;
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
            p.cx += (tx + rx - p.cx) * RETURN_SPEED;
            p.cy += (ty + ry - p.cy) * RETURN_SPEED;
            p.wob = wob;
        }
    }

    function drawTri(i1, i2, i3) {
        const mx = (pts[i1].cx + pts[i2].cx + pts[i3].cx) / 3;
        const my = (pts[i1].cy + pts[i2].cy + pts[i3].cy) / 3;
        const md = Math.sqrt((mx - mouseX)**2 + (my - mouseY)**2);
        const cg = Math.max(0, 1 - md / CURSOR_RADIUS);

        const avgWob = (pts[i1].wob + pts[i2].wob + pts[i3].wob) / 3;
        const depth = (avgWob / WAVE_AMP + 1) / 2;

        const r = Math.round(200 + depth * 55);
        const g = Math.round(210 + depth * 45);
        const b = 255;

        const alpha = (0.08 + cg * 0.28 + depth * 0.11) * INTENSITY;
        const lw    = 0.3 + cg * 0.55 + depth * 0.32;

        ctx.strokeStyle = `rgba(${r},${g},${b},${Math.min(alpha, 0.68 * INTENSITY)})`;
        ctx.lineWidth   = lw;
        ctx.beginPath();
        ctx.moveTo(pts[i1].cx, pts[i1].cy);
        ctx.lineTo(pts[i2].cx, pts[i2].cy);
        ctx.lineTo(pts[i3].cx, pts[i3].cy);
        ctx.closePath();
        ctx.stroke();
    }

    function drawNodes() {
        for (let i = 0; i < pts.length; i++) {
            const p = pts[i];
            const depth = (p.wob / WAVE_AMP + 1) / 2;
            if (depth < 0.72) continue;
            const md = Math.sqrt((p.cx - mouseX)**2 + (p.cy - mouseY)**2);
            const cg = Math.max(0, 1 - md / CURSOR_RADIUS);
            const size = 1 + depth * 1.3 + cg * 1.1;
            const alpha = ((depth - 0.72) * 2.4 + cg * 0.28) * INTENSITY;
            ctx.beginPath();
            ctx.arc(p.cx, p.cy, size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(220,232,255,${Math.min(alpha, 0.75 * INTENSITY)})`;
            ctx.fill();
        }
    }

    function draw() {
        ctx.clearRect(0, 0, W, H);
        update();
        for (let r = 0; r < gridRows; r++) {
            for (let c = 0; c < gridCols; c++) {
                const a = r * stride + c;
                drawTri(a, a+1, a+stride);
                drawTri(a+1, a+stride+1, a+stride);
            }
        }

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

        raf = requestAnimationFrame(draw);
    }

    window.addEventListener('resize', resize);
    resize();
    raf = requestAnimationFrame(draw);

    return () => {
        cancelAnimationFrame(raf);
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('touchmove', onTouch);
        window.removeEventListener('resize', resize);
        ctx.clearRect(0, 0, cvs.width, cvs.height);
    };
}

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
function initMesh(canvasId, opts) {
    const cvs = document.getElementById(canvasId);
    if (!cvs) return;
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

    let W, H, pts;
    let driftT = 0, waveT = 0;
    let mouseX = -9999, mouseY = -9999;

    window.addEventListener('mousemove', e => { mouseX = e.clientX; mouseY = e.clientY; });
    window.addEventListener('touchmove', e => {
        mouseX = e.touches[0].clientX; mouseY = e.touches[0].clientY;
    }, { passive: true });

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

        requestAnimationFrame(draw);
    }

    window.addEventListener('resize', resize);
    resize();
    draw();
}

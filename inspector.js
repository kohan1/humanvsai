/* The neural-network inspector, shared by all three games.
 *
 * Shows what the model actually receives and what it produces:
 *
 *   - the observation, one heat map per channel. This is the real tensor the
 *     network is fed, read straight back out of the same encoder that feeds
 *     onnxruntime — not a re-drawing of the board. It is deliberately coarse,
 *     because that is genuinely all the network sees.
 *   - the action distribution, softmaxed from the logits the model returns.
 *   - confidence, as normalised entropy over that distribution.
 *   - the critic's value estimate, if the loaded model exports one.
 *
 * IT LIVES BELOW THE FOLD, always present, reached by the scroll cue pinned to
 * the bottom of the first screen. There is no toggle button.
 *
 * COST WHEN OFF SCREEN IS ZERO. An IntersectionObserver tracks whether the
 * panel is actually in view, and update() returns on its first line when it is
 * not — so playing without ever scrolling costs nothing. That matters: these
 * games already run two boards and a WASM inference loop. The same signal
 * triggers the one-off critic download, so a visitor who never scrolls never
 * fetches it.
 *
 * THEMING. Snake and Tetris are near-black, Watermelon is cream and sage, so
 * every colour comes from a CSS custom property with a dark default. Each
 * game's stylesheet overrides the handful it needs.
 */
function createInspector(config) {
    const {
        mount,
        grid,           // { w, h, channels: [{ label, hint }] }
        readCell,       // (obs, row, col, channel) -> number, 0..1
        actions,        // { count, orientation: 'columns' | 'rows', label(i) }
        scalars,        // optional: (obs) -> [{ label, value }]
        valueLabel = 'position value',
        valueHint = '',
        onReveal,       // called once, the first time the panel comes into view
    } = config;

    let open = false;
    let revealed = false;
    let lastPayload = null;

    const root = document.createElement('div');
    root.className = 'insp';

    const chanWrap = document.createElement('div');
    chanWrap.className = 'insp-channels';
    const canvases = grid.channels.map(ch => {
        const cell = document.createElement('div');
        cell.className = 'insp-chan';
        const cvs = document.createElement('canvas');
        cvs.className = 'insp-chan-cvs';
        const cap = document.createElement('div');
        cap.className = 'insp-chan-label';
        cap.textContent = ch.label;
        if (ch.hint) cap.title = ch.hint;
        cell.append(cvs, cap);
        chanWrap.append(cell);
        return cvs;
    });

    const readouts = document.createElement('div');
    readouts.className = 'insp-readouts';
    readouts.innerHTML = `
        <div class="insp-metric">
            <div class="insp-metric-label">confidence</div>
            <div class="insp-bar"><span class="insp-bar-fill" data-conf></span></div>
            <div class="insp-metric-num" data-conf-num>—</div>
        </div>
        <div class="insp-metric" data-value-metric hidden>
            <div class="insp-metric-label">${valueLabel}</div>
            <div class="insp-metric-num insp-metric-big" data-value>—</div>
            <div class="insp-metric-hint">${valueHint}</div>
        </div>`;

    const actWrap = document.createElement('div');
    actWrap.className = 'insp-actions insp-actions-' + actions.orientation;
    const actTitle = document.createElement('div');
    actTitle.className = 'insp-section-label';
    actTitle.textContent = 'what it wants to do';
    const actBars = document.createElement('div');
    actBars.className = 'insp-bars';
    const bars = [];
    for (let i = 0; i < actions.count; i++) {
        const b = document.createElement('div');
        b.className = 'insp-act';
        const fill = document.createElement('span');
        fill.className = 'insp-act-fill';
        const lab = document.createElement('span');
        lab.className = 'insp-act-label';
        lab.textContent = actions.label ? actions.label(i) : String(i);
        b.append(fill, lab);
        b.title = `action ${i}`;
        actBars.append(b);
        bars.push({ el: b, fill });
    }
    actWrap.append(actTitle, actBars);

    const scalarWrap = document.createElement('div');
    scalarWrap.className = 'insp-scalars';

    const seesTitle = document.createElement('div');
    seesTitle.className = 'insp-section-label';
    seesTitle.textContent = `what it sees — ${grid.w}x${grid.h} grid, ${grid.channels.length} channels`;

    root.append(seesTitle, chanWrap, scalarWrap, actWrap, readouts);
    mount.append(root);

    /* ── Drawing ────────────────────────────────────────────────────────── */

    function paintChannels(obs) {
        const CELL = 6;
        canvases.forEach((cvs, ch) => {
            const dpr = window.devicePixelRatio || 1;
            const w = grid.w * CELL, h = grid.h * CELL;
            if (cvs.width !== w * dpr) {
                cvs.width = w * dpr; cvs.height = h * dpr;
                cvs.style.width = w + 'px'; cvs.style.height = h + 'px';
            }
            const ctx = cvs.getContext('2d');
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

            const style = getComputedStyle(root);
            ctx.fillStyle = style.getPropertyValue('--insp-cell-bg').trim() || '#111';
            ctx.fillRect(0, 0, w, h);
            const rgb = (style.getPropertyValue('--insp-cell-rgb').trim() || '150,190,255');

            for (let r = 0; r < grid.h; r++) {
                for (let c = 0; c < grid.w; c++) {
                    const v = readCell(obs, r, c, ch);
                    if (!v) continue;
                    ctx.fillStyle = `rgba(${rgb},${Math.min(1, Math.max(0.12, v))})`;
                    ctx.fillRect(c * CELL, r * CELL, CELL - 0.5, CELL - 0.5);
                }
            }
        });
    }

    function softmax(logits) {
        let max = -Infinity;
        for (const v of logits) if (v > max) max = v;
        const exp = new Array(logits.length);
        let sum = 0;
        for (let i = 0; i < logits.length; i++) { exp[i] = Math.exp(logits[i] - max); sum += exp[i]; }
        for (let i = 0; i < exp.length; i++) exp[i] /= sum;
        return exp;
    }

    function paintActions(probs) {
        let pMax = 0, best = 0;
        probs.forEach((p, i) => { if (p > pMax) { pMax = p; best = i; } });
        bars.forEach((b, i) => {
            const p = probs[i] || 0;
            // Scale to the strongest action, or a nearly-uniform distribution
            // would render as 24 invisible slivers.
            b.fill.style.setProperty('--p', (pMax > 0 ? p / pMax : 0).toFixed(4));
            b.el.classList.toggle('is-best', i === best);
            b.el.title = `action ${i} — ${(p * 100).toFixed(1)}%`;
        });
    }

    function paintConfidence(probs) {
        // Normalised entropy: 0 when the policy is certain, 1 when uniform.
        let H = 0;
        for (const p of probs) if (p > 0) H -= p * Math.log(p);
        const conf = probs.length > 1 ? 1 - H / Math.log(probs.length) : 1;
        readouts.querySelector('[data-conf]').style.setProperty('--p', conf.toFixed(4));
        readouts.querySelector('[data-conf-num]').textContent = (conf * 100).toFixed(0) + '%';
    }

    function paintValue(value) {
        const metric = readouts.querySelector('[data-value-metric]');
        if (value === undefined || value === null || !isFinite(value)) {
            metric.hidden = true;      // model predates the value head
            return;
        }
        metric.hidden = false;
        readouts.querySelector('[data-value]').textContent = value.toFixed(1);
    }

    function paintScalars(obs) {
        if (!scalars) return;
        scalarWrap.innerHTML = scalars(obs).map(s =>
            `<span class="insp-scalar"><b>${s.label}</b>${s.value}</span>`).join('');
    }

    function render() {
        if (!open || !lastPayload) return;
        const { obs, logits, value } = lastPayload;
        if (obs) { paintChannels(obs); paintScalars(obs); }
        if (logits && logits.length) {
            const probs = softmax(logits);
            paintActions(probs);
            paintConfidence(probs);
        }
        paintValue(value);
    }

    /* ── Scroll cue ─────────────────────────────────────────────────────────
       Pinned to the bottom of the viewport rather than placed in the flow,
       because the three games have different page heights and anything in the
       flow would land in the wrong place on at least one of them. It fades
       out as soon as the reader starts scrolling — it has done its job. */
    const cue = document.createElement('button');
    cue.type = 'button';
    cue.className = 'insp-cue';
    cue.innerHTML =
        '<span class="insp-cue-text">scroll down to see inside</span>' +
        '<svg class="insp-cue-arrow" viewBox="0 0 24 14" aria-hidden="true">' +
        '<path d="M1 1 L12 12 L23 1" fill="none" stroke="currentColor" ' +
        'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    cue.addEventListener('click', () => {
        root.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    document.body.append(cue);

    const onScroll = () => {
        cue.classList.toggle('is-gone', window.scrollY > 60);
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();

    /* Only paint while the panel is actually on screen. rootMargin gives it a
       screen of warning so it is already populated by the time it scrolls in,
       rather than appearing blank for one AI decision. */
    if ('IntersectionObserver' in window) {
        new IntersectionObserver((entries) => {
            open = entries[0].isIntersecting;
            if (open && !revealed) {
                revealed = true;
                if (onReveal) onReveal();
            }
            render();
        }, { rootMargin: '400px 0px' }).observe(root);
    } else {
        open = true;                       // no observer: just keep it live
        if (onReveal) onReveal();
    }

    return {
        /* Called on every AI decision. Returns immediately while off screen. */
        update(payload) {
            if (!open) return;
            lastPayload = payload;
            render();
        },
        get isOpen() { return open; },
    };
}

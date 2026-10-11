/* onnxruntime in a Web Worker — the other half of the session proxy in
 * shared/model-source.js. Read the note there first; this file only does what
 * it is told.
 *
 * Messages in (all carry `id`, answered by a message with the same id):
 *   init     { ortUrl, wasmPaths, wasmBinary? }   load ort, warm its runtime
 *   create   { model: ArrayBuffer | url, options }  -> { inputNames, outputNames }
 *   run      { sid, feeds: {name: {type, data, dims}} } -> { out: same shape }
 *   release  { sid }
 * Every answer is { id, ok: true, ... } or { id, ok: false, error }. A failed
 * create hands the model buffer back, so the page can fall back to running it
 * on the main thread without downloading it again.
 */
'use strict';

const sessions = new Map();
let nextSid = 1;

/* The same 67-byte Identity model model-source.js used to warm the runtime on
 * the main thread: creating any session is what makes ort fetch, compile and
 * instantiate its WebAssembly, and that start-up should overlap the model
 * download rather than follow it. */
const WARMUP_MODEL = 'CAgSADo3ChAKAXgSAXkiCElkZW50aXR5EgF3Wg8KAXgSCgoICAESBAoCCAFiDwoBeRIKCggIARIECgIIAUIECgAQEQ==';

function b64bytes(s) {
    const bin = atob(s);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
}

function errText(e) {
    return (e && (e.message || String(e))) || 'unknown error';
}

async function handle(msg) {
    switch (msg.type) {
    case 'init': {
        importScripts(msg.ortUrl);
        // No threads: the page is not cross-origin isolated, so ort would
        // fall back to one anyway, and asking explicitly skips its probe.
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.wasmPaths = msg.wasmPaths;
        // The page fetched the .wasm (consuming its <link rel=preload>) and
        // transferred it here, so it is not downloaded a second time.
        if (msg.wasmBinary) ort.env.wasm.wasmBinary = msg.wasmBinary;
        const s = await ort.InferenceSession.create(b64bytes(WARMUP_MODEL),
                                                    { executionProviders: ['wasm'] });
        try { await s.release(); } catch (e) { /* harmless */ }
        return {};
    }
    case 'create': {
        const model = typeof msg.model === 'string' ? msg.model : new Uint8Array(msg.model);
        const s = await ort.InferenceSession.create(model, msg.options || {});
        const sid = nextSid++;
        sessions.set(sid, s);
        return { sid, inputNames: s.inputNames, outputNames: s.outputNames };
    }
    case 'run': {
        const s = sessions.get(msg.sid);
        if (!s) throw new Error('session ' + msg.sid + ' was released');
        const feeds = {};
        for (const k of Object.keys(msg.feeds)) {
            const f = msg.feeds[k];
            feeds[k] = new ort.Tensor(f.type, f.data, f.dims);
        }
        const t0 = performance.now();
        const res = await s.run(feeds);
        const took = performance.now() - t0;
        const out = {}, transfer = [];
        for (const k of Object.keys(res)) {
            const t = res[k];
            out[k] = { type: t.type, data: t.data, dims: t.dims };
            if (t.data && t.data.buffer instanceof ArrayBuffer) transfer.push(t.data.buffer);
        }
        return { out, transfer, took };
    }
    case 'release': {
        const s = sessions.get(msg.sid);
        sessions.delete(msg.sid);
        if (s) await s.release();
        return {};
    }
    }
    throw new Error('unknown message ' + msg.type);
}

self.onmessage = (e) => {
    const msg = e.data;
    handle(msg).then((r) => {
        const transfer = r.transfer || [];
        delete r.transfer;
        r.id = msg.id; r.ok = true;
        // A buffer that cannot be transferred (a view of wasm memory) throws
        // before anything is sent; copying it instead is always possible.
        try { self.postMessage(r, transfer); } catch (err) { self.postMessage(r); }
    }, (err) => {
        const reply = { id: msg.id, ok: false, error: errText(err) };
        // Hand an undelivered model back (see the header).
        if (msg.type === 'create' && msg.model instanceof ArrayBuffer) {
            reply.model = msg.model;
            self.postMessage(reply, [msg.model]);
        } else {
            self.postMessage(reply);
        }
    });
};

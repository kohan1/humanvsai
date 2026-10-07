/* Choose where a game's ONNX model comes from.
 *
 * Each game ships the model twice: as a plain .onnx, and as base64 inside
 * model_data.js. The second exists only for opening the site from disk, where
 * fetch() cannot read a sibling file — under file:// the origin is opaque and
 * the request is rejected.
 *
 * The intent was always to fetch the .onnx over HTTP and keep the base64 for
 * file://. What actually happened is that game.html loaded model_data.js from
 * a static <script> tag, so the global was ALWAYS defined and the base64
 * branch always won. Every visitor was paying, before the page could settle:
 *
 *     snake       46 MB script, then a 34 MB decode
 *     watermelon  31 MB script, then a 23 MB decode
 *     tetris      10 MB script, then a  7 MB decode
 *
 * — as a render-blocking download, followed by atob() plus a byte-at-a-time
 * copy on the main thread. The .onnx is a third smaller (base64 costs 33%),
 * streams, is cached by the browser as a normal resource, and never blocks
 * parsing.
 *
 * So model_data.js is no longer in the markup at all. It is injected here,
 * on demand, only when there is no server to fetch from.
 */
(function () {
    "use strict";

    /* model_data.js declares `const SNAKE_MODEL_B64 = "..."` at the top level
       of a classic script. A top-level const is a GLOBAL LEXICAL binding, not a
       property of window — `window.SNAKE_MODEL_B64` is undefined while the bare
       identifier resolves fine. Reading it through window[] silently found
       nothing and reported the file as broken after loading it successfully.

       A Function body evaluates in global scope, so it can see those lexical
       bindings. The name is checked against an identifier pattern first, since
       it is being pasted into source. */
    function readGlobal(name) {
        if (!/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(name)) return undefined;
        try {
            return new Function(
                "return typeof " + name + " !== 'undefined' ? " + name + " : undefined;")();
        } catch (e) {
            return undefined;
        }
    }

    function decodeSync(b64) {
        var bin = atob(b64);
        var out = new Uint8Array(bin.length);
        for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
        return out;
    }

    /* Uint8Array.fromBase64 is native and measured 87 ms for Snake's 46 MB
       string, against ~320 ms for atob() plus the byte loop above — one
       main-thread task either way, during which the human board freezes.
       (fetch() on a data: URL looks like it should decode off the main thread
       and does the opposite: building and parsing the URL blocked for 1.5-1.8 s.)
       The loop stays for browsers without fromBase64. Returns a promise so the
       call sites need not care which ran. */
    function decode(b64) {
        try {
            if (typeof Uint8Array.fromBase64 === "function") {
                return Promise.resolve(Uint8Array.fromBase64(b64));
            }
            return Promise.resolve(decodeSync(b64));
        } catch (e) {
            return Promise.reject(e);
        }
    }

    function injectEmbedded(globalName, dataUrl) {
        var already = readGlobal(globalName);
        if (already !== undefined) return decode(already);
        return new Promise(function (resolve, reject) {
            var s = document.createElement("script");
            s.src = dataUrl || "model_data.js";
            s.onload = function () {
                var b64 = readGlobal(globalName);
                if (b64 === undefined) {
                    reject(new Error(s.src + " loaded but " + globalName + " is undefined"));
                    return;
                }
                resolve(decode(b64));   // a promise: resolve() adopts it
            };
            s.onerror = function () { reject(new Error("could not load " + s.src)); };
            document.head.appendChild(s);
        });
    }

    /* Start onnxruntime's own WebAssembly runtime while the model downloads.

       ort only fetches, compiles and instantiates its ~14 MB .wasm on the
       first InferenceSession.create(), and the games could only call that
       once the model had fully arrived, so the two ran back to back. Measured
       on Snake: the first create() took ~640 ms after the model landed and a
       second create() of the same model ~125 ms — the difference is runtime
       start-up sitting on the critical path.

       Creating a session for a 67-byte model (one Identity node) triggers
       exactly that start-up and nothing else. ort shares one init promise per
       backend, so the game's real create() simply joins it if it is still
       running. Every game sets ort.env.wasm.wasmPaths BEFORE calling
       modelSource(), which is what makes this safe to do from here: the
       runtime is fetched from the same place it would have been anyway.
       Failure is ignored — the real create() will report the real error. */
    var WARMUP_MODEL = "CAgSADo3ChAKAXgSAXkiCElkZW50aXR5EgF3Wg8KAXgSCgoICAESBAoCCAFiDwoBeRIKCggIARIECgIIAUIECgAQEQ==";
    var warming = null;
    function warmRuntime() {
        if (warming || typeof ort === "undefined" || !ort.InferenceSession) return;
        try {
            warming = ort.InferenceSession
                .create(decodeSync(WARMUP_MODEL), { executionProviders: ["wasm"] })
                .then(function (s) { return s.release && s.release(); })
                .catch(function () { /* the real create() reports it */ });
        } catch (e) { /* likewise */ }
    }

    /* ── Inference off the main thread ──────────────────────────────────
       onnxruntime's WebAssembly runs synchronously on whichever thread owns
       it. On the main thread that is the same thread as the game loop, the
       physics and the background, so every AI decision was a long task:
       measured on Snake at 4x CPU throttle, 22 ms median and 71-86 ms worst
       per run(), long enough to drop frames and miss decisions.

       So ort.InferenceSession.create() is replaced, once, by a version that
       builds the session inside a Worker (shared/ort-worker.js) and returns
       a stand-in with the same surface the games use — run(feeds) resolving
       to { name: ort.Tensor }, inputNames, outputNames, release(). No game
       code changes: the games, the checkpoint switcher and the critic
       loaders all keep calling ort.InferenceSession.create().

       Why not ort.env.wasm.proxy, which is ort's own version of this: it
       threw DataCloneError when tried here, and it cannot be told to reuse
       the .wasm the page has already preloaded.

       FALLBACK. Anything that goes wrong on the worker side — no Worker,
       file:// (a worker cannot be started from an opaque origin), the
       runtime failing to load in the worker, a create() failing there —
       falls back to the original main-thread create(), which is exactly how
       the games ran before. A failed create() hands the model buffer back,
       so the fallback does not download it again. */
    var WORKER_URL = (function () {
        try {
            var s = document.currentScript;
            if (!s || !s.src) return null;
            // Carry the deploy's ?v=<commit> cache-buster across, so a new
            // page never pairs with a stale worker.
            var u = new URL("ort-worker.js", s.src);
            u.search = new URL(s.src).search;
            return u.href;
        } catch (e) { return null; }
    }());
    var worker = null;     // null: not tried; false: unavailable; else the handle

    function startWorker() {
        if (worker !== null) return worker;
        worker = false;
        // window.ORT_MAIN_THREAD = true (set before the game loads) opts out,
        // for comparing the two paths.
        if (location.protocol === "file:" || typeof Worker === "undefined" || !WORKER_URL ||
            window.ORT_MAIN_THREAD ||
            typeof ort === "undefined" || !ort.env || !ort.env.wasm) return worker;
        var paths = ort.env.wasm.wasmPaths;
        var tag = document.querySelector('script[src*="onnxruntime-web@"]');
        if (typeof paths !== "string" || !tag) return worker;

        var w;
        try { w = new Worker(WORKER_URL, { name: "humanvsai-ort" }); }
        catch (e) { return worker; }

        var pending = {}, seq = 0, dead = null;
        function fail(reason) {
            if (dead) return;
            dead = reason;
            for (var id in pending) pending[id].reject({ error: reason });
            pending = {};
            try { w.terminate(); } catch (e) { /* gone */ }
        }
        w.onmessage = function (e) {
            var m = e.data, p = pending[m.id];
            if (!p) return;
            delete pending[m.id];
            if (m.ok) p.resolve(m); else p.reject(m);
        };
        w.onerror = function (e) {
            if (e && e.preventDefault) e.preventDefault();
            fail("worker error: " + ((e && e.message) || "could not start"));
        };
        function call(msg, transfer) {
            return new Promise(function (resolve, reject) {
                if (dead) { reject({ error: dead }); return; }
                msg.id = ++seq;
                pending[msg.id] = { resolve: resolve, reject: reject };
                try { w.postMessage(msg, transfer || []); }
                catch (e) { delete pending[msg.id]; reject({ error: String(e) }); }
            });
        }

        /* The page's <link rel=preload> already started the .wasm. Fetching
           it here (same URL, same CORS mode) takes over that download, and
           the buffer is then transferred to the worker rather than fetched a
           second time from inside it. */
        var ready = fetch(paths + "ort-wasm-simd-threaded.wasm")
            .then(function (r) { return r.ok ? r.arrayBuffer() : null; })
            .catch(function () { return null; })
            .then(function (wasm) {
                return call({ type: "init", ortUrl: tag.src, wasmPaths: paths, wasmBinary: wasm },
                            wasm ? [wasm] : []);
            })
            .then(function () { window.modelSource.backend = "worker"; return true; },
                  function (m) {
                      console.warn("model-source: inference worker unavailable (" +
                                   (m && m.error) + ") - running the AI on the main thread");
                      fail("unavailable");
                      window.modelSource.backend = "main";
                      warmRuntime();
                      return false;
                  });
        worker = { call: call, ready: ready };
        return worker;
    }

    function WorkerSession(info) {
        this.sid = info.sid;
        this.inputNames = info.inputNames;
        this.outputNames = info.outputNames;
    }
    WorkerSession.prototype.run = function (feeds) {
        var msg = {};
        for (var k in feeds) {
            var t = feeds[k], d = t.data;
            // A view onto a bigger buffer would clone the whole buffer.
            if (d && d.buffer && d.byteLength !== d.buffer.byteLength) d = d.slice();
            msg[k] = { type: t.type, data: d, dims: t.dims };
        }
        return worker.call({ type: "run", sid: this.sid, feeds: msg }).then(function (r) {
            var out = {};
            for (var name in r.out) {
                var o = r.out[name];
                out[name] = new ort.Tensor(o.type, o.data, o.dims);
            }
            return out;
        }, function (m) { throw new Error((m && m.error) || "inference failed"); });
    };
    WorkerSession.prototype.release = function () {
        return worker.call({ type: "release", sid: this.sid })
            .then(function () {}, function () {});
    };

    var patched = false;
    function patchOrt() {
        if (patched || typeof ort === "undefined" || !ort.InferenceSession) return;
        patched = true;
        var IS = ort.InferenceSession, original = IS.create;
        IS.create = function (model, options) {
            var args = arguments, self = this;
            var main = function () { return original.apply(self, args); };
            var isBytes = model instanceof ArrayBuffer || model instanceof Uint8Array;
            if (!worker || args.length > 2 || (typeof model !== "string" && !isBytes)) {
                return main();
            }
            return worker.ready.then(function (ok) {
                if (!ok) return main();
                var payload, transfer = [];
                if (typeof model === "string") {
                    payload = new URL(model, location.href).href;
                } else {
                    var u8 = model instanceof ArrayBuffer ? new Uint8Array(model) : model;
                    payload = u8.byteOffset === 0 && u8.byteLength === u8.buffer.byteLength &&
                              u8.buffer instanceof ArrayBuffer ? u8.buffer : u8.slice().buffer;
                    transfer = [payload];
                }
                return worker.call({ type: "create", model: payload, options: options || {} }, transfer)
                    .then(function (info) { return new WorkerSession(info); }, function (m) {
                        // A URL the worker could not load (the critic, when it
                        // is not deployed) would fail the same way here, and
                        // retrying would start a second runtime just to see it.
                        if (typeof model === "string") throw new Error((m && m.error) || "create failed");
                        console.warn("model-source: worker could not create the session (" +
                                     (m && m.error) + ") - trying on the main thread");
                        var back = m && m.model ? new Uint8Array(m.model)
                                 : (payload.byteLength ? new Uint8Array(payload) : null);
                        if (!back) throw new Error((m && m.error) || "worker failed");
                        return original.call(self, back, options);
                    });
            });
        };
    }

    /* Read a response body while reporting progress. A bare arrayBuffer()
       says nothing for the whole 22-34 MB download, which reads as "stuck on
       loading". Content-Length can be the COMPRESSED size (or absent), so
       callers get the raw byte count and a total that may be 0. */
    function readBody(r, onProgress) {
        if (!onProgress || !r.body || !r.body.getReader) {
            return r.arrayBuffer().then(function (b) { return new Uint8Array(b); });
        }
        var total = parseInt(r.headers.get("Content-Length") || "0", 10) || 0;
        var reader = r.body.getReader();
        var chunks = [], got = 0;
        function pump() {
            return reader.read().then(function (step) {
                if (step.done) {
                    var out = new Uint8Array(got), at = 0;
                    for (var i = 0; i < chunks.length; i++) {
                        out.set(chunks[i], at);
                        at += chunks[i].length;
                    }
                    return out;
                }
                chunks.push(step.value);
                got += step.value.length;
                try { onProgress(got, total); } catch (e) { /* UI only */ }
                return pump();
            });
        }
        return pump();
    }

    /* ── The visitor's saved model version ──────────────────────────────
       The checkpoint switcher remembers which rung a visitor chose. It used
       to start every page on the shipped model and only then switch, so a
       returning visitor who had picked an earlier version downloaded TWO
       models — the shipped one, preloaded, then theirs: 68 MB on Snake, and
       ~30 s of playing the wrong opponent on a slow link. So the choice is
       honoured here, before anything is fetched: the game asks for its
       shipped file and gets the saved rung's bytes instead, the head of
       game.html preloads that same file, and the switcher starts on it.

       The manifest (checkpoints.js) is the authority on which rungs exist; a
       saved id it no longer lists is ignored. */
    var CHOICE_KEY = "humanvsai.checkpoint";
    var FILE_KEY = "humanvsai.checkpointFile";   // read by the preload in game.html
    var outcome = {};                             // game -> { wanted, got }

    function readJson(key) {
        try { return JSON.parse(localStorage.getItem(key) || "{}") || {}; }
        catch (e) { return {}; }
    }

    function savedRung(onnxUrl) {
        var all = window.CHECKPOINTS;
        if (!all) return null;
        for (var game in all) {
            var rungs = (all[game] && all[game].rungs) || [];
            var top = rungs.filter(function (r) { return r.shipped; })[0] || rungs[rungs.length - 1];
            if (!top || top.file !== onnxUrl) continue;
            var id = readJson(CHOICE_KEY)[game];
            var r = rungs.filter(function (x) { return x.id === id; })[0];
            // Keep the preload's hint in step, including for a choice saved
            // before the file was recorded alongside it.
            try {
                var files = readJson(FILE_KEY);
                var want = r ? r.file : top.file;
                if (files[game] !== want) {
                    files[game] = want;
                    localStorage.setItem(FILE_KEY, JSON.stringify(files));
                }
            } catch (e) { /* private browsing */ }
            return { game: game, top: top, rung: r && !r.shipped ? r : null };
        }
        return null;
    }

    function fetchModel(url, onProgress) {
        return fetch(url).then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status + " for " + url);
            return readBody(r, onProgress);
        });
    }

    /* Returns the model bytes. Fetches the .onnx unless we are on file://,
       where fetch() cannot read a sibling file.

       Falls back to the embedded base64 if the fetch fails, which is not
       hypothetical: Tetris keeps its .onnx under training/ and only the deploy
       script copies it up beside the page, so tetris/tetris_ai.onnx is a 404
       on a local server and a 200 in production. Without the fallback this
       would work live and break in dev — the worst way round, since dev is
       where it would go unnoticed.

       onProgress(receivedBytes, totalBytesOr0) is optional and may be passed
       in place of dataUrl. It is called only for the .onnx download. */
    window.modelSource = function (onnxUrl, globalName, dataUrl, onProgress) {
        if (typeof dataUrl === "function") { onProgress = dataUrl; dataUrl = undefined; }
        patchOrt();
        if (!startWorker()) { window.modelSource.backend = "main"; warmRuntime(); }
        if (location.protocol === "file:") {
            return injectEmbedded(globalName, dataUrl);
        }
        var saved = savedRung(onnxUrl);
        var shipped = function () {
            return fetchModel(onnxUrl, onProgress).catch(function (err) {
                console.warn("model-source: " + err.message + " - falling back to embedded base64");
                return injectEmbedded(globalName, dataUrl).catch(function (err2) {
                    // Report both: on the live site model_data.js is not
                    // deployed, so the second error alone hides the cause.
                    throw new Error(err.message + "; " + err2.message);
                });
            });
        };
        if (!saved || !saved.rung) {
            if (saved) outcome[saved.game] = { wanted: saved.top.id, got: saved.top.id };
            return shipped();
        }
        var game = saved.game, rung = saved.rung;
        outcome[game] = { wanted: rung.id, got: null };
        return fetchModel(rung.file, onProgress).then(function (bytes) {
            outcome[game].got = rung.id;
            return bytes;
        }, function (err) {
            // The shipped model is always the safe answer.
            console.warn("model-source: saved version " + rung.id + " failed (" +
                         err.message + ") - loading the shipped model");
            outcome[game].got = saved.top.id;
            return shipped();
        });
    };

    /* Which rung modelSource() actually delivered for `game`, and which the
       visitor had asked for — for the checkpoint switcher to start on. Null
       when modelSource() has not run for that game (or ran under file://). */
    window.modelSource.outcome = function (game) { return outcome[game] || null; };
}());

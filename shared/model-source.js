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
        warmRuntime();
        if (location.protocol === "file:") {
            return injectEmbedded(globalName, dataUrl);
        }
        return fetch(onnxUrl)
            .then(function (r) {
                if (!r.ok) throw new Error("HTTP " + r.status + " for " + onnxUrl);
                return readBody(r, onProgress);
            })
            .catch(function (err) {
                console.warn("model-source: " + err.message + " - falling back to embedded base64");
                return injectEmbedded(globalName, dataUrl).catch(function (err2) {
                    // Report both: on the live site model_data.js is not
                    // deployed, so the second error alone hides the cause.
                    throw new Error(err.message + "; " + err2.message);
                });
            });
    };
}());

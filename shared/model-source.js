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

    function decode(b64) {
        var bin = atob(b64);
        var out = new Uint8Array(bin.length);
        for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
        return out;
    }

    function injectEmbedded(globalName, dataUrl) {
        var already = readGlobal(globalName);
        if (already !== undefined) return Promise.resolve(decode(already));
        return new Promise(function (resolve, reject) {
            var s = document.createElement("script");
            s.src = dataUrl || "model_data.js";
            s.onload = function () {
                var b64 = readGlobal(globalName);
                if (b64 === undefined) {
                    reject(new Error(s.src + " loaded but " + globalName + " is undefined"));
                    return;
                }
                resolve(decode(b64));
            };
            s.onerror = function () { reject(new Error("could not load " + s.src)); };
            document.head.appendChild(s);
        });
    }

    /* Returns the model bytes. Fetches the .onnx unless we are on file://,
       where fetch() cannot read a sibling file.

       Falls back to the embedded base64 if the fetch fails, which is not
       hypothetical: Tetris keeps its .onnx under training/ and only the deploy
       script copies it up beside the page, so tetris/tetris_ai.onnx is a 404
       on a local server and a 200 in production. Without the fallback this
       would work live and break in dev — the worst way round, since dev is
       where it would go unnoticed. */
    window.modelSource = function (onnxUrl, globalName, dataUrl) {
        if (location.protocol === "file:") {
            return injectEmbedded(globalName, dataUrl);
        }
        return fetch(onnxUrl)
            .then(function (r) {
                if (!r.ok) throw new Error("HTTP " + r.status + " for " + onnxUrl);
                return r.arrayBuffer();
            })
            .then(function (buf) { return new Uint8Array(buf); })
            .catch(function (err) {
                console.warn("model-source: " + err.message + " - falling back to embedded base64");
                return injectEmbedded(globalName, dataUrl);
            });
    };
}());

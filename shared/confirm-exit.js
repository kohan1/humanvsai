/* Confirm before leaving a game that is already running.
 *
 * The back link is a plain anchor a few pixels from the board, and clicking it
 * mid-game threw the run away with no warning — including a high score you
 * were part-way through beating. Games are not saved, so leaving is
 * destructive and irreversible; that is exactly the case a confirmation is
 * for.
 *
 * Only asks when a game is ACTUALLY in progress. A confirmation on an
 * untouched board would be the other failure: a dialog people learn to click
 * through without reading, which makes it useless on the one occasion it
 * matters. Each game supplies window.gameInProgress(); with no hook, or a
 * fresh board, the link behaves normally.
 */
(function () {
    "use strict";

    document.addEventListener("click", function (e) {
        var link = e.target.closest && e.target.closest(".back-nav");
        if (!link) return;

        var running = false;
        try {
            running = typeof window.gameInProgress === "function" && window.gameInProgress();
        } catch (err) {
            running = false;      // a broken hook must never trap you on the page
        }
        if (!running) return;

        if (!window.confirm("Leave this game? Your current run will be lost.")) {
            e.preventDefault();
        }
    });
}());

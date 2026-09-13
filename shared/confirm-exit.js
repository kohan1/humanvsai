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
        // Ctrl/Cmd/Shift-click and middle-click open the link somewhere else and
        // leave this tab, and its game, exactly where it is.
        if (e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey || e.altKey) return;

        var running = false;
        try {
            running = typeof window.gameInProgress === "function" && window.gameInProgress();
        } catch (err) {
            running = false;      // a broken hook must never trap you on the page
        }
        if (!running) return;

        // A game that saves itself says so (window.gameExitMessage); the default
        // is for one that does not.
        var message = typeof window.gameExitMessage === "string"
            ? window.gameExitMessage
            : "Leave this game? Your current run will be lost.";
        if (!window.confirm(message)) {
            e.preventDefault();
        }
    });
}());

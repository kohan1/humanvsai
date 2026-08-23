/* Stop the game keys from scrolling the page.
 *
 * The game pages are taller than the viewport — each one has the Inside
 * section below the fold — so the browser's default scrolling is live the
 * whole time you are playing. Arrow keys move the page, which on Snake meant
 * steering the snake and scrolling the board out of view in the same
 * keystroke.
 *
 * Tetris already did this inside its own key handler. Snake did not: its
 * listener returns early for any key it does not steer with and never calls
 * preventDefault for the ones it does. Watermelon has no key handling at all,
 * being mouse-driven, so every arrow press scrolled.
 *
 * WHY THE TARGET IS CHECKED. Space is the worst scroll offender, but on a
 * focused button or link space IS the activation key, and on a checkbox it
 * toggles; arrows move between radios and open selects. Blanket-preventing
 * either would break the speed buttons, the checkpoint switcher and keyboard
 * navigation generally. So the guard only applies when focus is somewhere
 * that has no keyboard behaviour of its own — which, while playing, is the
 * page body.
 */
(function () {
    "use strict";

    // Arrows and space only. PageUp/PageDown/Home/End scroll too, but nobody
    // hits them by accident mid-game, and every game page has the Inside
    // section below the fold — blocking them would leave a keyboard user no
    // way down the page at all. These four are the keys your hands are
    // already on while playing.
    var SCROLLERS = {
        ArrowUp: 1, ArrowDown: 1, ArrowLeft: 1, ArrowRight: 1,
        " ": 1, Spacebar: 1,
    };

    // Elements that do something of their own with these keys.
    var INTERACTIVE = /^(input|select|textarea|button|a|summary|option)$/i;

    function handlesItsOwnKeys(el) {
        if (!el || el === document.body || el === document.documentElement) return false;
        if (el.isContentEditable) return true;
        if (INTERACTIVE.test(el.tagName)) return true;
        var role = el.getAttribute && el.getAttribute("role");
        if (role && /^(button|link|checkbox|radio|menuitem|tab|slider|listbox|option|textbox)$/.test(role)) return true;
        // Anything explicitly focusable that opted into key handling.
        return el.hasAttribute && el.hasAttribute("tabindex");
    }

    window.addEventListener("keydown", function (e) {
        if (!SCROLLERS[e.key]) return;
        if (e.ctrlKey || e.metaKey || e.altKey) return;   // browser shortcuts
        if (handlesItsOwnKeys(e.target)) return;
        e.preventDefault();
    }, { passive: false });
}());

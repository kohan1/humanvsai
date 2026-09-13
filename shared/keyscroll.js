/* Stop the game keys from scrolling the page.
 *
 * The game pages are taller than the viewport — each has the Inside section
 * below the fold — so the browser's default scrolling is live the whole time
 * you are playing, and an arrow press that steers the snake would also scroll
 * the board out of view.
 *
 * Arrows and space only. PageUp/PageDown/Home/End scroll too, but nobody hits
 * them by accident mid-game, and blocking them would leave a keyboard user no
 * way down to the Inside section.
 *
 * WHAT COUNTS AS "HANDLES ITS OWN KEYS" IS PER KEY. The first version exempted
 * buttons and links from everything, which sounded careful and was wrong: a
 * mouse click leaves the button focused, so after clicking Restart or a speed
 * button the arrow keys went back to scrolling the page while also steering.
 * Buttons and links use Space and Enter; they have no use for arrows at all.
 * Arrows belong only to controls with a caret, a value or a list of options.
 */
(function () {
    "use strict";

    var ARROWS = { ArrowUp: 1, ArrowDown: 1, ArrowLeft: 1, ArrowRight: 1 };
    var SPACE  = { " ": 1, Spacebar: 1 };

    var ACTIVATES   = /^(button|a|summary|input|select|textarea|option)$/i;
    var ACTIVATE_ROLES = /^(button|link|checkbox|radio|switch|menuitem|menuitemcheckbox|menuitemradio|tab|option)$/;
    var NAVIGATES   = /^(input|select|textarea|option)$/i;
    var NAVIGATE_ROLES = /^(radio|radiogroup|slider|spinbutton|listbox|option|textbox|combobox|menu|menuitem|tablist|tab|grid|tree)$/;

    function usesKey(el, isArrow) {
        if (!el || el === document.body || el === document.documentElement) return false;
        if (el.isContentEditable) return true;
        var role = el.getAttribute ? (el.getAttribute("role") || "") : "";
        return isArrow
            ? NAVIGATES.test(el.tagName) || NAVIGATE_ROLES.test(role)
            : ACTIVATES.test(el.tagName) || ACTIVATE_ROLES.test(role);
    }

    window.addEventListener("keydown", function (e) {
        var isArrow = !!ARROWS[e.key];
        if (!isArrow && !SPACE[e.key]) return;
        if (e.ctrlKey || e.metaKey || e.altKey) return;   // browser shortcuts
        if (usesKey(e.target, isArrow)) return;
        e.preventDefault();
    }, { passive: false });
}());

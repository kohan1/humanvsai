"""
Compare the constants Watermelon duplicates between Python and JavaScript.

tools/install_model.sh already catches encoder WIDTH and ACTION COUNT drift,
because both are visible in the ONNX file's shapes. It cannot see anything that
does not change those shapes — and DIAMETERS is the worst case of exactly that.
Change the fruit sizes on one side only and every array keeps its size, every
assertion passes, the model loads, the page renders, and the policy is quietly
playing a different game from the one it trained on.

That is the project's most expensive bug class (see CLAUDE.md). This closes the
last gap in it.

    python tools/check_watermelon_sync.py
"""

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = ROOT / "watermelon" / "training" / "watermelon_env.py"
JS = ROOT / "watermelon" / "game.js"

py_src = PY.read_text(encoding="utf-8")
js_src = JS.read_text(encoding="utf-8")


def from_py(name):
    """Read a module-level assignment, ignoring comments entirely."""
    tree = ast.parse(py_src)
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        )):
            continue
        value = node.value
        # Several of these are `int(os.environ.get("NAME", <default>))` so the
        # value can be overridden per run. The default is what game.js has to
        # agree with, since the browser cannot read an environment variable.
        if isinstance(value, ast.Call):
            for arg in ast.walk(value):
                if (isinstance(arg, ast.Call)
                        and isinstance(arg.func, ast.Attribute)
                        and arg.func.attr == "get"
                        and len(arg.args) == 2):
                    return ast.literal_eval(arg.args[1])
        return ast.literal_eval(value)
    raise KeyError(f"{name} not found in {PY.name}")


def from_js(name):
    """Read `const NAME = <literal>;`, skipping /* */ comments that mention it."""
    stripped = re.sub(r"/\*.*?\*/", "", js_src, flags=re.S)
    # Arrays contain commas, so match a bracketed literal or a bare scalar --
    # not "everything up to the next comma", which truncates DIAMETERS to "[24".
    m = re.search(rf"\b{name}\s*=\s*(\[[^\]]*\]|[^;,\n]+)", stripped)
    if not m:
        raise KeyError(f"{name} not found in {JS.name}")
    return ast.literal_eval(m.group(1).strip())


CHECKS = [
    # (name in python, name in js)
    ("DIAMETERS", "DIAMETERS"),
    ("POINTS", "POINTS"),
    ("CANVAS_W", "CANVAS_W"),
    ("CANVAS_H", "CANVAS_H"),
    ("GRID_W", "GRID_W"),
    ("GRID_H", "GRID_H"),
    ("GRID_CHANNELS", "GRID_CHANNELS"),
    ("N_SCALARS", "N_SCALARS"),
    ("SPAWN_TIERS", "SPAWN_TIERS"),
    ("MAX_FRUIT_NORM", "MAX_FRUIT_NORM"),
    ("N_DROP_COLUMNS", "N_DROP_COLUMNS"),
]

bad = []
print(f"{'constant':<18} {'watermelon_env.py':<34} {'game.js'}")
print("-" * 88)
for py_name, js_name in CHECKS:
    try:
        a, b = from_py(py_name), from_js(js_name)
    except KeyError as e:
        print(f"{py_name:<18} !! {e}")
        bad.append(py_name)
        continue
    # N_DROP_COLUMNS is `int(os.environ.get(...))` in Python, so read the default.
    ok = a == b or (isinstance(a, float) and float(b) == a)
    print(f"{py_name:<18} {str(a):<34} {str(b)}{'' if ok else '   <-- MISMATCH'}")
    if not ok:
        bad.append(py_name)

print()
if bad:
    print(f"FAILED: {len(bad)} constant(s) differ — the browser is playing a "
          "different game from the trainer")
    for n in bad:
        print(f"  - {n}")
    sys.exit(1)
print("in sync")
sys.exit(0)

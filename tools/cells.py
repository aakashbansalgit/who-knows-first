"""Read a pipeline stage (a '# %%' percent-format script) as a list of cells.

Used by tools/make_notebooks.py to build the Colab notebooks and by the tests to run a stage cell by cell.
"""
import os
import re
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARK = re.compile(r"^# %%(.*)$")


def read_cells(path):
    """Return [(kind, name, source)] for a stage file; kind is 'md' or 'code'."""
    lines = open(path, encoding="utf-8").read().split("\n")
    cells, cur = [], None
    for line in lines:
        m = MARK.match(line)
        if m:
            if cur is not None:
                cells.append(cur)
            tag = m.group(1).strip()
            cur = ("md", None, []) if tag == "[markdown]" else ("code", tag or None, [])
        elif cur is not None:
            cur[2].append(line)
    if cur is not None:
        cells.append(cur)
    out = []
    for kind, name, body in cells:
        if kind == "md":
            text = "\n".join(l[2:] if l.startswith("# ") else l[1:] if l.startswith("#") else l for l in body).strip("\n")
            out.append(("md", None, text))
        else:
            out.append(("code", name, "\n".join(body).strip("\n") + "\n"))
    return out


def load_stage(stem):
    """The stage's code cells as attributes (by cell name) plus CELLS = [(kind, source)], like the original notebook modules."""
    cells = read_cells(os.path.join(ROOT, "pipeline", stem + ".py"))
    ns = {name: src for kind, name, src in cells if kind == "code" and name}
    return SimpleNamespace(CELLS=[(kind, src) for kind, name, src in cells], **ns)

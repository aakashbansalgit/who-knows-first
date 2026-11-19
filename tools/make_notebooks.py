"""Build Colab notebooks (notebooks/*.ipynb) from the pipeline stages, so both stay identical.

    python tools/make_notebooks.py

Each '# %%' cell of pipeline/<stage>.py becomes one notebook cell. Run this after editing a stage.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import ROOT, read_cells

STAGES = json.load(open(os.path.join(ROOT, "tools", "stages.json")))


def to_lines(text):
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def build(stem):
    cells = []
    for i, (kind, name, src) in enumerate(read_cells(os.path.join(ROOT, "pipeline", stem + ".py"))):
        cid = f"{stem.replace('_', '-')}-{i:02d}"
        if kind == "md":
            cells.append({"cell_type": "markdown", "id": cid, "metadata": {}, "source": to_lines(src)})
        else:
            cells.append({"cell_type": "code", "execution_count": None, "id": cid, "metadata": {}, "outputs": [],
                          "source": to_lines(src.rstrip("\n"))})
    meta = {"colab": {"provenance": [], "name": stem + ".ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
    if STAGES[stem].get("accelerator") == "GPU":
        meta["accelerator"] = "GPU"
        meta["colab"]["gpuType"] = "T4"
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": meta, "cells": cells}
    path = os.path.join(ROOT, "notebooks", stem + ".ipynb")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return path, len(cells)


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
    for stem in STAGES:
        path, n = build(stem)
        print(f"{os.path.relpath(path, ROOT)}: {n} cells")

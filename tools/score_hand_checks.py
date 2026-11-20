"""Score the hand-coded validation samples behind the error rates in Sections 3.1, 3.3 and 10.1.

    WKF_PROJECT_DIR=/path/to/project python tools/score_hand_checks.py

Reads features/nb7/validate_readouts.xlsx (200 readouts drawn by label) and features/nb7/validate_matches.xlsx
(100 disclosure matches drawn by match kind), both filled in by hand, and weights each stratum back to its size in
the full sample. Rates are printed under the definitions stated next to each line.
"""
import os

import numpy as np
import pandas as pd

PROJECT_DIR = os.environ.get("WKF_PROJECT_DIR", "/content/drive/MyDrive/biopharma_tiv")
FEAT = os.path.join(PROJECT_DIR, "features")


def wmean(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    return float((x * w).sum() / w.sum())


def readouts():
    V = pd.read_excel(os.path.join(FEAT, "nb7", "validate_readouts.xlsx"))
    R = pd.read_parquet(os.path.join(FEAT, "readout_events.parquet"))
    tone = V["true tone (positive/negative/mixed/unclear)"].astype(str).str.lower().str.strip()
    V["true"] = tone.replace({"neutral": "unclear", "nan": "unclear"})
    V["is_readout"] = V["is a trial readout (Y/N)"].astype(str).str.upper().str.strip().str[:1] == "Y"
    typ = V["topline or data update"].astype(str).str.lower().str.strip()
    V["true_topline"] = typ.str.startswith("top") | typ.str.startswith("end")
    V["label_topline"] = V["label_kind"] != "data_update"
    stratum = lambda d, kind, tone_: np.where(d[kind] == "data_update", "data_update", d[tone_])
    V["stratum"] = stratum(V, "label_kind", "label_tone")
    print("Readout labels (200 hand-coded, weighted back to each label's share)")
    for pop_name, pop in [("all 5,871 readouts", R), ("the 3,327 with prices", R[R["has_prices"].astype(bool)])]:
        N = pd.Series(stratum(pop, "readout_kind", "readout_tone")).value_counts()
        w = V["stratum"].map(N / V["stratum"].value_counts())
        top = V["label_topline"]
        right_type = V["is_readout"] & (V["label_topline"] == V["true_topline"])
        right_tone = V["true"] == V["label_tone"]
        swap = ((V["label_tone"] == "positive") & (V["true"] == "negative")) | ((V["label_tone"] == "negative") & (V["true"] == "positive"))
        print(f"  weighted to {pop_name}: trial readouts {wmean(V['is_readout'], w):.1%} | right type {wmean(right_type, w):.1%} | "
              f"topline right tone {wmean(right_tone[top], w[top]):.1%} | success and failure swapped {wmean(swap[top], w[top]):.1%}")
    for lab in ["positive", "negative", "mixed", "unclear"]:
        s = V[V["label_topline"] & (V["label_tone"] == lab)]
        clear = s["true"].isin(["positive", "negative", "mixed"])
        print(f"  labelled {lab:8s} n = {len(s):2d}: hand coding {s['true'].value_counts().to_dict()} | confirmed {(s['true'] == lab).mean():.0%} "
              f"of all, {(s['true'] == lab).sum()} of {int(clear.sum())} with a clear outcome")


def matches():
    V = pd.read_excel(os.path.join(FEAT, "nb7", "validate_matches.xlsx"))
    SD = pd.read_parquet(os.path.join(FEAT, "stop_disclosure.parquet"))
    ok = (V["about this trial (Y/N)"].astype(str).str.upper().str.strip().str[:1] == "Y") & \
         (V["says it stopped or will stop (Y/N)"].astype(str).str.upper().str.strip().str[:1] == "Y")
    print("\nDisclosure matches (100 hand-coded, by match kind)")
    rk = V["risk"].astype(str)
    print("  right by kind: " + " | ".join(f"{k} {ok[rk == k].mean():.0%} of {(rk == k).sum()}" for k in rk.value_counts().index))
    col = next((c for c in ["risk", "match_risk", "first_risk", "match_kind"] if c in SD.columns), None)
    if col is None:
        print(f"  (stop_disclosure.parquet has no match-kind column to weight by; columns: {list(SD.columns)[:20]})")
        return
    first = SD[SD[col].notna()]
    N = first[col].astype(str).value_counts()
    w = rk.map(N / rk.value_counts())
    if w.isna().any():
        print(f"  match kinds in the sample {sorted(rk.unique())} vs in the data {sorted(N.index)}: weights incomplete")
        return
    print(f"  weighted back to the {len(first):,} first disclosures: {wmean(ok, w):.1%} right")


if __name__ == "__main__":
    readouts()
    matches()

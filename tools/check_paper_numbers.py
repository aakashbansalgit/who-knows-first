"""Compare the run-up results saved by pipeline/10_robustness.py with the numbers reported in the paper.

    WKF_PROJECT_DIR=/path/to/project python tools/check_paper_numbers.py

Reads features/nb10/runup_20_placebos.csv (Steps 19, 19c and 20) and features/nb10/runup30_robust.csv (Step 21).
Point estimates should match the paper to the rounding shown there. Bootstrap intervals and p-values depend on the
random stream, which depends on which steps ran before in the same session, so they are flagged only when they move
by more than half a percentage point (intervals) or 0.02 (p-values).
"""
import os
import sys

import pandas as pd

PROJECT_DIR = os.environ.get("WKF_PROJECT_DIR", "/content/drive/MyDrive/biopharma_tiv")
NB10 = os.path.join(PROJECT_DIR, "features", "nb10")

# (file, check label as saved) -> (gap, lower, upper) as reported (gaps in return units); None = not reported
PAPER = {
    ("runup_20_placebos.csv", "Headline: days -10..-2, without releases 2-7 days before the 8-K"): (0.0229, -0.0009, 0.0491),
    ("runup_20_placebos.csv", "Same, days -5..-2"): (0.0089, -0.0048, 0.0240),
    ("runup_20_placebos.csv", "Same, medians"): (0.0080, -0.0185, 0.0284),
    ("runup_20_placebos.csv", "Same, all readouts"): (0.0312, 0.0053, 0.0584),
    ("runup_20_placebos.csv", "Same, all readouts, releases that mention no medical meeting"): (0.0279, 0.0014, 0.0574),
    ("runup_20_placebos.csv", "Same, releases dated on the 8-K day or the day before"): (0.0285, -0.0014, 0.0599),
    ("runup_20_placebos.csv", "Same, without placebo adjustment (raw), headline sample"): (0.0248, -0.0001, 0.0510),
    ("runup_20_placebos.csv", "Same, beta-adjusted (beta to XBI from days -250..-30)"): (0.0234, -0.0012, 0.0521),
    ("runup_20_placebos.csv", "Same, market model (alpha and beta from days -250..-30), no placebo"): (0.0229, -0.0032, 0.0520),
    ("runup_20_placebos.csv", "Headline, tone and language model agree"): (0.0259, 0.0017, 0.0515),
    ("runup_20_placebos.csv", "Days -30..-2, headline sample"): (0.0697, 0.0187, 0.1222),
    ("runup_20_placebos.csv", "Days -30..-2, medians"): (0.0404, -0.0144, 0.0892),
    ("runup_20_placebos.csv", "Days -30..-2, all readouts"): (0.0788, 0.0229, 0.1271),
    ("runup_20_placebos.csv", "Days -30..-2, tone and language model agree"): (0.0700, 0.0171, 0.1247),
    ("runup_20_placebos.csv", "Days -30..-2, tone labels (headline sample) (standardized, SD)"): (0.255, 0.042, 0.453),
    ("runup_20_placebos.csv", "    placebo dates, days -10..-2"): (-0.0116, -0.0189, -0.0041),
    ("runup30_robust.csv", "Without placebo adjustment (raw)"): (0.0699, 0.0165, 0.1183),
    ("runup30_robust.csv", "Beta-adjusted (beta to XBI from days -280..-31), net of placebo dates"): (0.0668, 0.0198, 0.1184),
    ("runup30_robust.csv", "Market model (alpha and beta from days -280..-31), no placebo"): (0.0472, -0.0049, 0.0944),
    ("runup30_robust.csv", "Placebo draw 2 of 20 dates (seed 2127)"): (0.0798, None, None),
    ("runup30_robust.csv", "Placebo draw 5 of 20 dates (seed 2130)"): (0.0759, None, None),
    ("runup30_robust.csv", "No clipping"): (0.0907, 0.0264, 0.1572),
    ("runup30_robust.csv", "Clipped 2.5%"): (0.0594, 0.0126, 0.1058),
    ("runup30_robust.csv", "Clipped 5%"): (0.0499, 0.0031, 0.0931),
    ("runup30_robust.csv", "Trimmed 5% within each outcome"): (0.0571, 0.0156, 0.0953),
    ("runup30_robust.csv", "All readouts, releases that mention no medical meeting"): (0.0809, 0.0213, 0.1462),
    ("runup30_robust.csv", "Releases dated on the 8-K day or the day before"): (0.0677, 0.0070, 0.1225),
    ("runup30_robust.csv", "Clustered by firm and calendar week"): (0.0697, 0.0184, 0.1210),
    ("runup30_robust.csv", "Clustered by firm and calendar month"): (0.0697, 0.0164, 0.1229),
    ("runup30_robust.csv", "Share of the separation before the filing, days -30..-2"): (0.126, 0.039, 0.201),
    ("runup30_robust.csv", "Share of the separation before the filing, days -10..-2"): (0.045, -0.002, 0.095),
}
# p-values reported for the window test (one-sided, best of 28 windows)
PAPER_P = {
    ("runup30_robust.csv", "Selection-adjusted, label permutation (2,000 draws)"): 0.037,
    ("runup30_robust.csv", "Selection-adjusted, placebo dates as pseudo-events (1,000 draws)"): 0.034,
}


def main():
    frames = {}
    for f in {k[0] for k in list(PAPER) + list(PAPER_P)}:
        p = os.path.join(NB10, f)
        if not os.path.exists(p):
            sys.exit(f"Missing {p}: run pipeline/10_robustness.py through Step 21 first.")
        frames[f] = pd.read_csv(p).drop_duplicates("check", keep="last").set_index("check")
    bad = warn = 0
    for (f, lab), (g, lo, hi) in PAPER.items():
        if lab not in frames[f].index:
            print(f"MISSING  {lab}")
            bad += 1
            continue
        r = frames[f].loc[lab]
        tol = 0.0006 if abs(g) < 1 else 0.002
        ok = abs(r["gap"] - g) <= (0.002 if "standardized" in lab or "Share" in lab else tol)
        drift = [abs(r[c] - v) for c, v in (("lo", lo), ("hi", hi)) if v is not None and pd.notna(r.get(c))]
        flag = "OK  " if ok else "DIFF"
        note = "" if not drift or max(drift) <= 0.005 else f"  (interval moved by up to {max(drift):.3f})"
        bad += not ok
        warn += bool(note)
        print(f"{flag}  {lab[:72]:72s} paper {g:+.4f}  now {r['gap']:+.4f}{note}")
    for (f, lab), p in PAPER_P.items():
        r = frames[f].loc[lab] if lab in frames[f].index else None
        if r is None:
            print(f"MISSING  {lab}")
            bad += 1
            continue
        note = "OK  " if abs(r["p"] - p) <= 0.02 else "NOTE"
        warn += note == "NOTE"
        print(f"{note}  {lab[:72]:72s} paper p = {p:.3f}  now p = {r['p']:.3f}")
    print(f"\n{bad} point estimates differ from the paper; {warn} intervals or p-values moved by more than the tolerance.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

"""Archived: Notebook 10, Step 14 (run-up without early releases, 5 placebo dates per readout).

Superseded by Step 19 of pipeline/10_robustness.py, which uses 20 placebo dates; nothing later depends on it.
Kept for the record and not run by the pipeline. It needs Steps 0 to 3 of 10_robustness.py in the same session.
"""


# %% [markdown]
# ## Step 14: Run-up without readouts whose press release came out days before the 8-K
# Step 13 shows that some releases came out 2 to 7 days before their 8-K. For those readouts the run-up window (days -10 to -2 before the 8-K) contains the release itself, which would inflate the success-failure gap. This step drops them.


# %% RUNUP_EARLY
if "F13" in globals() and RU is not None:
    R14 = RU.merge(F13[["accession", "lag"]], on="accession", how="left")
    early = R14["lag"].between(2, 7)
    print(f"Readouts whose release came out 2-7 days before the 8-K: {int(early.sum())} ("
          + ", ".join(f"{oc} {int((early & (R14['outcome'] == oc)).sum())}" for oc in ["success", "failure", "mixed"]) + ")")
    print("Their 'run-up', days -10..-2 minus placebo (contains the release): "
          + " | ".join(f"{oc} {R14.loc[early & (R14['outcome'] == oc), 'xrun_10'].mean():+.1%}" for oc in ["success", "failure"]))
    sf = lambda z: z.loc[z["outcome"] == "success", "y"].mean() - z.loc[z["outcome"] == "failure", "y"].mean()
    md = lambda z: z.loc[z["outcome"] == "success", "y"].median() - z.loc[z["outcome"] == "failure", "y"].median()
    print("\n=== Run-up, successes minus failures, minus placebo (clipped; 95% interval resampling firms) ===")
    for lab, keep in [("all (Step 3)", pd.Series(True, index=R14.index)), ("without releases 2-7 days early", ~early),
                      ("release dated on the 8-K day or the day before", R14["lag"].between(0, 1))]:
        for v, w in [("xrun_10", "days -10..-2"), ("xrun_5", "days -5..-2")]:
            d = R14[keep & R14["outcome"].isin(["success", "failure"])][["cik", "outcome", v]].dropna().rename(columns={v: "y"})
            d["y"] = clip(d["y"])
            g, lo, hi, p = firm_boot(d, sf)
            gm, lom, him, pm = firm_boot(d, md)
            print(f"  {lab:48s} {w}: mean {g:+.2%} [{lo:+.2%}, {hi:+.2%}] p = {p:.3f} | median {gm:+.2%} [{lom:+.2%}, {him:+.2%}] "
                  f"p = {pm:.3f} | n = {len(d)}")
            if lab.startswith("without") and w == "days -10..-2":
                RESULTS.append({"check": "Run-up, successes minus failures, without releases 2-7 days before the 8-K", "estimate": f"{g:+.2%}",
                                "interval": f"[{lo:+.2%}, {hi:+.2%}]", "p": p, "family": "",
                                "reading": "close to Step 3: early releases do not create the run-up"})
else:
    print("Skipped: run Steps 3 and 13 first")

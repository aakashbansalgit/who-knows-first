"""Offline test of Notebook 10 on synthetic inputs with planted effects: a run-up before successes only, failures more often
missing prices, a registry that is ignored, stop 8-Ks with restructurings that fall more, never-disclosed stops at large firms."""
import os, re, sys, shutil, io, contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "tools"))
from cells import load_stage
C = load_stage("10_robustness")

P = os.environ.get("P18", "/tmp/nb10_test")
shutil.rmtree(P, ignore_errors=True)
F, PX = f"{P}/features", f"{P}/prices"
for d in (F, PX, f"{F}/nb7", f"{F}/nb8/history", f"{F}/nb9"):
    os.makedirs(d, exist_ok=True)
rng = np.random.default_rng(int(os.environ.get("SEED10", "11")))
cal = pd.bdate_range("2014-01-01", "2025-12-31")
pd.DataFrame({"date": cal, "Close": 100 * np.exp(np.cumsum(rng.normal(0, 0.015, len(cal)))), "Volume": 1e6}).to_parquet(f"{PX}/XBI.parquet")

ciks = list(range(8000, 8120))
tick = {c: f"T{c}" for c in ciks}
ret = {}
for c in ciks:
    ret[c] = rng.normal(-0.0003, 0.04, len(cal))

ro = []
for i in range(2600):
    c = ciks[i % len(ciks)]
    j = int(rng.integers(300, len(cal) - 40))
    d = cal[j]
    kind = "topline_or_endpoint" if rng.random() < 0.45 else "data_update"
    tone = rng.choice(["positive", "negative", "mixed", "unclear"], p=[0.57, 0.14, 0.15, 0.14])
    priced = rng.random() < (0.45 if tone == "negative" else 0.6)
    if kind == "topline_or_endpoint" and tone == "positive":
        ret[c][j - 10:j - 1] += 0.008                                 # planted run-up before successes
    imm = {"positive": 0.12, "negative": -0.34, "mixed": -0.09, "unclear": 0.0}[tone] if kind == "topline_or_endpoint" else 0.0
    if kind == "topline_or_endpoint" and tone == "positive" and rng.random() < 0.01:
        imm = 4.0
    ret[c][j] += imm
    acc = f"0001{i:06d}-20-{i:06d}"
    ro.append({"accession": acc, "cik": c, "ticker": tick[c] if priced else None, "event_date": d, "day0": d, "readout_kind": kind,
               "readout_tone": tone, "has_prices": priced, "timing": rng.choice(["pre_open", "after_close", "market_hours"]),
               "earnings_day": rng.random() < 0.05, "phase": float(rng.choice([1, 2, 3])), "acceptance_et": d + pd.Timedelta(hours=7)})
ro = pd.DataFrame(ro)
for c in ciks:
    close = 10 * np.exp(np.cumsum(np.log1p(np.clip(ret[c], -0.9, 10))))
    pd.DataFrame({"date": cal, "Close": close, "Volume": 1e5}).to_parquet(f"{PX}/{tick[c]}.parquet")

# Notebook 4's stats for readouts with prices
xb = pd.read_parquet(f"{PX}/XBI.parquet").set_index("date")["Close"].pct_change()
stats = []
for r in ro.itertuples():
    if not r.has_prices:
        stats.append({})
        continue
    s = pd.read_parquet(f"{PX}/{r.ticker}.parquet").set_index("date")["Close"].pct_change() - xb
    j = cal.get_loc(r.day0)
    stats.append({"car_1_1": s.iloc[j - 1:j + 2].sum(), "car01": s.iloc[j:j + 2].sum(), "car_2_20": s.iloc[j + 2:j + 21].sum(),
                  "car_m10_m2": s.iloc[j - 10:j - 1].sum(), "pre_vol": s.iloc[j - 250:j - 30].std(), "log_dollar_vol": float(rng.normal(15, 1.5)),
                  "max_abs_ar": s.iloc[j - 1:j + 2].abs().max(), "price_pre": 10.0, "abn_volume": float(np.exp(rng.normal(1, 0.5)))})
ro = pd.concat([ro, pd.DataFrame(stats)], axis=1)
ro["abs_car"] = ro["car_1_1"].abs()
ro["year"] = ro["event_date"].dt.year
ro["overlap"], ro["recent_ipo"] = False, False
ro["extreme"] = ro["max_abs_ar"].gt(2.0)
ro["clean"] = ro["has_prices"] & ~ro["earnings_day"] & ~ro["extreme"]
ro.to_parquet(f"{F}/readout_events.parquet", index=False)

# stops at the registry date and first disclosures
se, dse, sd, hist = [], [], [], {c: [] for c in ciks}
for k in range(700):
    c = ciks[k % len(ciks)]
    d = cal[int(rng.integers(300, len(cal) - 200))]
    grp = rng.choice(["filing first", "registry first or only"], p=[0.45, 0.55])
    det = "" if grp == "filing first" else rng.choice(["never disclosed (within a year)", "disclosed after the registry"], p=[0.8, 0.2])
    mat = rng.random() < 0.3
    se.append({"event_id": f"{c}_{k}", "nct_ids": f"NCT{k:08d}", "cik": c, "registry_date": d, "day0": d, "group": grp, "registry_detail": det, "has_prices": True,
               "car01": rng.normal(0.005 if grp == "filing first" else 0.0, 0.05), "material_reason": mat, "late_stage": rng.random() < 0.3,
               "firm_lead_trials": int(rng.integers(1, 60)), "same_day_8k": rng.random() < 0.1, "near_readout": False, "ticker": tick[c]})
    if grp == "filing first" or det.startswith("disclosed after"):
        fd = d - pd.Timedelta(days=int(rng.integers(20, 200))) if grp == "filing first" else d + pd.Timedelta(days=int(rng.integers(20, 200)))
        fd = cal[cal.searchsorted(fd)]
        chan = rng.choice(["8-K", "10-Q/10-K/20-F/40-F"], p=[0.55, 0.45])
        rest = chan == "8-K" and rng.random() < 0.4
        dse.append({"cik": c, "first_date": fd, "day0": fd, "channel": chan, "when": "before the registry" if grp == "filing first" else "after the registry",
                    "has_prices": True, "car01": rng.normal((-0.18 if rest else -0.08) if chan == "8-K" else -0.02, 0.06)})
        if chan == "8-K":
            hist[c].append({"form": "8-K", "filingDate": fd.strftime("%Y-%m-%d"), "accessionNumber": f"{c}-{k}",
                            "items": "2.05,8.01" if rest else "8.01"})
    sd.append({"stop_id": k, "nct_id": f"NCT{k:08d}", "cik": c, "registry_date": d, "in_sample": True,
               "timing": "none" if det.startswith("never") else "before", "firm_lead_trials": int(rng.integers(1, 60)),
               "names_searched": f"drug-{k}, DRUG {k}", "firm": f"Firm {c}", "why_stopped": "business decision", "brief_title": f"Trial {k}"})
pd.DataFrame(se).to_parquet(f"{F}/stop_events.parquet", index=False)
pd.DataFrame(dse).to_parquet(f"{F}/stop_disclosure_events.parquet", index=False)
pd.DataFrame(sd).to_parquet(f"{F}/stop_disclosure.parquet", index=False)
for c, rows in hist.items():
    pd.DataFrame(rows, columns=["form", "filingDate", "accessionNumber", "items"]).to_parquet(f"{F}/nb8/history/{c}.parquet", index=False)

S = pd.DataFrame(se)
S["sgroup"] = np.where(S["group"] == "filing first", "filing first",
                       np.where(S["registry_detail"].str.startswith("never"), "registry first, never disclosed", "registry first, disclosed later"))
S["xcar_2_60"] = rng.normal(0.02, 0.2, len(S))
S["xcar_2_120"] = rng.normal(0.04, 0.3, len(S))
S["month"] = S["registry_date"].dt.to_period("M").astype(str)
S.to_parquet(f"{F}/nb7/drift_registry.parquet", index=False)

def outcomes(X, gcol, never_val):
    X = X.copy()
    big = X["cik"] % 3 == 0                                            # never-disclosed stops concentrate at large, healthy firms
    X["small"] = (~big).astype(int)
    X["material"] = rng.integers(0, 2, len(X))
    X["late"] = rng.integers(0, 2, len(X))
    X["year"] = rng.integers(2017, 2025, len(X))
    pf = np.where(big, 0.02, 0.09)
    X["fail_365"] = (rng.random(len(X)) < pf).astype(float)
    X["fail_730"] = np.maximum(X["fail_365"], (rng.random(len(X)) < pf).astype(float))
    X["notice_365"] = (rng.random(len(X)) < np.where(big, 0.15, 0.35)).astype(float)
    X["acq_365"] = (rng.random(len(X)) < 0.06).astype(float)
    X["acq_730"] = np.maximum(X["acq_365"], (rng.random(len(X)) < 0.03).astype(float))
    return X

R = S[["cik", "sgroup"]].copy()
R.loc[R["cik"] % 3 == 0, "sgroup"] = np.where(rng.random((R["cik"] % 3 == 0).sum()) < 0.75, "registry first, never disclosed", "filing first")
outcomes(R, "sgroup", "registry first, never disclosed").to_parquet(f"{F}/nb8/exit_outcomes_registry.parquet", index=False)
Ps = pd.DataFrame({"cik": [ciks[k % len(ciks)] for k in range(1000)], "stop_id": [k % 700 for k in range(1000)]})
Ps["registry_date"] = [cal[int(rng.integers(300, len(cal) - 200))] for _ in range(len(Ps))]
Ps["files_8k"] = np.where(Ps["cik"] % 5 == 0, 0, 1)
Ps["outcome"] = np.where(Ps["cik"] % 3 == 0, rng.choice(["never", "8-K first"], len(Ps), p=[0.7, 0.3]),
                         rng.choice(["never", "8-K first", "10-Q/10-K first"], len(Ps), p=[0.3, 0.45, 0.25]))
outcomes(Ps, "outcome", "never").to_parquet(f"{F}/nb8/exit_outcomes_stops.parquet", index=False)

top = ro[(ro["readout_kind"] == "topline_or_endpoint") & ro["has_prices"]].copy()
top["outcome"] = top["readout_tone"].map({"positive": "success", "negative": "failure", "mixed": "mixed"})
E = top.dropna(subset=["outcome"])[["accession", "cik", "event_date", "outcome", "year", "log_dollar_vol"]].copy()
E["fail"] = (E["outcome"] == "failure").astype(int)
E["post"] = (E["event_date"] >= "2023-06-01").astype(int)
E["base_sell"] = rng.normal(1, 0.5, len(E))
for v in ["sell", "sell_noplan", "sell_plan", "buy"]:
    E[f"{v}_m60_m1"] = (rng.random(len(E)) < {"sell": 0.3, "sell_noplan": 0.13, "sell_plan": 0.22, "buy": 0.08}[v]).astype(int)
E.to_parquet(f"{F}/nb9/readout_insiders.parquet", index=False)

# Notebook 3b hits: disclosures for the stops that have one, some through shared drug names
sdf = pd.DataFrame(sd)
hits = []
for r in sdf.itertuples():
    if r.timing != "none":
        d0 = int(rng.integers(-200, -1)) if rng.random() < 0.6 else int(rng.integers(1, 300))
        hits.append({"stop_id": r.stop_id, "channel": rng.choice(["8-K", "10-Q/10-K"]), "days": d0, "stop_hit": True,
                     "match_kind": "drug_name", "name_trials": int(rng.choice([1, 2, 8]))})
pd.DataFrame(hits).to_parquet(f"{F}/stop_disclosure_hits.parquet", index=False)

# Notebook 5 texts (with datelines, some meeting releases) and Notebook 9 trades
os.makedirs(f"{F}/nb5", exist_ok=True)
tx = []
for r in ro[ro["has_prices"]].itertuples():
    d = r.event_date - pd.Timedelta(days=int(rng.choice([0, 0, 0, 1, 3])))
    meet = " The data were presented at the American Society of Clinical Oncology Annual Meeting." if rng.random() < 0.3 else ""
    raw = f"CAMBRIDGE, Mass., {d:%B} {d.day}, {d.year} (GLOBE NEWSWIRE) -- Acme today announced topline results.{meet} More text."
    tx.append({"accession": r.accession, "version": "t1_5000", "source": "press_release", "n_chars": len(raw), "text_raw": raw, "text": raw})
pd.DataFrame(tx).to_parquet(f"{F}/nb5/texts.parquet", index=False)
trs = []
for c in ciks:
    for _ in range(int(rng.integers(20, 80))):
        code = rng.choice(["S", "P", "G", "M", "A"], p=[0.4, 0.1, 0.1, 0.2, 0.2])
        od = rng.random() < 0.7
        trs.append({"accession": f"x{c}{_}", "cik": c, "trans_date": cal[int(rng.integers(0, len(cal)))], "code": code,
                    "ad": "D" if code in ("S", "G") else "A", "shares": 100.0, "price": 10.0, "value": 1000.0, "owner": int(rng.integers(1, 6)),
                    "officer": od, "director": False, "ten_pct": (not od) or rng.random() < 0.1, "plan": rng.random() < 0.4, "od": od})
pd.DataFrame(trs).to_parquet(f"{F}/nb9/insider_trades.parquet", index=False)

# Notebook 2 files, AACT snapshot names and saved figures
os.makedirs(f"{P}/ctgov/aact_monthly", exist_ok=True)
for m in ["201801", "201802", "202601"]:
    pd.DataFrame({"x": [1]}).to_parquet(f"{P}/ctgov/aact_monthly/aact_{m}.parquet")
tfr, curr = [], []
for c in ciks:
    for k in range(int(rng.integers(1, 40))):
        n = f"NCTX{c}{k:03d}"
        tfr.append({"nct_id": n, "cik": c, "role": "lead"})
        curr.append({"nct_id": n, "first_posted": str(cal[int(rng.integers(0, len(cal)))].date())})
pd.DataFrame(tfr).to_parquet(f"{P}/ctgov/trial_firm.parquet", index=False)
pd.DataFrame(curr).to_parquet(f"{P}/ctgov/industry_trials_current.parquet", index=False)
os.makedirs(f"{F}/figures", exist_ok=True)
for fnm in ["fig6_readout_car_v2.png", "fig13_drift.png", "fig15_exits.png"]:
    open(f"{F}/figures/{fnm}", "wb").write(b"png")

# inputs for the paper figures
os.makedirs(f"{F}/nb6", exist_ok=True)
gr = ["all", "both labels agree", "clean events", "era 2010-2015", "era 2016-2020", "era 2021-2025", "2018 and later", "phase 3", "phase 2",
      "phase unknown", "small firms (dollar volume)", "mid firms (dollar volume)", "large firms (dollar volume)"]
ip = pd.DataFrame({"window": "days -1..+1", "mixed_as_failure": False, "group": gr, "implied": rng.uniform(0.6, 0.8, len(gr))})
ip["implied_lo"], ip["implied_hi"], ip["realized"] = ip["implied"] - 0.06, ip["implied"] + 0.07, ip["implied"] + 0.06
pd.concat([ip, ip.assign(mixed_as_failure=True)]).to_csv(f"{F}/nb6/implied_pos.csv", index=False)
pd.DataFrame({"reason": rng.choice(["efficacy", "safety", "business", "not given", "enrollment", "other"], 1083),
              "outcome": rng.choice(["8-K first", "10-Q/10-K first", "other filing first", "never"], 1083, p=[0.3, 0.2, 0.05, 0.45])}
             ).to_parquet(f"{F}/nb7/channel_choice.parquet", index=False)
DD = pd.DataFrame(dse)
DD["chan"] = np.where(DD["channel"] == "8-K", "8-K", "10-Q/10-K")
DD["ticker"] = DD["cik"].map(tick)
DD.to_parquet(f"{F}/nb7/drift_disclosure.parquet", index=False)
XSx = pd.read_parquet(f"{F}/nb8/exit_outcomes_stops.parquet")
XSx["failure_firm"] = rng.random(len(XSx)) < 0.5
XSx["exit_date"] = [d + pd.Timedelta(days=int(rng.integers(10, 1500))) if rng.random() < 0.15 else pd.NaT for d in XSx["registry_date"]]
XSx.to_parquet(f"{F}/nb8/exit_outcomes_stops.parquet", index=False)
t5 = ro[ro["readout_kind"] == "topline_or_endpoint"][["accession", "event_date", "abs_car"]].copy()
t5["exp_move"] = np.clip(t5["abs_car"].fillna(0.1) * 100 * rng.uniform(0.5, 1.5, len(t5)), 0, 99)
t5.to_parquet(f"{F}/nb5/readout_text_features.parquet", index=False)

# Step 18 inputs: NCT numbers on readouts, Notebook 6 flags, P(met), exit types
ro2 = pd.read_parquet(f"{F}/readout_events.parquet")
ro2["nct_ids"] = np.where(rng.random(len(ro2)) < 0.7, [f"NCT{k:08d}" for k in range(len(ro2))], None)
ro2.to_parquet(f"{F}/readout_events.parquet", index=False)
tl = ro2[(ro2["readout_kind"] == "topline_or_endpoint") & ro2["has_prices"] & ro2["nct_ids"].notna()]
fl6 = tl[["accession"]].sample(frac=0.6, random_state=1).copy()
fl6["any_flag"] = (rng.random(len(fl6)) < 0.3).astype(int)
fl6.to_parquet(f"{F}/nb6/readout_flags.parquet", index=False)
t5b = pd.read_parquet(f"{F}/nb5/readout_text_features.parquet").merge(ro2[["accession", "readout_tone"]], on="accession")
t5b["p_met"] = np.clip(np.where(t5b["readout_tone"] == "positive", 0.8, np.where(t5b["readout_tone"] == "negative", 0.2, 0.4))
                       + rng.normal(0, 0.25, len(t5b)), 0, 1)
t5b.drop(columns=["readout_tone"]).to_parquet(f"{F}/nb5/readout_text_features.parquet", index=False)
XSy = pd.read_parquet(f"{F}/nb8/exit_outcomes_stops.parquet")
XSy["exit_type"] = np.where(XSy["exit_date"].isna(), "", np.where(XSy["failure_firm"], "bankruptcy", "acquired"))
XSy.to_parquet(f"{F}/nb8/exit_outcomes_stops.parquet", index=False)

# run every cell
ns = {}
buf = io.StringIO()
names = [n for n in ["CONFIG", "SETUP", "SURV", "IMPLIED", "RUNUP", "REGISTRY", "NULLS", "EXITS", "DIAG", "EXTRAS", "SUMMARY",
                     "RUNUP_SPLIT", "BHAR", "INSIDERS2", "FRIDAY_PR", "RELIABLE", "REVIEW", "PAPERFIGS", "ROBUST", "RUNUP20", "SUMMARY2"]]
with contextlib.redirect_stdout(buf):
    for n in names:
        src = getattr(C, n)
        if n == "CONFIG":
            src = src.replace('"/content/drive/MyDrive/biopharma_tiv"', f'"{P}"').replace("N_BOOT = 1000", "N_BOOT = 200")
        exec(src, ns)
out = buf.getvalue()
print(out)

# checks on planted effects
RT = pd.read_csv(f"{F}/nb10/runup.csv")
r10 = RT[(RT["window"] == "days -10..-2") & (RT["measure"] == "minus placebo")].iloc[0]
assert float(r10["success minus failure (clipped)"].split("%")[0]) > 1.5, r10
sv = pd.read_csv(f"{F}/nb10/survivorship_readouts.csv")
assert sv.loc[sv["readouts"] == "all", "success share"].iloc[0] < sv.loc[sv["readouts"] == "with prices (Table 4)", "success share"].iloc[0]
IM = pd.read_csv(f"{F}/nb10/implied_sensitivity.csv")
assert IM["returns"].str.contains("clip 1%").any() and len(IM) >= 10
assert "after the registry" in out and "restructuring" in out
ET = pd.read_csv(f"{F}/nb10/exits_robust.csv")
assert {"Firth OR", "LPM, each firm once"} <= set(ET.columns)
assert "Breusch-Pagan" in out and "variance inflation" in out
SM = pd.read_csv(f"{F}/nb10/summary.csv")
assert SM["Holm p (within family)"].notna().sum() >= 3, SM
rc = pd.read_csv(f"{F}/nb10/recall_check.csv")
assert len(rc) == 50 and rc["search_links"].str.contains("edgar/search").all()
for f in ["nulls_ci.csv", "diagnostics.csv", "extreme_moves.csv"]:
    assert os.path.exists(f"{F}/nb10/{f}"), f
for f in ["runup_by_channel.csv", "implied_bhar.csv", "insiders_ten_pct_gifts.csv", "friday_release_date.csv"]:
    assert os.path.exists(f"{F}/nb10/{f}"), f
BHt = pd.read_csv(f"{F}/nb10/implied_bhar.csv")
assert BHt["gap"].notna().all(), BHt
assert re.search(r"correlation with CAR (0\.9\d\d|1\.000)", out), "buy-and-hold should track the CAR closely"
assert "without releases 2-7 days early" in out
assert "reliable matches only" in out and "Section 6, registry posting date" in out
for k in ["(a) AACT monthly snapshots: 3 files, first 2018-01, last 2026-01", "(b) Implied probability", "price at day -11",
          "(c) Press-release dateline", "release no more than 1 day before", "(d) Never-disclosed share", "8-K filers only",
          "(e) Small sponsor", "size counted at the stop date", "(f) Registry posting date", "(g) Figures", "paper_figures.zip"]:
    assert k in out, k
import zipfile
assert len(zipfile.ZipFile(f"{F}/nb10/paper_figures.zip").namelist()) == 3
for f_ in ["fig_readout_paths", "fig_implied", "fig_reasons", "fig_drift", "fig_exits", "fig_km", "fig_insiders", "fig_friday", "fig_llm_bins"]:
    assert os.path.exists(f"{F}/figures/paper/{f_}.png"), f_
RB = pd.read_csv(f"{F}/nb10/robust_runup.csv")
print(RB.to_string())
g = dict(zip(RB["check"], RB["gap"]))
assert g["market-adjusted, net of placebo dates (Step 3)"] > 0.03, g
assert g["beta-adjusted, net of placebo dates"] > 0.03 and g["market model, alpha and beta from days -250..-30"] > 0.03
assert abs(g["days -30..-11, net of placebo dates"]) < 0.06, g["days -30..-11, net of placebo dates"]
assert all(g[f"seed {2026 + s}, 5 placebo dates"] > 0.03 for s in range(1, 6))
assert "Run-up gap, clustered by firm and calendar week" in g and "Run-up gap, bootstrap resampling calendar months" in g
assert "run-up gap, tone and language model agree" in g and "linked readouts" in g
assert any(k.startswith("Failure exit within 3 years") for k in g)
for k in ["(e) Splits", "difference in the gap by size", "readouts citing at least one NCT number", "competing exits by type",
          "Implied-realized gap".lower()[:8]]:
    assert k in out, k
assert os.path.exists(f"{F}/figures/paper/fig_runup30.png") and os.path.exists(f"{F}/figures/paper/fig_cif.png")
assert len(zipfile.ZipFile(f"{F}/nb10/robust_figures.zip").namelist()) == 2
R9 = pd.read_csv(f"{F}/nb10/runup_20_placebos.csv")
g9 = dict(zip(R9["check"], R9["gap"]))
print(R9[["check", "gap", "lo", "hi", "p"]].to_string())
assert g9["Headline: days -10..-2, without releases 2-7 days before the 8-K"] > 0.03
draws = [v for k, v in g9.items() if k.startswith("draw ")]
assert len(draws) == 5 and max(draws) - min(draws) < 0.02, draws
assert abs(g9["Earlier window, days -30..-11, headline sample"]) < 0.06
for k in ["Tone and language model agree: implied", "runup20_figures.zip"]:   # Holm now runs once, in Step 19c
    assert k in out, k
assert len(zipfile.ZipFile(f"{F}/nb10/runup20_figures.zip").namelist()) == 2
print("ALL TESTS PASSED")

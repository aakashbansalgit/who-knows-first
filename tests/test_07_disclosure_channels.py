"""Offline test of Notebook 7: synthetic stops with planted drift after quiet channels, channel choice driven by reason,
a fat right tail in readouts, cached prices with beta 1.3."""
import io, os, sys, shutil, contextlib, datetime as dt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "tools"))
from cells import load_stage
C = load_stage("07_disclosure_channels")

P = "/tmp/nb7_test"
shutil.rmtree(P, ignore_errors=True)
F, PR, CT = f"{P}/features", f"{P}/prices", f"{P}/ctgov"
for d in (F, PR, CT, f"{F}/nb6"):
    os.makedirs(d)
rng = np.random.default_rng(7)
TODAY = pd.Timestamp(dt.date.today())
days = pd.bdate_range("2011-01-01", TODAY - pd.Timedelta(days=1))
N = len(days)
T = [f"T{i}" for i in range(60)]
ciks = {t: 1000 + i for i, t in enumerate(T)}
drift = {t: np.zeros(N) for t in T}

def positions(k=9):
    return [300 + j * 330 + int(rng.integers(0, 40)) for j in range(k)]

# ---- registry-date stop events ----
S, sid = [], 0
sets = [("never", T[0:10]), ("later", T[10:15]), ("first", T[15:25])]
for kind, ts in sets:
    for t in ts:
        for p in positions():
            if p + 130 >= N:
                continue
            if kind == "never":
                drift[t][p + 2:p + 62] += -0.002                      # about -12% over days +2..+61
            S.append({"event_id": f"{ciks[t]}_{days[p]:%Y%m%d}", "cik": ciks[t], "ticker": t, "registry_date": days[p],
                      "group": "filing first" if kind == "first" else "registry first or only",
                      "registry_detail": {"never": "never disclosed (within a year)", "later": "disclosed after the registry",
                                          "first": ""}[kind],
                      "never_disclosed": kind == "never", "late_stage": bool(rng.random() < 0.4),
                      "material_reason": bool(rng.random() < 0.3), "reason": "business", "firm_lead_trials": int(rng.integers(2, 40)),
                      "same_day_8k": False, "near_readout": False, "n_trials": 1, "log_dollar_vol": 14.0,
                      "day0": days[p], "car01": 0.0})
pd.DataFrame(S).to_parquet(f"{F}/stop_events.parquet", index=False)

# ---- first-disclosure events ----
D = []
for chan, ts, dr in [("8-K", T[25:35], 0.0), ("10-Q/10-K/20-F/40-F", T[35:45], -0.0015), ("prospectus", T[45:55], 0.0)]:
    for t in ts:
        for p in positions():
            if p + 130 >= N:
                continue
            drift[t][p + 2:p + 62] += dr                              # 10-Q channel: about -9%
            D.append({"event_id": f"d{ciks[t]}_{days[p]:%Y%m%d}", "cik": ciks[t], "ticker": t, "first_date": days[p], "channel": chan,
                      "when": "before the registry", "late_stage": False, "material_reason": bool(rng.random() < 0.5),
                      "firm_lead_trials": 10, "near_readout": False, "n_trials": 1, "car_1_1": 0.0, "day0": days[p]})
pd.DataFrame(D).to_parquet(f"{F}/stop_disclosure_events.parquet", index=False)

# ---- prices: stock = 1.3 x XBI + noise + planted drift ----
rb = rng.normal(0.0003, 0.012, N)
pd.DataFrame({"date": days, "Close": 50 * np.exp(np.cumsum(np.log1p(rb))), "Volume": 1e6}).to_parquet(f"{PR}/XBI.parquet", index=False)
for t in T:
    r = 1.3 * rb + rng.normal(-0.0002, 0.02, N) + drift[t]
    pd.DataFrame({"date": days, "Close": 20 * np.exp(np.cumsum(np.log1p(r))), "Volume": 1e5}).to_parquet(f"{PR}/{t}.parquet", index=False)

# ---- per-stop disclosure file and hits: safety/efficacy stops go to 8-Ks more often ----
sd, hits = [], []
WHY = {"safety": "safety concerns", "efficacy": "futility at interim analysis", "business": "business decision",
       "enrollment": "slow enrollment"}
for i in range(700):
    t = T[i % 40]
    reason = rng.choice(list(WHY))
    mat = reason in ("safety", "efficacy")
    u = rng.random()
    ch = ("8-K" if u < 0.65 else "10-Q/10-K/20-F/40-F" if u < 0.8 else "none") if mat else \
         ("8-K" if u < 0.2 else "10-Q/10-K/20-F/40-F" if u < 0.5 else "none")
    days_first = np.nan if ch == "none" else float(rng.integers(-300, 200))
    timing = "none" if ch == "none" else ("before" if days_first < 0 else "0-30 days" if days_first <= 30 else "31-365 days")
    rd = days[int(rng.integers(300, N - 200))]
    sd.append({"stop_id": i, "nct_id": f"NCT{i:08d}", "cik": ciks[t], "ticker": t, "firm": f"Firm {t}", "registry_date": rd,
               "in_sample": i % 13 != 0, "why_stopped": WHY[reason], "phase": "PHASE3" if i % 3 == 0 else "PHASE2",
               "firm_lead_trials": int(rng.integers(1, 60)), "firm_8ks_year": 0 if i % 17 == 0 else 5, "first_channel": ch,
               "timing": timing, "days_first": days_first, "group": "8-K filer, 1-20 trials"})
    if ch != "none":
        for k in range(int(rng.integers(1, 4))):
            hits.append({"stop_id": i, "match_kind": rng.choice(["drug_name", "acronym", "nct"], p=[0.7, 0.28, 0.02]),
                         "name_trials": int(rng.choice([1, 2, 8])), "form": "8-K" if ch == "8-K" else "10-Q",
                         "filing_date": rd + pd.Timedelta(days=days_first + k), "channel": ch, "days": days_first + k,
                         "stop_hit": True, "snippet": "The company discontinued the trial.", "matched": "drugx",
                         "accession": f"0000{i:06d}-20-{k:06d}"})
pd.DataFrame(sd).to_parquet(f"{F}/stop_disclosure.parquet", index=False)
pd.DataFrame(hits).to_parquet(f"{F}/stop_disclosure_hits.parquet", index=False)

# ---- readouts: positives with a fat right tail ----
ro = []
for i in range(1500):
    t = T[i % 40]
    p = int(rng.integers(300, N - 30))
    kind = "topline_or_endpoint" if i % 4 else "data_update"
    tone = rng.choice(["positive", "negative", "mixed", "unclear"], p=[0.6, 0.15, 0.13, 0.12])
    car = {"positive": 0.03, "negative": -0.35, "mixed": -0.09, "unclear": 0.0}[tone] + rng.normal(0, 0.06)
    if tone == "positive" and rng.random() < 0.03:
        car += rng.uniform(2, 5)                                       # rare huge winners
    acc = pd.Timestamp(days[p]) + pd.Timedelta(hours=int(rng.choice([7, 17])))
    ro.append({"accession": f"A{i:05d}", "cik": ciks[t], "ticker": t, "event_date": days[p], "acceptance_et": acc,
               "timing": "pre_open" if acc.hour < 9 else "after_close", "readout_kind": kind, "readout_tone": tone,
               "has_prices": True, "car_1_1": car, "headline": "Topline results", "readout_snippet": "met its primary endpoint",
               "day0": days[p]})
pd.DataFrame(ro).to_parquet(f"{F}/readout_events.parquet", index=False)

# ---- registry events (pipeline size) and Notebook 6 outputs ----
pd.DataFrame({"cik": list(ciks.values()), "firm_lead_trials": [3 if i < 30 else 30 for i in range(60)]}).to_parquet(
    f"{F}/registry_events.parquet", index=False)
pd.DataFrame({"accession": [f"A{i:05d}" for i in range(400)], "cik": [ciks[T[i % 40]] for i in range(400)],
              "outcome": rng.choice(["success", "failure"], 400, p=[0.8, 0.2]), "has_outcome_edit": rng.integers(0, 2, 400),
              "any_flag": 1, "car": 0.0}).to_parquet(f"{F}/nb6/readout_flags.parquet", index=False)
pd.DataFrame({"cik": [ciks[T[i % 40]] for i in range(200)], "registry_date": [days[int(rng.integers(300, N - 150))] for _ in range(200)],
              "types": rng.choice(["outcome_edit", "pcd_later"], 200), "n": 1, "ticker": [T[i % 40] for i in range(200)],
              "near_readout": False, "day0": pd.NaT, "car_0_1": 0.0, "car_0_5": 0.0}).to_parquet(f"{F}/nb6/flag_events.parquet", index=False)
pd.DataFrame({"nct_id": [f"NCT{i:08d}" for i in range(700)], "phases": "PHASE2"}).to_parquet(f"{CT}/industry_trials_current.parquet", index=False)

CODE = [src for kind, src in C.CELLS if kind == "code"]

def run(overrides):
    ns, buf = {}, io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            for src in CODE:
                exec(compile(src, "<cell>", "exec"), ns)
                if src is C.CONFIG:
                    ns.update(PROJECT_DIR=P, **overrides)
    except Exception:
        print(buf.getvalue())
        raise
    print(buf.getvalue())
    return ns

ns = run({"N_BOOT": 200})
Sx = pd.read_parquet(f"{F}/nb7/drift_registry.parquet")
m = Sx.groupby("sgroup")["xcar_2_60"].mean()
assert m["registry first, never disclosed"] < -0.06, m
assert Sx.groupby("sgroup")["xcar_2_120"].mean()["registry first, never disclosed"] < -0.06
assert Sx["placebo_rule"].eq("strict").mean() > 0.8
assert abs(m["filing first"]) < 0.03 and abs(m["registry first, disclosed later"]) < 0.04, m
assert Sx["beta"].between(1.0, 1.6).mean() > 0.8, Sx["beta"].describe()
pb = Sx.groupby("sgroup")["xbcar_2_60"].mean()
assert pb["registry first, never disclosed"] < -0.06 and abs(pb["filing first"]) < 0.03, pb
assert abs(Sx["xcar_m60_m2"].mean()) < 0.03
Dx = pd.read_parquet(f"{F}/nb7/drift_disclosure.parquet")
md = Dx.groupby("chan")["xcar_2_60"].mean()
assert md["10-Q/10-K"] < -0.05 and abs(md["8-K"]) < 0.035 and abs(md["other filing"]) < 0.035, md
assert set(Dx["chan"]) == {"8-K", "10-Q/10-K", "other filing"}
rd = ns["res_drift"][("registry date", "2_60")]
key = [k for k in rd.params.index if "never" in k][0]
assert rd.params[key] < -0.05 and rd.pvalues[key] < 0.01, rd.summary()
assert ns["res_channel"]["8-K first (disclosed stops)"].params["material"] > 0.5
assert ns["res_channel"]["never disclosed"].params["material"] < 0
cc = pd.read_parquet(f"{F}/nb7/channel_choice.parquet")
assert len(cc) == sum(1 for i in range(700) if i % 13 != 0) and cc["late"].sum() > 0
rc = ns["recon"]
assert rc["raw"] > rc["clipped_topline"] and rc["without_top10"] < rc["raw"], rc
vr = pd.read_csv(f"{F}/nb7/validate_readouts.csv")
vm = pd.read_csv(f"{F}/nb7/validate_matches.csv")
assert 190 <= len(vr) <= 200 and vr["label_tone"].nunique() >= 4, len(vr)
assert 90 <= len(vm) <= 100 and (vm["risk"] == "name shared by 6+ trials").any() and vm["filing_url"].str.startswith("https://www.sec.gov").all()
assert len(ns["FE"]) > 30 and ns["FE"]["conc"].any()
for f in ["fig13_drift", "fig14_channels"]:
    assert os.path.exists(f"{F}/figures/{f}.png"), f
print("ALL TESTS PASSED")

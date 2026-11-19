"""Offline test of Notebook 6: synthetic readouts with a planted success gap, red flags that raise failure risk, cached prices."""
import io, os, sys, shutil, contextlib, datetime as dt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "tools"))
from cells import load_stage
C = load_stage("06_failure_risk")

P = "/tmp/nb6_test"
shutil.rmtree(P, ignore_errors=True)
F, PR, CT, E = f"{P}/features", f"{P}/prices", f"{P}/ctgov", f"{P}/edgar"
for d in (F, PR, CT, E):
    os.makedirs(d)
rng = np.random.default_rng(11)
TODAY = pd.Timestamp(dt.date.today())
days = pd.bdate_range("2014-01-01", TODAY - pd.Timedelta(days=1))
N = len(days)
T = [f"Q{i}" for i in range(30)]
ciks = {t: 500 + i for i, t in enumerate(T)}

# prices: noise only (flag postings have no effect: the market ignores them)
for t in T + ["XBI"]:
    r = rng.normal(0, 0.03 if t != "XBI" else 0.01, N)
    pd.DataFrame({"date": days, "Close": 20 * np.exp(np.cumsum(np.log1p(r))), "Volume": 1e5}).to_parquet(f"{PR}/{t}.parquet", index=False)
pd.DataFrame({"cik": list(ciks.values()) + [999], "tickers": T + ["ZZZ"]}).to_parquet(f"{E}/universe_x.parquet")

# readouts: success with 80% probability (60% if the trial had a red flag), success +14%, failure -35% => implied about 71%
ro, reg, cur = [], [], []
ARCH = pd.Timestamp("2017-01-01")
for i in range(1500):
    t = T[i % 30]
    k = int(rng.integers(300, N - 30))
    day = days[k]
    kind = "topline_or_endpoint" if i % 5 else "data_update"
    nct = f"NCT{i:08d}"
    flag = day > ARCH + pd.Timedelta(days=400) and rng.random() < 0.3
    p_succ = 0.6 if flag else 0.85
    succ = rng.random() < p_succ
    tone = "positive" if succ else "negative"
    if rng.random() < 0.12:
        tone = rng.choice(["mixed", "unclear"])
    car = (0.14 if succ else -0.35) + rng.normal(0, 0.08)
    sent = "The trial met its primary endpoint with a statistically significant improvement." if succ else \
           "The trial did not meet its primary endpoint."
    ro.append({"accession": f"A{i:05d}", "cik": ciks[t], "ticker": t, "event_date": day, "day0": day, "readout_kind": kind,
               "readout_tone": tone, "timing": "pre_open", "has_prices": True, "car_1_1": car, "abs_car": abs(car),
               "car_m10_m2": rng.normal(0, 0.03), "pre_vol": 0.04, "log_dollar_vol": rng.normal(15, 1), "phase": [3.0, 2.0, np.nan][i % 3],
               "same_day_accessions": f"A{i:05d}", "nct_ids": nct if i % 7 else None, "clean": True, "year": day.year,
               "headline": "Company announces topline results", "readout_snippet": sent})
    cur.append({"nct_id": nct})
    if flag:
        fd = day - pd.Timedelta(days=int(rng.integers(30, 300)))
        et = rng.choice(["pcd_later", "enrollment_down", "outcome_edit"])
        reg.append({"nct_id": nct, "cik": float(ciks[t]), "registry_date": fd, "snap": fd + pd.Timedelta(days=10), "event_type": et,
                    "date_source": "posted", "stale_record": False, "owner_source": "sponsor_at_time", "firm_8ks_year": 10,
                    "firm_lead_trials": 5, "same_day_8k": False, "is_stop": False})
    if rng.random() < 0.1 and day > ARCH + pd.Timedelta(days=400):          # contrast events, and an old (stale) red flag
        reg.append({"nct_id": nct, "cik": float(ciks[t]), "registry_date": day - pd.Timedelta(days=60), "snap": day,
                    "event_type": "enrollment_up", "date_source": "posted", "stale_record": False, "owner_source": "sponsor_at_time",
                    "firm_8ks_year": 10, "firm_lead_trials": 5, "same_day_8k": False, "is_stop": False})
        reg.append({"nct_id": nct, "cik": float(ciks[t]), "registry_date": day - pd.Timedelta(days=90), "snap": day,
                    "event_type": "pcd_later", "date_source": "posted", "stale_record": True, "owner_source": "sponsor_at_time",
                    "firm_8ks_year": 10, "firm_lead_trials": 5, "same_day_8k": False, "is_stop": False})
reg.append({"nct_id": "NCT99999999", "cik": np.nan, "registry_date": ARCH, "snap": ARCH, "event_type": "status",
            "date_source": "snapshot", "stale_record": False, "owner_source": "collaborator", "firm_8ks_year": 0,
            "firm_lead_trials": 0, "same_day_8k": False, "is_stop": False})
pd.DataFrame(ro).to_parquet(f"{F}/readout_events.parquet", index=False)
pd.DataFrame(reg).to_parquet(f"{F}/registry_events.parquet", index=False)
pd.DataFrame(cur).to_parquet(f"{CT}/industry_trials_current.parquet", index=False)

CELLS = [C.CONFIG, C.SETUP, C.LOAD, C.POS, C.FLAGS, C.PRICED, C.FIGS]

def run(overrides):
    ns, buf = {}, io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            for src in CELLS:
                exec(compile(src, "<cell>", "exec"), ns)
                if src is C.CONFIG:
                    ns.update(PROJECT_DIR=P, **overrides)
    except Exception:
        print(buf.getvalue())
        raise
    print(buf.getvalue())
    return ns

ns = run({"N_BOOT": 200})
pos = pd.read_csv(f"{F}/nb6/implied_pos.csv")
a = pos[(pos["group"] == "all") & (pos["window"] == "days -1..+1") & ~pos["mixed_as_failure"]].iloc[0]
assert 0.60 < a["implied"] < 0.80 and a["realized"] > a["implied"] and a["gap_lo"] > 0, a.to_dict()
fl = pd.read_parquet(f"{F}/nb6/readout_flags.parquet")
b = fl[fl["outcome"].isin(["success", "failure"])]
r1 = (b.loc[b["any_flag"] == 1, "outcome"] == "failure").mean()
r0 = (b.loc[b["any_flag"] == 0, "outcome"] == "failure").mean()
assert r1 > r0 + 0.1, (r0, r1)
assert fl["has_enrollment_up"].sum() > 0 and "any red flag" in ns["res_logit"] and ns["res_logit"]["any red flag"].params["any_flag"] > 0
assert (fl["event_date"] >= pd.Timestamp("2017-01-01") + pd.Timedelta(days=365)).all()
fe = pd.read_parquet(f"{F}/nb6/flag_events.parquet")
q = fe[fe["car_0_1"].notna()]
assert len(q) > 50 and abs(q["car_0_1"].mean()) < 0.02, (len(q), q["car_0_1"].mean())
for f in ["fig11_implied_pos", "fig12_flags"]:
    assert os.path.exists(f"{F}/figures/{f}.png"), f
print("ALL TESTS PASSED")

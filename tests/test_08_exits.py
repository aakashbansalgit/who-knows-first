"""Offline test of Notebook 8: cached synthetic filing histories with planted exits (bankruptcy, acquisition, delisting after
notices, stopped filing, still filing), never-disclosed stops concentrated at failing firms."""
import io, os, sys, shutil, contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "tools"))
from cells import load_stage
C = load_stage("08_exits")

P = "/tmp/nb8_test"
shutil.rmtree(P, ignore_errors=True)
F, E = f"{P}/features", f"{P}/edgar"
H = f"{F}/nb8/history"
for d in (F, E, f"{F}/nb7", H):
    os.makedirs(d)
rng = np.random.default_rng(8)
TODAY = pd.Timestamp.today().normalize()
kinds = ["listed"] * 60 + ["bankruptcy"] * 15 + ["acquired"] * 15 + ["notices"] * 10 + ["stopped"] * 10 + ["still"] * 10
ciks = list(range(5000, 5000 + len(kinds)))
kind_of = dict(zip(ciks, kinds))
exit_of = {}
for c, k in kind_of.items():
    base = pd.bdate_range("2017-01-01", TODAY - pd.Timedelta(days=30), freq="QS")
    rows = [{"form": "10-Q", "filingDate": d, "accessionNumber": f"{c}-{i}", "items": ""} for i, d in enumerate(base)]
    if k in ("bankruptcy", "acquired", "notices", "stopped"):
        ed = pd.Timestamp("2020-06-01") + pd.Timedelta(days=int(rng.integers(0, 900)))
        exit_of[c] = ed
        rows = [r for r in rows if r["filingDate"] < ed]
        if k == "bankruptcy":
            rows += [{"form": "8-K", "filingDate": ed - pd.Timedelta(days=40), "accessionNumber": f"{c}-b", "items": "1.03,9.01"},
                     {"form": "25-NSE", "filingDate": ed, "accessionNumber": f"{c}-25", "items": ""},
                     {"form": "15-12B", "filingDate": ed + pd.Timedelta(days=10), "accessionNumber": f"{c}-15", "items": ""}]
        elif k == "acquired":
            rows += [{"form": "DEFM14A", "filingDate": ed - pd.Timedelta(days=60), "accessionNumber": f"{c}-m", "items": ""},
                     {"form": "8-K", "filingDate": ed, "accessionNumber": f"{c}-c", "items": "2.01,3.01,5.01,9.01"},
                     {"form": "25-NSE", "filingDate": ed, "accessionNumber": f"{c}-25", "items": ""}]
        elif k == "notices":
            rows += [{"form": "8-K", "filingDate": ed - pd.Timedelta(days=200), "accessionNumber": f"{c}-n1", "items": "3.01"},
                     {"form": "8-K", "filingDate": ed - pd.Timedelta(days=20), "accessionNumber": f"{c}-n2", "items": "3.01,9.01"},
                     {"form": "25-NSE", "filingDate": ed, "accessionNumber": f"{c}-25", "items": ""}]
        # "stopped": nothing after ed
    if k == "listed" and c % 4 == 0:                                   # some listed firms get a notice
        rows.append({"form": "8-K", "filingDate": pd.Timestamp("2021-03-01"), "accessionNumber": f"{c}-n", "items": "3.01"})
    pd.DataFrame(rows).assign(filingDate=lambda d: pd.to_datetime(d["filingDate"])).to_parquet(f"{H}/{c}.parquet", index=False)
pd.DataFrame({"cik": [c for c in ciks if kind_of[c] == "listed"], "tickers": [f"L{c}" for c in ciks if kind_of[c] == "listed"]}).to_parquet(
    f"{E}/universe_x.parquet", index=False)

def ev_date(c):
    if c in exit_of:
        return exit_of[c] - pd.Timedelta(days=int(rng.integers(60, 300)))
    return pd.Timestamp("2019-01-01") + pd.Timedelta(days=int(rng.integers(0, 1500)))

se, dse, cc = [], [], []
fail_kinds = {"bankruptcy", "notices", "stopped"}
for i in range(600):
    c = ciks[int(rng.integers(0, len(ciks)))]
    k = kind_of[c]
    if k in fail_kinds:
        g = rng.choice(["never", "first", "later"], p=[0.7, 0.2, 0.1])
    else:
        g = rng.choice(["never", "first", "later"], p=[0.3, 0.6, 0.1])
    d = ev_date(c)
    se.append({"event_id": f"{c}_{i}", "cik": c, "ticker": f"L{c}" if k == "listed" else None, "registry_date": d,
               "group": "filing first" if g == "first" else "registry first or only",
               "registry_detail": {"never": "never disclosed (within a year)", "later": "disclosed after the registry", "first": ""}[g],
               "firm_lead_trials": int(rng.integers(1, 50)), "material_reason": bool(rng.random() < 0.3), "late_stage": bool(rng.random() < 0.3)})
    ch = "10-Q/10-K/20-F/40-F" if g == "never" else "8-K"
    dse.append({"event_id": f"d{c}_{i}", "cik": c, "ticker": f"L{c}" if k == "listed" else None, "first_date": d, "channel": ch,
                "firm_lead_trials": 10, "material_reason": False, "late_stage": False})
    cc.append({"stop_id": i, "nct_id": f"NCT{i:08d}", "cik": c, "registry_date": d,
               "outcome": {"never": "never", "first": "8-K first", "later": "10-Q/10-K first"}[g],
               "material": int(rng.random() < 0.3), "late": int(rng.random() < 0.3), "small": int(rng.random() < 0.6)})
se, dse, cc = pd.DataFrame(se), pd.DataFrame(dse), pd.DataFrame(cc)
se.to_parquet(f"{F}/stop_events.parquet", index=False)
dse.to_parquet(f"{F}/stop_disclosure_events.parquet", index=False)
cc.to_parquet(f"{F}/nb7/channel_choice.parquet", index=False)
sg = np.select([se["group"].eq("filing first"), se["registry_detail"].str.startswith("never")], ["filing first", "registry first, never disclosed"],
               "registry first, disclosed later")
lst = se["ticker"].notna()
pd.DataFrame({"event_id": se.loc[lst, "event_id"], "sgroup": sg[lst], "xcar_2_60": rng.normal(0.05, 0.2, lst.sum()),
              "xcar_2_120": rng.normal(0.08, 0.3, lst.sum())}).to_parquet(f"{F}/nb7/drift_registry.parquet", index=False)
chn = np.where(dse["channel"].str.contains("8-K"), "8-K", "10-Q/10-K")
lst2 = dse["ticker"].notna()
pd.DataFrame({"event_id": dse.loc[lst2, "event_id"], "chan": chn[lst2], "xcar_2_60": rng.normal(0.05, 0.2, lst2.sum()),
              "xcar_2_120": rng.normal(0.08, 0.3, lst2.sum())}).to_parquet(f"{F}/nb7/drift_disclosure.parquet", index=False)

CODE = [src for kind, src in C.CELLS if kind == "code"]
ns, buf = {}, io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        for src in CODE:
            exec(compile(src, "<cell>", "exec"), ns)
            if src is C.CONFIG:
                ns.update(PROJECT_DIR=P, USER_AGENT="Test Person test@example.com")
except Exception:
    print(buf.getvalue())
    raise
print(buf.getvalue())

assert ns["SEC"].n_requests == 0
ex = pd.read_parquet(f"{F}/nb8/firm_exits.parquet").set_index("cik")
want = {"listed": "listed today", "bankruptcy": "bankruptcy", "acquired": "acquired", "notices": "delisted after listing-rule notices",
        "stopped": "stopped filing", "still": "still filing, no listed ticker"}
bad = [(c, k, ex.loc[c, "exit_type"]) for c, k in kind_of.items() if c in ex.index and ex.loc[c, "exit_type"] != want[k]]
assert len(ex) > 100
assert not bad, bad[:10]
for c, ed in exit_of.items():
    if kind_of[c] != "stopped" and c in ex.index:
        assert ex.loc[c, "exit_date"] == ed, (c, ex.loc[c, "exit_date"], ed)
R = pd.read_parquet(f"{F}/nb8/exit_outcomes_registry.parquet")
f = R.groupby("sgroup")["fail_365"].mean()
assert f["registry first, never disclosed"] > f["filing first"] + 0.1, f
assert R["notice_365"].notna().any() and R["acq_365"].max() == 1
Pq = pd.read_parquet(f"{F}/nb8/exit_outcomes_stops.parquet")
assert Pq.groupby("outcome")["fail_730"].mean()["never"] > Pq.groupby("outcome")["fail_730"].mean()["8-K first"]
m = ns["res_logit"][("registry date", "fail_365")]
key = [k for k in m.params.index if "never" in k][0]
assert m.params[key] > 0, m.params
b = pd.read_csv(f"{F}/nb8/drift_bounds.csv")
row = b[(b["events"] == "registry date") & (b["window"] == "2_120") & (b["fail_return"] == -0.55) & (b["acq_return"] == 0.0)].iloc[0]
assert row["registry first, never disclosed: mean"] < row["registry first, never disclosed: survivors only"], row
assert row["registry first, never disclosed: failures in window"] > 0
assert os.path.exists(f"{F}/figures/fig15_exits.png")
print("ALL TESTS PASSED")

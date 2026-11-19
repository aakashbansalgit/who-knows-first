"""Offline test of Notebook 9: synthetic SEC insider data-set zips (with header rows and DD-MON-YYYY dates), insiders who sell
before failures until the 2023 reform, and Friday bad news with a smaller immediate reaction and more drift."""
import io, os, sys, shutil, zipfile, contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "tools"))
from cells import load_stage
C = load_stage("09_insiders")

P, Z = "/tmp/nb9_test", "/tmp/nb9_zips"
for d in (P, Z):
    shutil.rmtree(d, ignore_errors=True)
F = f"{P}/features"
os.makedirs(F)
os.makedirs(Z)
rng = np.random.default_rng(9)
ciks = list(range(7000, 7060))
fpi = set(ciks[-5:])                                            # foreign filers: no Form 4s
days = pd.bdate_range("2018-01-01", "2025-06-30")

# readouts
ro = []
for i in range(1800):
    c = ciks[i % 60]
    d = days[int(rng.integers(150, len(days) - 40))]
    oc = rng.choice(["positive", "negative", "mixed"], p=[0.7, 0.15, 0.15])
    bad = oc != "positive"
    fri = rng.random() < (0.2 if bad else 0.08)
    acc_day = d - pd.Timedelta(days=int(d.dayofweek) - 4) if fri else d - pd.Timedelta(days=int(d.dayofweek) % 4)
    imm = {"positive": 0.10, "negative": -0.35, "mixed": -0.10}[oc] * (0.6 if (bad and fri) else 1.0) + rng.normal(0, 0.05)
    drift = (-0.08 if (bad and fri) else 0.0) + rng.normal(0, 0.04)
    ro.append({"accession": f"R{i:05d}", "cik": c, "ticker": f"T{c}", "event_date": d, "day0": d,
               "acceptance_et": acc_day + pd.Timedelta(hours=7), "readout_kind": "topline_or_endpoint" if i % 6 else "data_update",
               "readout_tone": oc, "has_prices": True, "car_1_1": imm, "car_2_20": drift, "car_m10_m2": rng.normal(0, 0.03),
               "log_dollar_vol": rng.normal(15, 1), "abn_volume": float(np.exp(rng.normal(1.0 if not fri else 0.7, 0.3)))})
ro = pd.DataFrame(ro)
ro.to_parquet(f"{F}/readout_events.parquet", index=False)
pd.DataFrame({"cik": [7001, 7002], "first_date": [days[300], days[600]], "channel": ["8-K", "10-Q/10-K/20-F/40-F"],
              "car_1_1": [-0.1, -0.02], "car_2_20": [0.0, 0.01], "day0": [days[300], days[600]]}).to_parquet(
    f"{F}/stop_disclosure_events.parquet", index=False)

# insider transactions
rows, k = [], 0
def add(c, d, code, plan, rel="Officer", owner=None):
    global k
    k += 1
    rows.append({"acc": f"0000{k:08d}", "cik": c, "d": d, "code": code, "plan": plan, "rel": rel, "owner": owner or f"9{c}{k % 3}"})
for c in ciks:
    if c in fpi:
        continue
    for d in pd.date_range("2016-01-01", "2025-06-30", freq="45D"):              # routine selling
        if rng.random() < 0.3:
            add(c, d + pd.Timedelta(days=int(rng.integers(0, 40))), "S", rng.random() < 0.4)
    add(c, pd.Timestamp("2019-03-01"), "A", False)
for r in ro[ro["readout_kind"] == "topline_or_endpoint"].itertuples():
    if r.cik in fpi:
        continue
    post = r.event_date >= pd.Timestamp("2023-06-01")
    p_sell = 0.2 if post else (0.65 if r.readout_tone == "negative" else 0.15)
    if rng.random() < p_sell:
        add(r.cik, r.event_date - pd.Timedelta(days=int(rng.integers(5, 55))), "S", bool(post))
    if r.readout_tone == "positive" and rng.random() < 0.15:
        add(r.cik, r.event_date - pd.Timedelta(days=int(rng.integers(5, 55))), "P", False, "Director")
    if rng.random() < 0.05:
        add(r.cik, r.event_date - pd.Timedelta(days=20), "S", False, "TenPercentOwner")
X = pd.DataFrame(rows)
X["q"] = X["d"].dt.year.astype(str) + "q" + X["d"].dt.quarter.astype(str)
fmt = lambda d: d.strftime("%d-%b-%Y").upper()
for q in C.__dict__ and []:
    pass
allq = [f"{y}q{n}" for y in range(2016, 2026) for n in range(1, 5)]
for q in allq:
    x = X[X["q"] == q]
    sub = pd.DataFrame({"ACCESSION_NUMBER": x["acc"], "FILING_DATE": [fmt(d + pd.Timedelta(days=2)) for d in x["d"]],
                        "PERIOD_OF_REPORT": [fmt(d) for d in x["d"]], "DOCUMENT_TYPE": "4", "ISSUERCIK": x["cik"].map(lambda v: f"{v:010d}"),
                        "ISSUERNAME": "Co", "ISSUERTRADINGSYMBOL": "T", "REMARKS": "",
                        "AFF10B5ONE": np.where(x["plan"] & (x["d"] >= "2023-04-01"), "1", "0")})
    own = pd.DataFrame({"ACCESSION_NUMBER": x["acc"], "RPTOWNERCIK": x["owner"], "RPTOWNERNAME": "Person",
                        "RPTOWNER_RELATIONSHIP": x["rel"], "RPTOWNER_TITLE": "CEO"})
    tr = pd.DataFrame({"ACCESSION_NUMBER": x["acc"], "NONDERIV_TRANS_SK": range(len(x)), "SECURITY_TITLE": "Common Stock",
                       "TRANS_DATE": [fmt(d) for d in x["d"]], "TRANS_CODE": x["code"], "TRANS_SHARES": "1000",
                       "TRANS_PRICEPERSHARE": "12.5", "TRANS_ACQUIRED_DISP_CD": np.where(x["code"] == "S", "D", "A")})
    fn = pd.DataFrame({"ACCESSION_NUMBER": x.loc[x["plan"] & (x["d"] < "2023-04-01"), "acc"], "FOOTNOTE_ID": "F1",
                       "FOOTNOTE_TXT": "Sold under a Rule 10b5-1 trading plan adopted by the reporting person."})
    # another firm outside the sample, to be filtered out
    extra = pd.DataFrame({"ACCESSION_NUMBER": ["X" + q], "FILING_DATE": ["01-JAN-2020"], "PERIOD_OF_REPORT": ["01-JAN-2020"],
                          "DOCUMENT_TYPE": ["4"], "ISSUERCIK": ["0000000001"], "ISSUERNAME": ["Other"], "ISSUERTRADINGSYMBOL": ["O"],
                          "REMARKS": [""], "AFF10B5ONE": ["0"]})
    sub = pd.concat([sub, extra])
    with zipfile.ZipFile(f"{Z}/{q}_form345.zip", "w") as zf:
        for name, df in [("SUBMISSION", sub), ("REPORTINGOWNER", own), ("NONDERIV_TRANS", tr), ("FOOTNOTES", fn)]:
            zf.writestr(f"{name}.tsv", df.to_csv(sep="\t", index=False))

CODE = [src for kind, src in C.CELLS if kind == "code"]
ns, buf = {}, io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        for src in CODE:
            exec(compile(src, "<cell>", "exec"), ns)
            if src is C.CONFIG:
                ns.update(PROJECT_DIR=P, USER_AGENT="Test Person test@example.com", ZIP_DIR=Z, FIRST_Q="2016q1", LAST_Q="2025q4",
                          KEEP_ZIPS=True)
except Exception:
    print(buf.getvalue())
    raise
print(buf.getvalue())
assert ns["SEC"].n_requests == 0
T = pd.read_parquet(f"{F}/nb9/insider_trades.parquet")
assert set(T["cik"]) <= set(ciks) and not (set(T["cik"]) & fpi) and len(T) > 500
assert T["trans_date"].notna().all() and T["value"].iloc[0] == 12500
s_ = T[T["code"] == "S"]
assert s_["plan"].any() and (~s_["plan"]).any()
E = pd.read_parquet(f"{F}/nb9/readout_insiders.parquet")
assert not (set(E["cik"]) & fpi)
pre = E[E["post"] == 0]
g = pre.groupby("outcome")["sell_m60_m1"].mean()
assert g["failure"] > g["success"] + 0.25, g
m = ns["res"]["sell_m60_m1"]
assert m.params["fail"] > 0.8 and m.pvalues["fail"] < 0.01, m.summary()
d = ns["res_did"]["sell_m60_m1"]
assert d.params["fail:post"] < -0.2 and d.params["fail"] > 0.2, d.params
assert ns["res"]["buy_m60_m1"].params["fail"] < 0
r1, r2 = ns["res_fri"]["car_1_1"], ns["res_fri"]["car_2_20"]
assert r1.params["bad:fri"] > 0.03 and r1.pvalues["bad:fri"] < 0.01, r1.params
assert r2.params["bad:fri"] < -0.04 and r2.pvalues["bad:fri"] < 0.01, r2.params
assert ns["res_fri"]["choice"].params["bad"] > 0.3
assert ns["res_fri"]["log_abn_volume"].params["fri"] < 0
for f in ["fig16_insiders", "fig17_friday"]:
    assert os.path.exists(f"{F}/figures/{f}.png"), f
print("ALL TESTS PASSED")

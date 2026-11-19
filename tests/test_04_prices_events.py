"""Offline test of Notebook 4 v2: synthetic prices with planted moves and data problems, fake Yahoo, fake OCC."""
import io, os, sys, json, types, shutil, contextlib, datetime as dt, time as _time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "tools"))
from cells import load_stage
C = load_stage("04_prices_events")

P = "/tmp/nb4_test"
shutil.rmtree(P, ignore_errors=True)
E, F, PR, O = f"{P}/edgar", f"{P}/features", f"{P}/prices", f"{P}/occ"
for d in (E, F, PR, O):
    os.makedirs(d)
rng = np.random.default_rng(7)
TODAY = pd.Timestamp(dt.date.today())
days = pd.bdate_range("2014-01-01", TODAY - pd.Timedelta(days=1))
N = len(days)

T = [f"T{i}" for i in range(1, 9)]
ciks = {t: 10 + i for i, t in enumerate(T, 1)}
uni = [{"cik": c, "name": f"{t} BIO", "tickers": (f"{t}W;{t}" if t == "T1" else t), "former_names": ""} for t, c in ciks.items()]
uni += [{"cik": 3, "name": "GONE BIO", "tickers": None, "former_names": ""},
        {"cik": 4, "name": "DEAD BIO", "tickers": "DEAD", "former_names": ""}]
pd.DataFrame(uni).to_parquet(f"{E}/universe_x.parquet")
shocks = {t: {} for t in T}                                  # ticker -> {day: planted abnormal return}

def plant(t, d, x):
    if d not in shocks[t]:
        shocks[t][d] = x
        return True
    return False

# ---- readouts ----
ro = []
for i in range(160):
    t = T[i % 6]                                             # T1-T6; T7/T8 carry the special cases
    tone = ["positive", "negative", "mixed", "unclear"][i % 4]
    kind = "topline_or_endpoint" if i % 3 else "data_update"
    timing = ["pre_open", "after_close"][(i // 4) % 2]
    for _ in range(50):
        k = int(rng.integers(300, N - 60))
        if all(abs(k - days.get_loc(x)) > 8 for x in shocks[t]):
            break
    x = {"positive": 0.12, "negative": -0.35}.get(tone, 0.0) if kind == "topline_or_endpoint" else 0.0
    plant(t, days[k - 1] if timing == "after_close" else days[k], x)     # after-close 8-K: the press release moved day -1
    ro.append({"accession": f"R{i}", "cik": ciks[t], "ticker": t, "event_date": days[k], "timing": timing,
               "acceptance_et": days[k], "items": "2.02;9.01" if i % 10 == 0 else "8.01", "readout_kind": kind,
               "readout_tone": tone, "nct_ids": f"NCT9{i:07d};NCT8{i:07d}" if i % 2 else None})

kH, kS, kB, kD = N - 400, N - 380, N - 360, N - 60            # special cases on T7 / T8
ro += [
    {"accession": "RH", "cik": ciks["T7"], "ticker": "T7", "event_date": days[kH], "timing": "pre_open", "items": "8.01",
     "readout_kind": "topline_or_endpoint", "readout_tone": "negative"},              # halted days 0 and +1, drop on +2
    {"accession": "RS", "cik": ciks["T7"], "ticker": "T7", "event_date": days[kS], "timing": "pre_open", "items": "8.01",
     "readout_kind": "topline_or_endpoint", "readout_tone": "positive"},              # no trade on day 0, jump on +1
    {"accession": "RB", "cik": ciks["T7"], "ticker": "T7", "event_date": days[kB], "timing": "pre_open", "items": "8.01",
     "readout_kind": "topline_or_endpoint", "readout_tone": "unclear"},               # bad print (x10) on day 0
    {"accession": "RD1", "cik": ciks["T8"], "ticker": "T8", "event_date": days[kD], "timing": "pre_open", "items": "8.01",
     "readout_kind": "data_update", "readout_tone": "unclear", "acceptance_et": days[kD]},
    {"accession": "RD2", "cik": ciks["T8"], "ticker": "T8", "event_date": days[kD], "timing": "pre_open", "items": "8.01",
     "readout_kind": "topline_or_endpoint", "readout_tone": "positive", "acceptance_et": days[kD] + pd.Timedelta(hours=1)},
    {"accession": "RN", "cik": ciks["T8"], "ticker": "T8", "event_date": None, "timing": "pre_open", "items": "8.01",
     "readout_kind": "topline_or_endpoint", "readout_tone": "positive"},              # no market date: dropped
    {"accession": "RG", "cik": 3, "ticker": None, "event_date": days[400], "timing": "pre_open", "items": "8.01",
     "readout_kind": "topline_or_endpoint", "readout_tone": "positive"},              # no ticker
    {"accession": "RX", "cik": 4, "ticker": "DEAD", "event_date": days[500], "timing": None, "items": None,
     "readout_kind": "topline_or_endpoint", "readout_tone": "negative"}]              # not on Yahoo, missing timing
shocks["T7"][days[kH + 2]] = -0.35
shocks["T7"][days[kS + 1]] = 0.12
shocks["T8"][days[kD]] = 0.12
recent = [days[-40], days[-25], days[-12]]                    # OCC readouts on T3; options volume jumps around day 0
for j, d in enumerate(recent):
    ro.append({"accession": f"O{j + 1}", "cik": ciks["T3"], "ticker": "T3", "event_date": d, "timing": "pre_open",
               "items": "8.01", "readout_kind": "topline_or_endpoint", "readout_tone": "positive"})
    shocks["T3"][d] = 0.12
ro = pd.DataFrame(ro)
ro = pd.concat([ro, ro[ro["accession"] == "R0"]], ignore_index=True)   # a repeated row
ro.to_parquet(f"{F}/readouts.parquet", index=False)

# ---- registry stops on T1/T2 (Notebook 3b timing; registry-first stops drop 15% on the posting day) ----
reg, sdr = [], []
FIRST = {"before": -56, "0-30 days": 14, "none": np.nan}       # calendar days from the registry date to the first filing
for j in range(36):
    t = T[j % 2]
    timing = ["before", "0-30 days", "none"][j % 3]
    for _ in range(200):
        k = int(rng.integers(400, N - 60))
        kf = days.searchsorted(days[k] + pd.Timedelta(days=FIRST[timing])) if timing != "none" else k
        if all(abs(k - days.get_loc(x)) > 8 and abs(kf - days.get_loc(x)) > 8 for x in shocks[t]):
            break
    d = days[k]
    plant(t, d, 0.0 if timing == "before" else -0.15)
    if timing == "before":
        plant(t, days[kf], -0.10)                             # the disclosing filing moves the stock
    reg.append({"nct_id": f"NCT{j:08d}", "cik": float(ciks[t]), "registry_date": d, "is_stop": True, "date_source": "posted",
                "firm_8ks_year": 20, "firm_lead_trials": 8, "disclosed_before": [True, False, None][j % 3],
                "same_day_8k": [False, None, False][j % 3], "event_type": "status", "detail": "RECRUITING -> TERMINATED",
                "phase": ["Phase 3", "PHASE2", "Phase 2/Phase 3", "Early Phase 1"][j % 4]})
    if j != 11:                                               # NCT11 missing from Notebook 3b's output
        sdr.append({"stop_id": j, "nct_id": f"NCT{j:08d}", "registry_date": d.strftime("%Y-%m-%d"), "timing": timing,
                    "in_sample": j not in (5, 7), "exclude_reason": "" if j not in (5, 7) else "collaborator only",
                    "days_first": FIRST[timing], "first_channel": "8-K" if j % 2 else "10-Q/10-K/20-F/40-F",
                    "why_stopped": ["Lack of efficacy at the interim analysis", "Business decision", "Slow enrollment", None][j % 4],
                    "timing_detail": timing})
b0, b1 = reg[0], reg[1]                                       # batches: same firm and posting day
reg.append({**b0, "nct_id": "NCTB0"}); sdr.append({"stop_id": 90, "nct_id": "NCTB0", "registry_date": b0["registry_date"].strftime("%Y-%m-%d"),
                                                  "timing": "none", "in_sample": True, "exclude_reason": "",
                                                  "days_first": np.nan, "first_channel": None, "why_stopped": "safety concern",
                                                  "timing_detail": "none"})
reg.append({**b1, "nct_id": "NCTB1"}); sdr.append({"stop_id": 91, "nct_id": "NCTB1", "registry_date": b1["registry_date"].strftime("%Y-%m-%d"),
                                                  "timing": "0-30 days", "in_sample": True, "exclude_reason": "",
                                                  "days_first": 14, "first_channel": "8-K", "why_stopped": "business",
                                                  "timing_detail": "0-30 days"})
reg.append({**reg[2], "nct_id": "NCTX", "date_source": "snapshot"})          # imprecise date: left out
reg.append({**reg[3], "nct_id": "NCTY", "firm_lead_trials": 500})           # big pharma: left out
reg.append({**reg[4], "nct_id": "NCTZ", "cik": None})                        # no firm: left out
pd.DataFrame(reg).to_parquet(f"{F}/registry_events.parquet", index=False)
os.makedirs(f"{P}/ctgov")
pd.DataFrame({"nct_id": [f"NCT9{i:07d}" for i in range(160)], "phases": ["PHASE2|PHASE3", "PHASE1", "", "PHASE2"] * 40}).to_parquet(
    f"{P}/ctgov/industry_trials_current.parquet", index=False)
pd.DataFrame(sdr).to_parquet(f"{F}/stop_disclosure.parquet", index=False)

# ---- prices: planted moves plus data problems ----
def prices(t):
    r = rng.normal(0, 0.03, N)
    for d, x in shocks.get(t, {}).items():
        r[days.get_loc(d)] += x
    df = pd.DataFrame({"Close": 10 * np.exp(np.cumsum(np.log1p(r))), "Volume": rng.integers(100_000, 200_000, N).astype(float)},
                      index=pd.DatetimeIndex(days, name="Date"))
    if t == "T7":
        df = df.drop(index=[days[kH], days[kH + 1]])                          # halt
        df.loc[days[kS], "Close"] = df.loc[days[kS - 1], "Close"]             # no trade: last close repeated, volume 0
        df.loc[days[kS], "Volume"] = 0.0
        df.loc[days[kB], "Close"] *= 10                                       # bad print
    return df

PRICES = {t: prices(t) for t in T}
PRICES["XBI"] = pd.DataFrame({"Close": 80 * np.exp(np.cumsum(rng.normal(0, 0.01, N))), "Volume": 1e6},
                             index=pd.DatetimeIndex(days, name="Date"))
PRICES["AAPL"] = PRICES["XBI"]

old = (TODAY - pd.Timedelta(days=700)).timestamp()
for t in ("T8", "XBI"):                                       # stale cached files: stop 300 days ago, fetched long ago
    PRICES[t].iloc[:-300].rename_axis("date").reset_index().to_parquet(f"{PR}/{t}.parquet", index=False)
    os.utime(f"{PR}/{t}.parquet", (old, old))
pd.DataFrame(columns=["date", "Close", "Volume"]).to_parquet(f"{PR}/DEAD.parquet", index=False)   # old 'not on Yahoo' marker
os.utime(f"{PR}/DEAD.parquet", (old, old))

# ---- OCC ----
OCC_START = TODAY - pd.Timedelta(days=700)                    # OCC has nothing before this
w0 = days[days.get_loc(recent[0]) - 8]
wz = [days[days.get_loc(recent[0]) - k] for k in (7, 6)]
with open(f"{O}/stock_daily_volume.csv", "w") as f:           # old cache: scratch header, misaligned and unreadable rows
    f.write("ticker,date,call_customer,call_firm,call_market_maker,put_customer,put_firm,put_market_maker,contracts,n_exchanges\n")
    f.write("VRTX,2025-01-15,10,2,8,6,1,3,15.0,9\n")
    f.write(f"T3,{w0.date().isoformat()},100,100,50,125.0\n")
    f.write("T3,2025-02-03,abc,1\n")
with open(f"{O}/stock_daily_done.txt", "w") as f:             # wz were failed requests saved as 'no options'
    f.write("".join(k + "\n" for k in ["VRTX 20250115", f"T3 {w0:%Y%m%d}", "T3 20250203"] + [f"T3 {x:%Y%m%d}" for x in wz]))
FAIL_ONCE, FAILED = {f"T3 {days[days.get_loc(recent[2]) - 3]:%Y%m%d}"}, set()
BOM_DAY = f"{days[days.get_loc(recent[2]) - 2]:%Y%m%d}"

class Fakes:
    def __init__(self, fail_once=(), down=False):
        self.yf_calls, self.occ_calls, self.occ_days = [], 0, []
        self.fail_once, self.down = set(fail_once), down
    def yf_download(self, symbols):
        self.yf_calls.append(list(symbols))
        if self.down:
            return {}
        out = {}
        for s in symbols:
            if s in self.fail_once:
                self.fail_once.discard(s)
                continue
            if s in PRICES:
                out[s] = PRICES[s]
        return out
    def occ_fetch(self, params):
        self.occ_calls += 1
        sym, rd = params["symbol"], params["reportDate"]
        self.occ_days.append(pd.Timestamp(rd))
        key = f"{sym} {rd}"
        if key in FAIL_ONCE and key not in FAILED:
            FAILED.add(key)
            return None
        d = pd.Timestamp(rd)
        if d < OCC_START:
            return "No record(s) found"
        q = 1000 if sym == "T3" and any(0 <= days.get_loc(d) - days.get_loc(e) <= 1 for e in recent if d in days) else 100
        if rd.endswith("5") and sym != "AAPL" and not (sym == "T3" and any(abs((d - e).days) <= 20 for e in recent)):
            return "No record(s) found"
        firm = "" if rd.endswith("3") else f"{q // 2},P,F,Y\n"
        text = f"quantity,porc,actype,exchange\n{q},C,C,X\n{q},C,M,Y\n{q // 2},P,C,X\n" + firm
        return ("﻿" + text).lstrip("﻿ \r\n\t") if rd == BOM_DAY else text

CELLS = [C.CONFIG, C.SETUP, C.LOAD, C.PRICES, C.EVENTS, C.OCC, C.TESTS]
fast_time = types.SimpleNamespace(sleep=lambda s: None, time=_time.time)

def run(overrides, fk, cells=CELLS, expect=None):
    ns, buf = {}, io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            for src in cells:
                exec(compile(src, "<cell>", "exec"), ns)
                if src is C.CONFIG:
                    ns.update(PROJECT_DIR=P, USER_AGENT="Test User test@example.com")
                    ns.update(overrides)
                if src is C.SETUP:
                    ns.update(yf_download=fk.yf_download, occ_fetch=fk.occ_fetch, time=fast_time)
    except Exception as e:
        if expect and isinstance(e, expect):
            print(buf.getvalue()[-600:])
            return ns, fk, e
        print(buf.getvalue())
        raise
    print(buf.getvalue())
    return ns, fk, None

# 1) dry run: old OCC cache repaired, OCC probe, one OCC failure (not saved), DEAD retried
ns, fk, _ = run({"DRY_RUN": True, "DRY_RUN_TICKERS": 3, "DRY_RUN_OCC": 1}, Fakes())
assert ns["tickers"] == ["DEAD", "T1", "T2"], ns["tickers"]
assert os.path.exists(f"{F}/readout_events_dryrun.parquet")
miss = json.load(open(f"{PR}/_missing.json"))
assert miss["DEAD"]["tries"] == 2 and not os.path.exists(f"{PR}/DEAD.parquet"), miss
assert os.path.exists(f"{O}/cache_v2_checked.txt") and any(n.startswith("stock_daily_volume.csv.old_") for n in os.listdir(O))
dv = pd.read_csv(f"{O}/stock_daily_volume.csv").set_index(["ticker", "date"])
assert dv.loc[("VRTX", "2025-01-15"), "contracts"] == 15.0 and dv.loc[("T3", w0.date().isoformat()), "contracts"] == 125.0
oldest = pd.Timestamp(json.load(open(f"{O}/oldest_day.json"))["oldest"])
assert OCC_START <= oldest <= OCC_START + pd.Timedelta(days=12), (oldest, OCC_START)
assert max(fk.occ_days) <= TODAY - pd.Timedelta(days=3), "asked a day OCC may not have published"
print("TEST DRY RUN OK\n" + "=" * 70)

# 2) full run: T4 fails once at Yahoo; a line cut off mid-write sits at the end of the OCC cache
with open(f"{O}/stock_daily_volume.csv", "a") as f:
    f.write("T3,2025-01")
ns, fk, _ = run({"DRY_RUN": False}, Fakes(fail_once={"T4"}))
assert all("T1" not in c and "T2" not in c for c in fk.yf_calls), "dry-run tickers are cached"
miss = json.load(open(f"{PR}/_missing.json"))
assert miss.get("T4", {}).get("tries") == 1 and miss["DEAD"]["tries"] == 2, miss      # not retried the same day
for t in ("T8", "XBI"):
    assert pd.to_datetime(pd.read_parquet(f"{PR}/{t}.parquet")["date"]).max() >= days[-5], f"{t} not refreshed"
re_ = pd.read_parquet(f"{F}/readout_events.parquet").set_index("accession")
assert "RN" not in re_.index and re_.index.is_unique
assert "RD2" in re_.index and "RD1" not in re_.index and re_.loc["RD2", "same_day_readouts"] == 2
assert re_.loc["RD2", "car_1_1"] > 0.05
assert re_.loc["RH", "car_1_1"] < -0.25, re_.loc["RH", ["ar0", "car_1_1"]]            # halt: drop on resumption counted
c7, cx = PRICES["T7"]["Close"], PRICES["XBI"]["Close"]               # no-trade day 0: the return runs from day -1 to day +1
exp_rs = (c7[days[kS + 1]] / c7[days[kS - 1]] - 1) - (cx[days[kS + 1]] / cx[days[kS - 1]] - 1)
assert pd.isna(re_.loc["RS", "ar0"]) and abs(re_.loc["RS", "car01"] - exp_rs) < 1e-9, (re_.loc["RS", "car01"], exp_rs)
assert ns["bad_prints"].get("T7", 0) >= 1 and abs(re_.loc["RB", "car_1_1"]) < 0.5 and not re_.loc["RB", "extreme"]
assert pd.isna(re_.loc["RX", "car_1_1"]) and re_.loc["RX", "timing"] == "unknown"
top = re_[(re_["readout_kind"] == "topline_or_endpoint") & re_["has_prices"] & re_.index.str.match(r"R\d")]
neg, pos = top[top["readout_tone"] == "negative"], top[top["readout_tone"] == "positive"]
assert -0.42 < neg["car_1_1"].mean() < -0.28 and 0.06 < pos["car_1_1"].mean() < 0.18, (neg["car_1_1"].mean(), pos["car_1_1"].mean())
ac = neg[neg["timing"] == "after_close"]
assert ac["ar_m1"].mean() < -0.25 and abs(ac["ar0"].mean()) < 0.05, "after-close: move on day -1"
assert re_["overlap"].dtype == bool and re_["earnings_day"].sum() > 0
assert set(re_.loc["RD2", "same_day_accessions"].split(";")) == {"RD1", "RD2"}
assert re_.loc["R1", "phase"] == 1 and re_.loc["R0", "phase"] != re_.loc["R0", "phase"] and re_.loc["R3", "phase"] == 2
h = re_[re_["has_prices"]]
for c in ("car_m10_m2", "car_2_20", "price_pre", "log_dollar_vol"):
    assert h[c].notna().mean() > 0.9, c
se = pd.read_parquet(f"{F}/stop_events.parquet").set_index("event_id")
nct = set(";".join(se["nct_ids"]).split(";"))
assert not nct & {"NCT00000011", "NCT00000005", "NCT00000007", "NCTX", "NCTY", "NCTZ"}, "left-out stops"
b0id = f"{ciks['T1']}_{b0['registry_date']:%Y%m%d}"
b1id = f"{ciks['T2']}_{b1['registry_date']:%Y%m%d}"
assert se.loc[b0id, "group"] == "mixed batch" and se.loc[b0id, "n_trials"] == 2
assert se.loc[b1id, "group"] == "registry first or only" and se.loc[b1id, "n_trials"] == 2
g = se[se["has_prices"]].groupby("group")["car01"].mean()
assert g["registry first or only"] < -0.08 and abs(g["filing first"]) < 0.04, g
assert se["same_day_8k"].dtype == bool
assert se.loc[b0id, "late_stage"] and se.loc[b0id, "material_reason"]               # Phase 3 + efficacy / safety in the batch
assert set(se["registry_detail"]) >= {"never disclosed (within a year)", "disclosed after the registry", ""}
assert set(";".join(se["reason"]).split(";")) >= {"efficacy", "business", "enrollment", "not given", "safety"}
dse = pd.read_parquet(f"{F}/stop_disclosure_events.parquet")
q = dse[dse["has_prices"] & ~dse["near_readout"]].groupby("when")["car_1_1"].mean()
assert q["before the registry"] < -0.06 and abs(q["after the registry"]) < 0.04, q
assert (dse["n_trials"] == 2).any()                                                  # NCT1 and NCTB1 share a filing date
assert FAILED, "the OCC failure happened"
occ = pd.read_parquet(f"{F}/occ_events.parquet").set_index("accession")
assert pd.isna(occ.loc["O3", "opt_event"]), "days after a failed request are asked next run, not saved as zero"
assert max(fk.occ_days) <= TODAY - pd.Timedelta(days=3)
dv = pd.read_csv(f"{O}/stock_daily_volume.csv", dtype=str, on_bad_lines="skip")
assert all(((dv["ticker"] == "T3") & (dv["date"] == x.date().isoformat())).any() for x in wz), "old zeros asked again"
assert pd.to_numeric(dv["contracts"], errors="coerce").notna().sum() >= len(dv) - 1
for f_ in ["fig6_readout_car", "fig6b_readout_car_by_timing", "fig7_stop_car", "fig8_occ_volume"]:
    assert os.path.exists(f"{P}/features/figures/{f_}.png"), f_
print("TEST FULL OK\n" + "=" * 70)

# 3) another day: T4 and DEAD are retried; T4 comes back, DEAD is given up after its third try
miss = json.load(open(f"{PR}/_missing.json"))
for t in miss:
    miss[t]["last"] = "2000-01-01"
json.dump(miss, open(f"{PR}/_missing.json", "w"))
ns, fk, _ = run({"DRY_RUN": False}, Fakes())
assert sorted(sum(fk.yf_calls, [])) == ["DEAD", "T4"], fk.yf_calls
miss = json.load(open(f"{PR}/_missing.json"))
assert "T4" not in miss and miss["DEAD"]["tries"] == 3 and os.path.exists(f"{PR}/T4.parquet")
occ = pd.read_parquet(f"{F}/occ_events.parquet").set_index("accession")
o3 = occ.loc[["O1", "O2", "O3"]]
assert (o3["log_opt_event"] > 1.0).all() and (o3["log_opt_pre"].abs() < 0.5).all() and o3["opt_ok"].all(), o3
print("TEST RETRY OK\n" + "=" * 70)

# 4) resume: nothing asked again
ns, fk, _ = run({"DRY_RUN": False}, Fakes())
assert fk.yf_calls == [] and fk.occ_calls == 0, (fk.yf_calls, fk.occ_calls)
print("TEST RESUME OK\n" + "=" * 70)

# 5) Yahoo down: stops with a message, records nothing as missing; the next run gets them
for t in ("T5", "T6"):
    os.remove(f"{PR}/{t}.parquet")
ns, fk, err = run({"DRY_RUN": False}, Fakes(down=True), cells=[C.CONFIG, C.SETUP, C.LOAD, C.PRICES], expect=RuntimeError)
assert err and "not answering" in str(err) and "T5" not in json.load(open(f"{PR}/_missing.json"))
ns, fk, _ = run({"DRY_RUN": False}, Fakes(), cells=[C.CONFIG, C.SETUP, C.LOAD, C.PRICES])
assert os.path.exists(f"{PR}/T5.parquet") and os.path.exists(f"{PR}/T6.parquet")
print("TEST YAHOO DOWN OK\n" + "=" * 70)

# 6) a cache checked by the patched v1 cell ('ok' marker): side columns kept, 'no options' days asked again once
open(f"{O}/cache_v2_checked.txt", "w").write("ok\n")
zero_keys = {k.strip() for k in open(f"{O}/stock_daily_done.txt")} - {f"{a} {b.replace('-', '')}" for a, b in
             pd.read_csv(f"{O}/stock_daily_volume.csv", dtype=str)[["ticker", "date"]].itertuples(index=False)}
n_side = pd.read_csv(f"{O}/stock_daily_volume.csv")["call_customer"].notna().sum()
ns, fk, _ = run({"DRY_RUN": False}, Fakes(), cells=[C.CONFIG, C.SETUP, C.LOAD, C.PRICES, C.EVENTS, C.OCC])
assert open(f"{O}/cache_v2_checked.txt").read().startswith("v2") and fk.occ_calls >= len(zero_keys) > 0
assert pd.read_csv(f"{O}/stock_daily_volume.csv")["call_customer"].notna().sum() >= n_side > 0, "side columns kept"
ns, fk, _ = run({"DRY_RUN": False}, Fakes(), cells=[C.CONFIG, C.SETUP, C.LOAD, C.PRICES, C.EVENTS, C.OCC])
assert fk.occ_calls == 0
print("TEST V1 CACHE OK\n" + "=" * 70)

# 7) an older Notebook 3b file (no in_sample column) is not used
pd.DataFrame(sdr).drop(columns=["in_sample"]).to_parquet(f"{F}/stop_disclosure.parquet", index=False)
ns, fk, _ = run({"DRY_RUN": False}, Fakes(), cells=[C.CONFIG, C.SETUP, C.LOAD])
assert ns["sd"] is None and ns["source"].startswith("Notebook 3 ")
print("TEST OLD 3B OK\nALL TESTS PASSED")

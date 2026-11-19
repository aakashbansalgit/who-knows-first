"""Notebook 8: Exits after trial stops

Pipeline stage of the who-knows-first replication code. Cells are separated by '# %%' markers, so the file
runs top to bottom as a script and opens as a notebook in Jupyter, VS Code or (after tools/make_notebooks.py) Colab.
See README.md for inputs, outputs and the run order.
"""


# %% [markdown]
# # Notebook 8: Exits after trial stops
#
# Notebook 7's long-window drift only sees firms that still trade today: about 4 in 10 stop events have no prices because the firm has no ticker any more (acquired, delisted, deregistered). This notebook finds out **how** each of those firms left, from its SEC filing history, and uses that in two ways.
#
# **1. Exits as an outcome (no prices needed).** For every stop event: within a year (and two), did the firm
# - exit by failure: bankruptcy (8-K Item 1.03), delisting after listing-rule notices (Item 3.01), any other delisting or deregistration without a merger (Form 25 or Form 15), or simply stop filing;
# - get acquired: a completed change in control (8-K Item 5.01) before it stopped filing (merger paperwork alone does not count: deals can fail);
# - receive a listing-rule notice (Item 3.01, not counting the 3.01 filed with a merger closing), which also covers firms that still trade.
# Compared across groups: firms that disclosed the stop first, later, or never; first channel 8-K vs 10-Q/10-K; per stop. If hiding a stop goes with later failure, firms that never disclose should fail more often.
#
# **2. Drift bounds.** Notebook 7's drift for survivors, with the missing events put back: a failure exit inside the window gets a delisting return (Shumway and Warther 1999 estimate -55% for performance-related Nasdaq delistings; -30% and -90% are shown too), an acquisition gets 0% or +50%, and missing events with no exit inside the window get their group's survivor mean.
#
# **Needs:** Notebook 4 v2 (`stop_events.parquet`, `stop_disclosure_events.parquet`), Notebook 7 (`drift_registry.parquet`, `drift_disclosure.parquet`, `channel_choice.parquet`). Set `USER_AGENT` (same as Notebooks 1 and 3b). About one SEC request per firm (more for firms with long histories), saved per firm; about 5 to 10 minutes. No GPU.
#
# **Output** (in `MyDrive/biopharma_tiv/features/nb8/`): `firm_exits.parquet`, `exit_outcomes_registry.parquet`, `exit_outcomes_disclosure.parquet`, `exit_outcomes_stops.parquet`, `drift_bounds.csv`, figure `fig15_exits.png`.


# %% CONFIG
# ==== Settings: edit USER_AGENT ====
import os
USER_AGENT = os.environ.get("SEC_USER_AGENT", "Your Name your.email@example.com")   # same as Notebooks 1 and 3b: the SEC requires a real name and email
PROJECT_DIR = os.environ.get("WKF_PROJECT_DIR", "/content/drive/MyDrive/biopharma_tiv")
MAX_RPS = 8                         # SEC limit is 10 requests/second
FOLLOW = [365, 730]                 # outcome horizons, calendar days after the event
TERMINAL_SPAN = 400                 # a Form 25 or 15 counts as the exit if it is within this many days of the firm's last filing
STILL_FILING = 180                  # no ticker, but filed within this many days of the data date: still a registrant
SIGNAL_SPAN = 365                   # merger signals in the year before the exit
FAIL_RETURNS = [-0.30, -0.55, -0.90]  # delisting returns for failure exits (Shumway and Warther 1999: -55% on Nasdaq)
ACQ_RETURNS = [0.0, 0.50]           # returns for acquisitions inside the window
SMALL_LEAD = 20


# %% [markdown]
# ## Step 0: Setup


# %% SETUP
import os, re, glob, json, time, random, threading, warnings
import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf

if PROJECT_DIR.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
assert "@" in USER_AGENT and "Your Name" not in USER_AGENT, "Set USER_AGENT to your real name and email first (first code cell)."
FEAT_DIR = os.path.join(PROJECT_DIR, "features")
EDGAR_DIR = os.path.join(PROJECT_DIR, "edgar")
NB7_DIR = os.path.join(FEAT_DIR, "nb7")
NB8_DIR = os.path.join(FEAT_DIR, "nb8")
HIST_DIR = os.path.join(NB8_DIR, "history")
FIG_DIR = os.path.join(FEAT_DIR, "figures")
for d in (NB8_DIR, HIST_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)
rng = np.random.default_rng(2026)

class FatalSECBlock(Exception):
    pass

class SECClient:
    """Polite SEC client: declared User-Agent, rate limit, retries with backoff (same as Notebooks 1 and 3b)."""
    def __init__(self, user_agent, max_rps=8):
        self.min_gap, self._next, self._lock = 1.0 / max_rps, 0.0, threading.Lock()
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.n_requests = 0

    def get(self, url, timeout=60, max_tries=6):
        last = None
        for attempt in range(max_tries):
            with self._lock:
                t = max(time.monotonic(), self._next)
                self._next = t + self.min_gap
            if t > time.monotonic():
                time.sleep(t - time.monotonic())
            try:
                r = self.s.get(url, timeout=timeout)
                self.n_requests += 1
            except requests.RequestException as e:
                last = f"{type(e).__name__}: {e}"
            else:
                if r.status_code in (200, 404):
                    return r
                if r.status_code == 403 and "Undeclared Automated Tool" in r.text:
                    raise FatalSECBlock("The SEC refused the User-Agent: put your real name and email in USER_AGENT.")
                last = f"HTTP {r.status_code}"
            time.sleep(min(60, 2 ** attempt + random.random()))
        raise RuntimeError(f"GET failed after {max_tries} tries: {url} ({last})")

SEC = SECClient(USER_AGENT, MAX_RPS)

def as_bool(s):
    return pd.Series(s).astype("boolean").fillna(False).astype(bool)

def col(df, c, default):
    return df[c] if c in df else pd.Series(default, index=df.index)

def as_day(s):
    d = pd.to_datetime(s, errors="coerce")
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    return d.dt.normalize().astype("datetime64[ns]")

def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)

def main_ticker(x):
    opts = [t.strip().upper() for t in re.split(r"[;,\s]+", str(x or "")) if t.strip()]
    opts = [t for t in opts if t not in ("NAN", "NONE", "<NA>")]
    return min(opts, key=len) if opts else None

TODAY = pd.Timestamp.today().normalize()
print("Saving to:", NB8_DIR)


# %% [markdown]
# ## Step 1: Firms and their filing histories
# Every firm with a stop event or a stop in Notebook 7's per-stop file. A firm has a ticker today if SEC's current list (Notebook 1's universe file) or Notebook 4 gives it one. Filing histories (form, date, 8-K items) come from `data.sec.gov/submissions`, saved per firm so a rerun makes no requests.


# %% LOAD
def need(path, step):
    assert os.path.exists(path), f"Missing {path}: run {step} first."
    return pd.read_parquet(path)

se = need(os.path.join(FEAT_DIR, "stop_events.parquet"), "Notebook 4 v2")
dse = need(os.path.join(FEAT_DIR, "stop_disclosure_events.parquet"), "Notebook 4 v2")
cc = need(os.path.join(NB7_DIR, "channel_choice.parquet"), "Notebook 7")
se["registry_date"], dse["first_date"], cc["registry_date"] = as_day(se["registry_date"]), as_day(dse["first_date"]), as_day(cc["registry_date"])
for d in (se, dse, cc):
    d["cik"] = pd.to_numeric(d["cik"], errors="coerce")
    d.dropna(subset=["cik"], inplace=True)
    d["cik"] = d["cik"].astype(int)

tick = {}
uni = sorted(glob.glob(os.path.join(EDGAR_DIR, "universe_*.parquet")), key=os.path.getmtime)
if uni:
    u = pd.read_parquet(uni[-1], columns=["cik", "tickers"])
    for c, t in zip(pd.to_numeric(u["cik"], errors="coerce"), u["tickers"]):
        if pd.notna(c) and main_ticker(t):
            tick[int(c)] = main_ticker(t)
for d in (se, dse):
    for c, t in zip(d["cik"], col(d, "ticker", None)):
        if isinstance(t, str) and t:
            tick.setdefault(int(c), t)
firms = sorted(set(se["cik"]) | set(dse["cik"]) | set(cc["cik"]))
print(f"Firms: {len(firms):,} | with a ticker today: {sum(c in tick for c in firms):,}")

def firm_history(cik):
    fp = os.path.join(HIST_DIR, f"{cik}.parquet")
    if os.path.exists(fp):
        return pd.read_parquet(fp)
    cols = ["form", "filingDate", "accessionNumber", "items"]
    r = SEC.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    if r.status_code != 200:
        fl = pd.DataFrame(columns=cols)
    else:
        j = r.json()
        pages = [j.get("filings", {}).get("recent", {})]
        for f in j.get("filings", {}).get("files", []):
            r2 = SEC.get(f"https://data.sec.gov/submissions/{f['name']}")
            if r2.status_code == 200:
                pages.append(r2.json())
        fl = pd.concat([pd.DataFrame({c: p.get(c, [""] * len(p.get("form", []))) for c in cols}) for p in pages], ignore_index=True)
    fl["filingDate"] = pd.to_datetime(fl["filingDate"], errors="coerce")
    fl["items"] = fl["items"].fillna("").astype(str)
    fl["form"] = fl["form"].fillna("").astype(str)
    fl.to_parquet(fp, index=False)
    return fl

t0, hist = time.time(), {}
for k, c in enumerate(firms, 1):
    hist[c] = firm_history(c)
    if k % 100 == 0 or k == len(firms):
        print(f"  filing histories: {k}/{len(firms)} | {SEC.n_requests:,} requests | {time.time() - t0:.0f}s")


# %% [markdown]
# ## Step 2: How each firm without a ticker left
# `exit_date`: the first Form 25 or Form 15 in the last `TERMINAL_SPAN` days of the firm's filings, else its last filing. `exit_type` looks at the year before the exit: bankruptcy (Item 1.03 within two years), acquisition (a completed change in control, Item 5.01), delisting after listing-rule notices (Item 3.01 outside a merger closing), other delisting or deregistration, stopped filing without either form. Firms with a ticker are `listed today`; firms without one that still file are `still filing, no listed ticker`.


# %% EXITS
TERM = re.compile(r"^(?:25|25-NSE|15-12B|15-12G|15-15D|15F-12B|15F-12G|15F-15D)(?:/A)?$")
MERGER = re.compile(r"^(?:DEFM14A|PREM14A|DEFM14C|PREM14C|SC 14D9|SC TO-T|425)(?:/A)?$")
FAIL_TYPES = {"bankruptcy", "delisted after listing-rule notices", "delisted or deregistered, other", "stopped filing"}

def has_item(items, code):
    return items.str.contains(rf"(?<![\d.]){re.escape(code)}(?!\d)", regex=True)

def classify(cik):
    fl = hist.get(cik)
    if cik in tick:
        return {"cik": cik, "status": "listed today", "exit_type": "listed today", "exit_date": pd.NaT}
    if fl is None or not len(fl) or fl["filingDate"].isna().all():
        return {"cik": cik, "status": "no history", "exit_type": "no history", "exit_date": pd.NaT}
    fl = fl.dropna(subset=["filingDate"]).sort_values("filingDate")
    last = fl["filingDate"].max()
    term = fl[fl["form"].str.match(TERM) & (fl["filingDate"] >= last - pd.Timedelta(days=TERMINAL_SPAN))]
    if len(term):
        exit_date, how = term["filingDate"].min(), "form 25/15"
    elif last >= TODAY - pd.Timedelta(days=STILL_FILING):
        return {"cik": cik, "status": "still filing, no listed ticker", "exit_type": "still filing, no listed ticker",
                "exit_date": pd.NaT, "last_filing": last}
    else:
        exit_date, how = last, "stopped filing"
    w = fl[(fl["filingDate"] >= exit_date - pd.Timedelta(days=SIGNAL_SPAN)) & (fl["filingDate"] <= exit_date + pd.Timedelta(days=60))]
    w8 = w[w["form"].str.startswith("8-K")]
    b8 = fl[fl["form"].str.startswith("8-K") & (fl["filingDate"] >= exit_date - pd.Timedelta(days=730))
            & (fl["filingDate"] <= exit_date + pd.Timedelta(days=60))]
    bankrupt = bool(has_item(b8["items"], "1.03").any())
    acquired = bool(has_item(w8["items"], "5.01").any())          # a completed change in control, not just merger paperwork
    rule8 = w8[~has_item(w8["items"], "2.01") & ~has_item(w8["items"], "5.01")]    # 3.01 filed with a merger closing is not a notice
    notices = bool(has_item(rule8["items"], "3.01").any())
    if bankrupt:
        et = "bankruptcy"
    elif acquired:
        et = "acquired"
    elif how == "stopped filing":
        et = "stopped filing"
    elif notices:
        et = "delisted after listing-rule notices"
    else:
        et = "delisted or deregistered, other"
    return {"cik": cik, "status": "exited", "exit_type": et, "exit_date": exit_date, "exit_how": how, "last_filing": last}

ex = pd.DataFrame([classify(c) for c in firms])
ex["failure_exit"] = ex["exit_type"].isin(FAIL_TYPES)
ex.to_parquet(os.path.join(NB8_DIR, "firm_exits.parquet"), index=False)
print("=== Firms by exit type ===")
print(ex["exit_type"].value_counts().to_string())
print(f"\nExit year (exited firms): {ex.loc[ex['status'] == 'exited', 'exit_date'].dt.year.value_counts().sort_index().to_dict()}")


# %% [markdown]
# ## Step 3: Outcomes after each event
# For each event date: failure exit, acquisition, listing-rule notice and bankruptcy filing within one and two years. Events without a full year (two years) of follow-up before today are left out of that horizon; events at firms that had already exited are dropped. Three sets: registry posting dates (by who spoke first), first-disclosure dates (by channel), and every in-sample stop (by first channel, from Notebook 7). Logits clustered by firm.


# %% OUTCOMES
EXI = ex.set_index("cik")
NOTICE = {c: np.sort(fl.loc[fl["form"].str.startswith("8-K") & has_item(fl["items"], "3.01") & ~has_item(fl["items"], "2.01")
                             & ~has_item(fl["items"], "5.01"), "filingDate"].dropna().to_numpy())
          for c, fl in hist.items() if len(fl)}
BANKR = {c: np.sort(fl.loc[fl["form"].str.startswith("8-K") & has_item(fl["items"], "1.03"), "filingDate"].dropna().to_numpy())
         for c, fl in hist.items() if len(fl)}

def any_in(arr, a, b):
    return bool(arr is not None and len(arr) and ((arr > np.datetime64(a)) & (arr <= np.datetime64(b))).any())

def outcomes(df, date_col):
    out = df.copy()
    e = EXI.reindex(out["cik"])
    out["exit_type"] = e["exit_type"].to_numpy()
    out["exit_date"] = e["exit_date"].to_numpy()
    out["failure_firm"] = e["failure_exit"].astype("boolean").fillna(False).astype(bool).to_numpy()
    out["exited_before"] = out["exit_date"].notna() & (out["exit_date"] <= out[date_col])
    for h in FOLLOW:
        end = out[date_col] + pd.Timedelta(days=h)
        ok = end <= TODAY
        in_w = out["exit_date"].notna() & (out["exit_date"] > out[date_col]) & (out["exit_date"] <= end)
        out[f"fail_{h}"] = np.where(ok, (in_w & out["failure_firm"]).astype(float), np.nan)
        out[f"acq_{h}"] = np.where(ok, (in_w & out["exit_type"].eq("acquired")).astype(float), np.nan)
        out[f"notice_{h}"] = np.where(ok, [float(any_in(NOTICE.get(c), d, d + pd.Timedelta(days=h))) for c, d in zip(out["cik"], out[date_col])], np.nan)
        out[f"bankr_{h}"] = np.where(ok, [float(any_in(BANKR.get(c), d, d + pd.Timedelta(days=h))) for c, d in zip(out["cik"], out[date_col])], np.nan)
    out["year"] = out[date_col].dt.year
    return out[~out["exited_before"]].copy()

def table(df, gcol, groups, title):
    print(f"\n=== {title} ===")
    rows = []
    for g in groups:
        x = df[df[gcol] == g]
        r = {"group": g}
        for v in ["fail", "acq", "notice", "bankr"]:
            for h in FOLLOW:
                y = x[f"{v}_{h}"].dropna()
                r[f"{v}_{h}"], r[f"n_{h}"] = (y.mean() if len(y) else np.nan), len(y)
        lo, hi = wilson(int(x["fail_365"].dropna().sum()), int(x["fail_365"].notna().sum()))
        r["fail_365_ci"] = f"{lo:.0%} to {hi:.0%}" if not np.isnan(lo) else ""
        rows.append(r)
    t = pd.DataFrame(rows).set_index("group")
    show = t[["n_365", "fail_365", "fail_365_ci", "acq_365", "notice_365", "bankr_365", "n_730", "fail_730", "acq_730", "notice_730"]]
    print(show.to_string(float_format=lambda v: f"{v:.0%}"))
    return t

res_logit = {}
def logits(df, gcol, base, lab, controls):
    for y in ["fail_365", "notice_365", "fail_730"]:
        z = df.dropna(subset=[y]).copy()
        z = z[z[gcol].isin(df[gcol].dropna().unique())]
        if len(z) < 80 or z[y].sum() < 10 or z["cik"].nunique() < 10 or z[gcol].nunique() < 2:
            print(f"  {lab}, {y}: too few events")
            continue
        rhs = f"C({gcol}, Treatment(reference='{base}'))" + "".join(f" + {c}" for c in controls if c in z and z[c].nunique() > 1) + " + year"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = smf.logit(f"{y} ~ {rhs}", z).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(z["cik"])[0]})
            res_logit[(lab, y)] = m
            keep = [i for i in m.params.index if i not in ("Intercept", "year")]
            print(f"-- {lab}: {y} (n = {len(z):,}, yes = {int(z[y].sum()):,}; base: {base})")
            print(pd.DataFrame({"odds ratio": np.exp(m.params[keep]), "p": m.pvalues[keep]}).round(3).to_string())
        except Exception as e:
            print(f"  {lab}, {y}: logit did not fit ({type(e).__name__}: {str(e)[:120]})")

# registry posting dates
R = se.copy()
R["registry_detail"] = col(R, "registry_detail", "").fillna("").astype(str)
R["sgroup"] = np.select([R["group"].eq("filing first"), R["registry_detail"].str.startswith("never"),
                         R["registry_detail"].str.startswith("disclosed after")],
                        ["filing first", "registry first, never disclosed", "registry first, disclosed later"], "mixed or unknown")
R["small"] = pd.to_numeric(col(R, "firm_lead_trials", np.nan), errors="coerce").le(SMALL_LEAD).astype(int)
R["material"] = as_bool(col(R, "material_reason", False)).astype(int).to_numpy()
R["late"] = as_bool(col(R, "late_stage", False)).astype(int).to_numpy()
R = outcomes(R, "registry_date")
SG = ["filing first", "registry first, disclosed later", "registry first, never disclosed"]
tR = table(R, "sgroup", SG, "Registry posting date: outcomes by who spoke first")
logits(R, "sgroup", "filing first", "registry date", ["small", "material", "late"])

# first-disclosure dates
Dq = dse.copy()
ch = col(Dq, "channel", "").fillna("").astype(str)
Dq["chan"] = np.select([ch.str.contains("8-K"), ch.str.contains(r"10-Q|10-K|20-F|40-F")], ["8-K", "10-Q/10-K"], "other filing")
Dq["small"] = pd.to_numeric(col(Dq, "firm_lead_trials", np.nan), errors="coerce").le(SMALL_LEAD).astype(int)
Dq["material"] = as_bool(col(Dq, "material_reason", False)).astype(int).to_numpy()
Dq["late"] = as_bool(col(Dq, "late_stage", False)).astype(int).to_numpy()
Dq = outcomes(Dq, "first_date")
tD = table(Dq, "chan", ["8-K", "10-Q/10-K", "other filing"], "First disclosure in a filing: outcomes by channel")
logits(Dq, "chan", "8-K", "disclosure date", ["small", "material", "late"])

# every in-sample stop (Notebook 7), dated by the registry posting
Ps = outcomes(cc, "registry_date")
PG = ["8-K first", "10-Q/10-K first", "other filing first", "never"]
tP = table(Ps, "outcome", PG, "Every in-sample stop: outcomes by first channel")
logits(Ps, "outcome", "8-K first", "per stop", ["small", "material", "late"])

for name, X in [("registry", R), ("disclosure", Dq), ("stops", Ps)]:
    X.to_parquet(os.path.join(NB8_DIR, f"exit_outcomes_{name}.parquet"), index=False)


# %% [markdown]
# ## Step 4: Drift with the missing firms put back
# Notebook 7's drift vs placebo for events with prices, plus the events without prices: a failure exit inside the window gets each `FAIL_RETURNS` value, an acquisition each `ACQ_RETURNS` value, and the rest their group's survivor mean (no information either way). Windows +2..+60 and +2..+120 trading days are taken as 87 and 175 calendar days.


# %% BOUNDS
CAL_DAYS = {"2_60": 87, "2_120": 175}
rows = []
for lab, X, ev, dcol, gcol, groups in [
        ("registry date", pd.read_parquet(os.path.join(NB7_DIR, "drift_registry.parquet")), R, "registry_date", "sgroup", SG),
        ("disclosure date", pd.read_parquet(os.path.join(NB7_DIR, "drift_disclosure.parquet")), Dq, "first_date", "chan", ["8-K", "10-Q/10-K"])]:
    have = set(X["event_id"])
    miss = ev[~ev["event_id"].isin(have)].copy()
    for k, days in CAL_DAYS.items():
        y = f"xcar_{k}"
        surv = X[X[gcol].isin(groups)].groupby(gcol)[y].agg(["mean", "size"])
        in_w = miss["exit_date"].notna() & (miss["exit_date"] > miss[dcol]) & (miss["exit_date"] <= miss[dcol] + pd.Timedelta(days=days))
        for fr in FAIL_RETURNS:
            for ar in ACQ_RETURNS:
                r = {"events": lab, "window": k, "fail_return": fr, "acq_return": ar}
                for g in groups:
                    m = miss[miss[gcol] == g]
                    w = in_w[miss[gcol] == g]
                    base = surv.loc[g, "mean"] if g in surv.index else np.nan
                    vals = np.where(w & m["failure_firm"], fr, np.where(w & m["exit_type"].eq("acquired"), ar, base))
                    s_ = X.loc[X[gcol] == g, y].dropna()
                    allv = np.concatenate([s_.to_numpy(), vals[np.isfinite(vals)]])
                    r[f"{g}: mean"] = allv.mean() if len(allv) else np.nan
                    r[f"{g}: n"] = len(allv)
                    r[f"{g}: failures in window"] = int((w & m["failure_firm"]).sum())
                    r[f"{g}: survivors only"] = base
                rows.append(r)
bounds = pd.DataFrame(rows)
bounds.to_csv(os.path.join(NB8_DIR, "drift_bounds.csv"), index=False)
SHORT = {"filing first": "filing first", "registry first, disclosed later": "disclosed later", "registry first, never disclosed": "never",
         "8-K": "8-K", "10-Q/10-K": "10-Q/10-K"}
for lab, groups, ref in [("registry date", SG, "filing first"), ("disclosure date", ["8-K", "10-Q/10-K"], "8-K")]:
    b = bounds[bounds["events"] == lab]
    print(f"\n=== {lab}: drift vs placebo, survivors only and with missing events put back ===")
    for k in CAL_DAYS:
        bk = b[b["window"] == k]
        if not len(bk):
            continue
        first = bk.iloc[0]
        print(f"-- days +{k.replace('_', '..+')} | failure exits inside the window: "
              + ", ".join(f"{SHORT[g]} {int(first[f'{g}: failures in window'])}" for g in groups))
        print(f"   {'survivors only':34s}" + " | ".join(f"{SHORT[g]} {first[f'{g}: survivors only']:+.1%}" for g in groups))
        for r in bk.itertuples(index=False):
            d = dict(zip(bk.columns, r))
            print(f"   {'failure ' + format(d['fail_return'], '+.0%') + ', acquisition ' + format(d['acq_return'], '+.0%'):34s}"
                  + " | ".join(f"{SHORT[g]} {d[f'{g}: mean']:+.1%}" for g in groups)
                  + "   gap vs " + SHORT[ref] + ": " + ", ".join(f"{SHORT[g]} {d[f'{g}: mean'] - d[f'{ref}: mean']:+.1%}" for g in groups if g != ref))


# %% [markdown]
# ## Step 5: Figure
# Share of stops followed by a failure exit, by days since the registry posting (Kaplan-Meier: firms are followed until today; acquisitions end the follow-up). Check the printed output of Steps 2 to 4 and the figure.


# %% FIGS
import matplotlib.pyplot as plt
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"8-K first": "#2a78d6", "10-Q/10-K first": "#1baf7a", "never": "#eb6834", "other filing first": "#eda100"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "legend.frameon": False, "axes.axisbelow": True})

def km(t, event, horizon=1095):
    """Cumulative incidence 1 - S(t) on a daily grid, Kaplan-Meier with censoring."""
    t, event = np.asarray(t, float), np.asarray(event, bool)
    grid = np.arange(0, horizon + 1)
    s, out, at_risk = 1.0, [], len(t)
    order = np.argsort(t)
    t, event = t[order], event[order]
    i = 0
    for d in grid:
        while i < len(t) and t[i] < d:
            at_risk -= 1
            i += 1
        j, deaths = i, 0
        while j < len(t) and t[j] == d:
            deaths += int(event[j])
            j += 1
        if at_risk > 0 and deaths:
            s *= 1 - deaths / at_risk
        out.append(1 - s)
    return grid, np.array(out)

P2 = Ps.copy()
end = P2["exit_date"].where(P2["exit_date"].notna(), TODAY)
P2["t"] = (end - P2["registry_date"]).dt.days.clip(lower=0)
P2["ev"] = P2["exit_date"].notna() & P2["failure_firm"]
fig, ax = plt.subplots(figsize=(8, 4))
for g in ["8-K first", "10-Q/10-K first", "never"]:
    x = P2[P2["outcome"] == g]
    if len(x) < 10:
        continue
    grid, ci = km(x["t"], x["ev"])
    ax.step(grid, ci * 100, where="post", color=COL[g], linewidth=2.2, label=f"{g} (n = {len(x):,})")
ax.set_xlabel("Days since the registry posted the stop")
ax.set_ylabel("Firms that exited by failure (%)")
ax.set_title("Failure exits after a trial stop, by where the stop was first disclosed", loc="left")
ax.legend(fontsize=9, loc="upper left")
fig.savefig(os.path.join(FIG_DIR, "fig15_exits.png"), dpi=200, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## What next
# - Check the printed output of Steps 2 to 4 and the figure.
# - If never-disclosed stops are followed by more failure exits than 8-K stops, quiet disclosure goes with worse outcomes, even if survivors' prices don't show it. If the rates match, and the drift bounds keep the groups close, the materiality reading holds with the dead firms included.
# - Spot-check a few firms in `firm_exits.parquet` on EDGAR (the exit type is read from form types and 8-K item numbers, not from the text).

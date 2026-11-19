"""Notebook 4 — Prices, event studies, options volume, first tests (v2)

Pipeline stage of the who-knows-first replication code. Cells are separated by '# %%' markers, so the file
runs top to bottom as a script and opens as a notebook in Jupyter, VS Code or (after tools/make_notebooks.py) Colab.
See README.md for inputs, outputs and the run order.
"""


# %% [markdown]
# # Notebook 4 — Prices, event studies, options volume, first tests (v2)
#
# 1. **Prices.** Daily adjusted closes and volume from Yahoo (`yfinance`) for every firm with a readout or a trial stop, plus the biotech ETF **XBI** as the benchmark. Abnormal return = stock return − XBI return over the same days.
# 2. **Readouts.** The main window is **days −1 to +1** (`car_1_1`): many firms put out the press release during the day and file the 8-K after the close, so the market reacts the day before the 8-K's day 0. Also kept: `ar0`, `car_1_0` (days −1..0), `car01`, `car05`, abnormal volume (days −1..+1 over the median of days −60..−11) and `vol_ratio` (std of abnormal returns on days +2..+21 over days −60..−11).
# 3. **Registry stops** around the day ClinicalTrials.gov posted the stop, split by whether a filing disclosed it first (Notebook 3b if its v3 output exists, else Notebook 3); registry-first stops are split into never disclosed vs disclosed later, and there is a subsample of Phase 3 stops for safety or efficacy at firms leading 1–20 trials (`why_stopped` sorted into safety / efficacy / enrollment / business / other). Also the reaction around the **filing that first disclosed the stop** (Notebook 3b's date), before vs after the registry.
# 4. **Coverage** by year (survivorship: delisted firms are mostly missing from Yahoo).
# 5. **OCC options volume** for readouts inside OCC's two-year window.
# 6. **First tests** (t-stats clustered by firm and by event date) and four figures.
#
# **Edge cases handled**
# - *Prices.* A return runs from the stock's previous close, and XBI's return is taken over the same days, so a trading halt or a missing day moves the return to the day trading resumes (returns spanning more than `MAX_GAP` trading days are dropped). Removed: non-positive prices, zero-volume days that repeat the last close, and one-day bad prints (a close 3x or more away from the days on both sides of it). A cached price file is downloaded again when it stops before the data an event needs. A ticker Yahoo returns nothing for is retried on up to three different days, so one rate-limited day doesn't drop it. If Yahoo stops answering, the cell stops and keeps what it has.
# - *Readouts.* One per firm per market day (topline first). Flagged and used in robustness checks: another readout from the same firm within 5 trading days, an earnings 8-K (Item 2.02), fewer than 60 trading days listed, and an abnormal move above 200% in days −1..+1 (sometimes a data error).
# - *Stops.* Several stops posted by one firm on one day are one event (`mixed batch` when some were disclosed first and some weren't). Flagged: an 8-K or a readout within a trading day. Notebook 3b's file is used only if it has v3's `in_sample` column; stops outside its sample, or missing from it, are left out and counted.
# - *OCC.* A failed request is never saved as "no options traded"; only OCC's own "no record" answer is. Rows have fixed columns, a day's volume is saved before the day is marked as asked, and a line cut off by a disconnect is skipped. OCC's oldest available day is checked first, and the last three days aren't asked yet (OCC may not have published them). A cache made by an earlier version is checked once: its days with volume are kept and its "no options" days are asked again, because those versions could save a failed, out-of-window or not-yet-published request that way.
#
# **How to run**
# - Set `USER_AGENT` (sent to OCC).
# - **First run:** `DRY_RUN = True` (40 tickers, 5 OCC readouts, a few minutes). Check the last cell's output.
# - **Full run:** `DRY_RUN = False`. Prices take ~10–20 minutes. OCC: the first run after this update asks again for the cache's no-volume days (roughly 1–2 hours at one request a second); later runs only ask new days. `RUN_OCC = False` skips it.
#
# **Output** (in `MyDrive/biopharma_tiv/`): `prices/` (one file per ticker, `_missing.json`), `features/readout_events.parquet`, `features/stop_events.parquet`, `features/stop_disclosure_events.parquet`, `features/event_paths.parquet`, `features/occ_events.parquet`, `occ/stock_daily_volume.csv`, `features/figures/fig6-8*.png`.
#
# **For Notebook 5** `readout_events.parquet` carries, per readout: the outcomes (`car_1_1`, `abs_car`, `ar0`, `car05`, `post_vol`, `vol_ratio`), controls (`pre_vol`, `log_dollar_vol` as a size proxy, `price_pre`, `car_m10_m2` run-up, `timing`, `earnings_day`, `year`, `phase` of the linked trials), labels (`readout_kind`, `readout_tone`), flags, `car_2_20` (drift after), and `same_day_accessions` to pull every 8-K text of that firm-day.
#
# **Known limits.** Yahoo is missing most delisted firms (survivorship). The OCC query uses today's ticker, so a firm that changed ticker inside the two-year window shows no options volume before the change. Yahoo data is for personal and academic research use, not for commercial products.


# %% CONFIG
# ==== Settings ====
import os
PROJECT_DIR = os.environ.get("WKF_PROJECT_DIR", "/content/drive/MyDrive/biopharma_tiv")
USER_AGENT = os.environ.get("SEC_USER_AGENT", "Your Name your.email@example.com")   # name and email, sent to OCC
START = "2009-06-01"           # first price date
BENCH = "XBI"                  # SPDR S&P Biotech ETF: benchmark for abnormal returns
DRY_RUN = True                 # True: first DRY_RUN_TICKERS tickers and DRY_RUN_OCC readouts; outputs end in _dryrun
DRY_RUN_TICKERS = 40
DRY_RUN_OCC = 5
RUN_OCC = True                 # OCC options volume for readouts in OCC's two-year window
OCC_BEFORE, OCC_AFTER = 10, 3  # trading days around each readout
OCC_MIN_BASE = 5               # contracts a day on days -10..-6 needed for the options build-up tests
EST_WINDOW = (-60, -11)        # trading days for pre-event volatility and volume
MIN_EST_DAYS = 30              # one-day returns needed in that window
MAX_GAP = 5                    # a return may span at most this many trading days (halts, missing prints)
WINSOR = 0.01                  # tails clipped in the regression and the figures


# %% [markdown]
# ## Step 0 — Setup


# %% SETUP
import os, re, io, csv, json, glob, time, warnings, datetime as dt
import numpy as np
import pandas as pd
try:
    import yfinance as yf
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yfinance", "statsmodels"], check=True)
    import yfinance as yf
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
import requests

if PROJECT_DIR.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
assert "@" in USER_AGENT and "Your Name" not in USER_AGENT, "Set USER_AGENT to your name and email first (first code cell)."

EDGAR_DIR = os.path.join(PROJECT_DIR, "edgar")
FEAT_DIR = os.path.join(PROJECT_DIR, "features")
PRICE_DIR = os.path.join(PROJECT_DIR, "prices")
OCC_DIR = os.path.join(PROJECT_DIR, "occ")
FIG_DIR = os.path.join(FEAT_DIR, "figures")
for d in (PRICE_DIR, OCC_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)
SUFFIX = "_dryrun" if DRY_RUN else ""
TODAY = pd.Timestamp(dt.date.today())

def yf_download(symbols):
    """Adjusted daily Close and Volume from Yahoo -> {symbol: DataFrame}. Symbols Yahoo returns nothing for are left out."""
    kw = dict(start=START, auto_adjust=True, group_by="ticker", progress=False, threads=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            try:
                data = yf.download(symbols, multi_level_index=True, **kw)
            except TypeError:                                     # older yfinance without multi_level_index
                data = yf.download(symbols, **kw)
        except Exception as e:                                    # network or rate-limit error for the whole batch
            print(f"    Yahoo error: {type(e).__name__}: {str(e)[:150]}")
            return {}
    out = {}
    if data is None or len(data) == 0:
        return out
    if isinstance(data.columns, pd.MultiIndex) and "Close" in data.columns.get_level_values(0):
        data = data.swaplevel(0, 1, axis=1)                       # (Price, Ticker) -> (Ticker, Price)
    for s in symbols:
        if isinstance(data.columns, pd.MultiIndex):
            sub = data[s] if s in data.columns.get_level_values(0) else None
        else:
            sub = data if len(symbols) == 1 else None
        if sub is None or "Close" not in sub:
            continue
        sub = sub[["Close", "Volume"]] if "Volume" in sub else sub[["Close"]].assign(Volume=np.nan)
        sub = sub.dropna(subset=["Close"])
        if len(sub):
            out[s] = sub
    return out

OCC_URL = "https://marketdata.theocc.com/volume-query"

def occ_fetch(params):
    """One OCC volume-query answer: csv text (starts with 'quantity') or OCC's 'No record(s) found'.
    None when OCC gave no proper answer after three tries; that is never saved as 'no options traded'."""
    for attempt in range(3):
        try:
            r = requests.get(OCC_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
            time.sleep(1)
            text = r.text.lstrip("﻿ \r\n\t")
            if r.status_code == 200 and (text.startswith("quantity") or "no record" in text[:200].lower()):
                return text
        except requests.RequestException:
            pass
        time.sleep(10 * (attempt + 1))
    return None

def ensure_newline(path):
    """A run cut off mid-write can leave the last line without its newline; the next append must start a new line."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            end = f.read(1)
        if end != b"\n":
            with open(path, "a") as f:
                f.write("\n")

def winsor(s, q=None):
    """Clip the q and 1-q tails (left as is with fewer than 20 values)."""
    q = WINSOR if q is None else q
    s = pd.Series(s, dtype=float)
    if not q or s.notna().sum() < 20:
        return s
    lo, hi = s.quantile([q, 1 - q])
    return s.clip(lo, hi)

def cluster_groups(firm, date=None):
    """Cluster codes: by firm, and also by event date when given (two-way). None with fewer than 10 firms."""
    f = pd.factorize(pd.Series(firm).astype(str))[0]
    if len(set(f)) < 10:
        return None
    if date is None:
        return f
    d = pd.factorize(pd.Series(date).astype(str))[0]
    return np.column_stack([f, d]) if len(set(d)) >= 10 else f

def mean_t(df, col, pct=True, short=False):
    """Mean of df[col] with a t-stat clustered by firm and event date, plus the median (short: 'mean (t)' only)."""
    if col not in df or "cik" not in df or "day0" not in df:
        return "n = 0"
    d = df[[col, "cik", "day0"]].dropna()
    n = len(d)
    if n <= 2:
        return f"n = {n}"
    y = d[col].astype(float).to_numpy()
    fmt = (lambda v: f"{v:+.2%}") if pct else (lambda v: f"{v:+.3f}")
    if y.std() == 0:
        return f"mean {fmt(y.mean())} | n = {n:,}"
    g = cluster_groups(d["cik"], d["day0"])
    kw = {"cov_type": "cluster", "cov_kwds": {"groups": g}} if g is not None else {}
    t = float(np.asarray(sm.OLS(y, np.ones(n)).fit(**kw).tvalues)[0])
    if short:
        return f"{fmt(y.mean())} (t = {t:.1f})"
    return (f"mean {fmt(y.mean())} (t = {t:.1f}) | median {fmt(np.median(y))} | n = {n:,}, firms = {d['cik'].nunique():,}"
            + ("" if g is not None else " (fewer than 10 firms: t not clustered)"))
print("Saving to:", PROJECT_DIR)


# %% [markdown]
# ## Step 1 — Load Notebook 3 (and 3b) outputs
# Readouts: one per firm per market day (a topline readout is kept over a data update; `same_day_readouts` counts them). Stops: the registry's posting date, 8-K-filing firms leading 1–100 trials, then one event per firm per posting day.


# %% LOAD
def newest(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None

def need(path, step):
    assert os.path.exists(path), f"Missing {path}: run {step} (the full run, not the dry run) first."
    return pd.read_parquet(path)

def main_ticker(x):
    """Common-stock ticker from SEC's list, e.g. 'ACMEW;ACME' -> 'ACME' (the shortest: warrants, units, rights are longer)."""
    if isinstance(x, (list, tuple, np.ndarray)):
        opts = [str(t) for t in x]
    elif x is None or (isinstance(x, float) and np.isnan(x)):
        opts = []
    else:
        opts = re.split(r"[;,\s]+", str(x))
    opts = [t.strip().upper() for t in opts if t.strip() and t.strip().upper() not in ("NAN", "NONE", "<NA>")]
    return min(opts, key=len) if opts else None

def boolcol(df, col):
    if col not in df:
        return pd.Series(False, index=df.index)
    return df[col].astype("boolean").fillna(False).astype(bool)

def as_day(s):
    return pd.to_datetime(s, errors="coerce").dt.normalize().astype("datetime64[ns]")

def phase_level(p):
    """Highest phase in a registry phase string ('Phase 2/Phase 3', 'PHASE2|PHASE3', 'Early Phase 1') -> 1-4, else NaN."""
    nums = [int(x) for x in re.findall(r"PHASE([1-4])", str(p or "").upper().replace(" ", "").replace("_", ""))]
    return max(nums) if nums else np.nan

REASONS = [("safety", r"safety|adverse|toxicit|death|clinical hold|serious"),
           ("efficacy", r"futil|efficacy|lack of (?:benefit|response|activity)|did not meet|interim analys|endpoint|"
                        r"data monitoring|\bdsmb\b|\bdmc\b|\bidmc\b|unlikely to"),
           ("enrollment", r"enrol|recruit|accru"),
           ("business", r"business|strateg|sponsor|portfolio|priorit|fund|financ|commercial|company decision|resourc|acqui|"
                        r"merger|partner")]

def reason_class(x):
    """ClinicalTrials.gov 'why stopped' text -> safety / efficacy / enrollment / business / other / not given."""
    t = str(x or "").strip().lower()
    if not t or t in ("nan", "none", "<na>"):
        return "not given"
    return next((k for k, rx in REASONS if re.search(rx, t)), "other")

ro = need(os.path.join(FEAT_DIR, "readouts.parquet"), "Notebook 3")
reg = need(os.path.join(FEAT_DIR, "registry_events.parquet"), "Notebook 3")
uni_path = newest(os.path.join(EDGAR_DIR, "universe_*.parquet"))
assert uni_path, f"No universe_*.parquet in {EDGAR_DIR}: run Notebook 1 first."
universe = pd.read_parquet(uni_path)
universe["cik"] = pd.to_numeric(universe["cik"], errors="coerce")
universe = universe[universe["cik"].notna()]
cik2ticker = {int(c): main_ticker(t) for c, t in zip(universe["cik"], universe["tickers"])}

# ---- readouts ----
flow = {"readout 8-Ks from Notebook 3": len(ro)}
ro = ro.drop_duplicates("accession").copy()
ro["cik"] = pd.to_numeric(ro["cik"], errors="coerce")
ro["event_date"] = as_day(ro["event_date"])
ro = ro[ro["cik"].notna() & ro["event_date"].notna()].copy()
ro["cik"] = ro["cik"].astype(int)
flow["with a CIK and a market date"] = len(ro)
nb3_ticker = ro["ticker"].map(main_ticker) if "ticker" in ro else pd.Series(None, index=ro.index, dtype=object)
ro["ticker"] = ro["cik"].map(cik2ticker).astype(object).where(lambda s: s.notna(), nb3_ticker)
for col, val in [("timing", "unknown"), ("readout_kind", "unknown"), ("readout_tone", "unclear"), ("items", "")]:
    ro[col] = ro[col].fillna(val).astype(str) if col in ro else val
ro["earnings_day"] = ro["items"].str.contains(r"(?<![\d.])2\.02(?!\d)", regex=True)
ro["year"] = ro["event_date"].dt.year
ro["_rank"] = (ro["readout_kind"] != "topline_or_endpoint").astype(int)
ro = ro.sort_values(["cik", "event_date", "_rank"] + (["acceptance_et"] if "acceptance_et" in ro else []))
g = ro.groupby(["cik", "event_date"])
ro["same_day_readouts"] = g["accession"].transform("size")
ro["same_day_accessions"] = g["accession"].transform(lambda s: ";".join(s))       # all 8-K texts of that firm-day, for Notebook 5
ro["same_day_tones"] = g["readout_tone"].transform(lambda s: ";".join(sorted(set(s))))
ro = ro.drop_duplicates(["cik", "event_date"]).drop(columns="_rank").reset_index(drop=True)
flow["one per firm per market day"] = len(ro)
flow["with a ticker"] = int(ro["ticker"].notna().sum())

trial_phase = {}                                                  # trial -> highest phase today (Notebook 2)
cur_path = newest(os.path.join(PROJECT_DIR, "ctgov", "industry_trials_current*.parquet"))
if cur_path:
    cur = pd.read_parquet(cur_path)
    pcol = next((c for c in ("phases", "phase") if c in cur), None)
    if pcol:
        trial_phase = {n: v for n, v in zip(cur["nct_id"], cur[pcol].map(phase_level)) if pd.notna(v)}

def max_phase(ids):
    v = [trial_phase[n] for n in str(ids or "").split(";") if n in trial_phase]
    return max(v) if v else np.nan

ro["phase"] = ro["nct_ids"].map(max_phase) if "nct_ids" in ro else np.nan
print(f"Readouts with a linked trial phase: {ro['phase'].notna().mean():.0%}" + ("" if cur_path else " (no Notebook 2 trial file found)"))

# ---- registry stops ----
reg = reg.copy()
reg["cik"] = pd.to_numeric(reg["cik"], errors="coerce")
reg["registry_date"] = as_day(reg["registry_date"])
stops = reg[boolcol(reg, "is_stop") & reg["date_source"].eq("posted") & reg["firm_8ks_year"].fillna(0).gt(0)
            & reg["firm_lead_trials"].fillna(0).between(1, 100) & reg["cik"].notna() & reg["registry_date"].notna()].copy()
stops["cik"] = stops["cik"].astype(int)
stops["same_day_8k"] = boolcol(stops, "same_day_8k")
stops["disclosed_before"] = boolcol(stops, "disclosed_before")
stops = stops.drop_duplicates(["nct_id", "registry_date"])
stops["phase_level"] = stops["phase"].map(phase_level) if "phase" in stops else np.nan
stops["phase_level"] = stops["phase_level"].fillna(stops["nct_id"].map(trial_phase))
sflow = {"stops (posting date, 8-K filers leading 1-100 trials)": len(stops)}

SD_PATH = os.path.join(FEAT_DIR, "stop_disclosure.parquet")
sd, source = None, "Notebook 3 (an 8-K naming the trial in the 30 days before)"
if os.path.exists(SD_PATH):
    sd = pd.read_parquet(SD_PATH)
    if not {"nct_id", "registry_date", "timing", "in_sample"} <= set(sd.columns):
        print("stop_disclosure.parquet is from an older Notebook 3b (no in_sample column), so it is not used. Rerun Notebook 3b v3.1.")
        sd = None
if sd is not None:
    sd["registry_date"] = as_day(sd["registry_date"])
    sd = sd.drop_duplicates(["nct_id", "registry_date"]).set_index(["nct_id", "registry_date"])
    key = pd.MultiIndex.from_arrays([stops["nct_id"], stops["registry_date"]])
    found = np.asarray(key.isin(sd.index))
    stops["timing_3b"] = sd["timing"].reindex(key).to_numpy()
    stops["in_sample_3b"] = sd["in_sample"].reindex(key).astype("boolean").fillna(False).astype(bool).to_numpy()
    stops["exclude_reason"] = sd["exclude_reason"].reindex(key).fillna("").to_numpy() if "exclude_reason" in sd else ""
    for col in ("days_first", "first_channel", "why_stopped", "timing_detail"):
        stops[col] = sd[col].reindex(key).to_numpy() if col in sd else np.nan
    sflow["not in Notebook 3b's output (left out)"] = int((~found).sum())
    for k, v in stops.loc[found & ~stops["in_sample_3b"].to_numpy(), "exclude_reason"].value_counts().items():
        sflow[f"left out by Notebook 3b: {k or 'other'}"] = int(v)
    stops = stops[found & stops["in_sample_3b"].to_numpy()].copy()
    stops["filing_first"] = stops["timing_3b"].eq("before")
    stops["disclosure"] = stops["timing_3b"].map({"before": "filing first", "0-30 days": "filing 0-30 days after",
                                                  "31-365 days": "filing 31-365 days after"}).fillna("no filing with stop wording")
    source = "Notebook 3b (any SEC filing with stop wording, -365 to +365 days)"
else:
    stops["filing_first"] = stops["disclosed_before"]
    stops["timing_3b"] = np.where(stops["filing_first"], "before", "unknown")
    stops["disclosure"] = np.where(stops["filing_first"], "filing first", "no 8-K naming it in the 30 days before")
    for col in ("days_first", "first_channel", "why_stopped", "timing_detail"):
        stops[col] = np.nan
stops["ticker"] = stops["cik"].map(cik2ticker)
stops["reason"] = stops["why_stopped"].map(reason_class)
sflow["stops used"] = len(stops)

ev = stops.groupby(["cik", "registry_date"]).agg(
    nct_ids=("nct_id", lambda s: ";".join(sorted(s))), n_trials=("nct_id", "size"), n_first=("filing_first", "sum"),
    same_day_8k=("same_day_8k", "any"), firm_lead_trials=("firm_lead_trials", "max"), ticker=("ticker", "first"),
    disclosure=("disclosure", lambda s: ";".join(sorted(set(s)))),
    never_disclosed=("timing_3b", lambda s: bool(s.eq("none").all())),
    late_stage=("phase_level", lambda s: bool((s >= 3).any())), reason=("reason", lambda s: ";".join(sorted(set(s)))),
    material_reason=("reason", lambda s: bool(s.isin(["safety", "efficacy"]).any()))).reset_index()
ev["group"] = np.select([ev["n_first"] == ev["n_trials"], ev["n_first"] == 0], ["filing first", "registry first or only"],
                        "mixed batch")
ev["event_id"] = ev["cik"].astype(str) + "_" + ev["registry_date"].dt.strftime("%Y%m%d")
for col in ("never_disclosed", "late_stage", "material_reason"):
    ev[col] = ev[col].astype(bool)
ev["registry_detail"] = np.select([ev["group"] != "registry first or only", ev["never_disclosed"], np.full(len(ev), sd is not None)],
                                  ["", "never disclosed (within a year)", "disclosed after the registry"], "unknown (Notebook 3 split)")
sflow["stop events (one per firm per posting day)"] = len(ev)

# the day a filing first disclosed each stop (Notebook 3b): one event per firm per filing date
ds = stops[stops["days_first"].notna()].copy() if sd is not None else stops.iloc[0:0].copy()
ds["first_date"] = (ds["registry_date"] + pd.to_timedelta(pd.to_numeric(ds["days_first"]), unit="D")).astype("datetime64[ns]")
de = ds.groupby(["cik", "first_date"]).agg(
    nct_ids=("nct_id", lambda s: ";".join(sorted(s))), n_trials=("nct_id", "size"), ticker=("ticker", "first"),
    n_before=("days_first", lambda s: int((s < 0).sum())), channel=("first_channel", lambda s: ";".join(sorted(set(map(str, s))))),
    firm_lead_trials=("firm_lead_trials", "max"), late_stage=("phase_level", lambda s: bool((s >= 3).any())),
    material_reason=("reason", lambda s: bool(s.isin(["safety", "efficacy"]).any()))).reset_index()
for col in ("late_stage", "material_reason"):
    de[col] = de[col].astype(bool)
de["when"] = np.select([de["n_before"] == de["n_trials"], de["n_before"] == 0], ["before the registry", "after the registry"], "mixed")
de["event_id"] = "d" + de["cik"].astype(str) + "_" + de["first_date"].dt.strftime("%Y%m%d")
sflow["first-disclosure events (one per firm per filing date)"] = len(de)

tickers = sorted(set(ro["ticker"].dropna()) | set(ev["ticker"].dropna()) | set(de["ticker"].dropna()))
if DRY_RUN:
    tickers = tickers[:DRY_RUN_TICKERS]
print("Readouts:", " -> ".join(f"{k} {v:,}" for k, v in flow.items()))
print("Stops:   ", " -> ".join(f"{k} {v:,}" for k, v in sflow.items()))
print(f"Stop disclosure from: {source}\nTickers: {len(tickers):,}")


# %% [markdown]
# ## Step 2 — Daily prices from Yahoo
# Batches of 50 tickers, one file per ticker. `_missing.json` lists tickers Yahoo returned nothing for and how often they were tried (up to three different days). Files are downloaded again when they stop before the data an event needs.


# %% PRICES
MISSING = os.path.join(PRICE_DIR, "_missing.json")
missing = json.load(open(MISSING)) if os.path.exists(MISSING) else {}
today = str(TODAY.date())

def ysym(t):
    return t.replace(".", "-").replace("/", "-").upper()

def ppath(t):
    return os.path.join(PRICE_DIR, f"{t}.parquet")

def read_cache(t):
    try:
        df = pd.read_parquet(ppath(t))
    except Exception:
        return None
    return df if len(df) and {"date", "Close"} <= set(df.columns) else None

def file_day(p):
    return pd.Timestamp(dt.datetime.fromtimestamp(os.path.getmtime(p)).date())

last_event = pd.concat([ro[["ticker", "event_date"]],
                        ev[["ticker", "registry_date"]].rename(columns={"registry_date": "event_date"}),
                        de[["ticker", "first_date"]].rename(columns={"first_date": "event_date"})]).groupby("ticker")["event_date"].max()

def needs_download(t):
    p = ppath(t)
    df = read_cache(t) if os.path.exists(p) else None
    if os.path.exists(p) and df is None:                      # empty file (an older 'not on Yahoo' marker) or damaged
        missing.setdefault(t, {"tries": 1, "last": str(file_day(p).date())})
        os.remove(p)
    if df is None:
        m = missing.get(t)
        return m is None or (m["tries"] < 3 and m["last"] != today)
    have = pd.to_datetime(df["date"]).max()
    have = have.tz_localize(None) if have.tzinfo else have
    want = TODAY - pd.Timedelta(days=1)
    if t != BENCH:
        want = min(want, last_event.get(t, TODAY) + pd.Timedelta(days=45))
    return have < want - pd.Timedelta(days=7) and file_day(p) < want   # stops before what an event needs, and not fetched since

def save_missing():
    with open(MISSING + ".tmp", "w") as f:
        json.dump(missing, f)
    os.replace(MISSING + ".tmp", MISSING)

todo = [t for t in [BENCH] + tickers if needs_download(t)]
print(f"Tickers to download or refresh: {len(todo):,} of {len(tickers) + 1:,}")
for k in range(0, len(todo), 50):
    chunk = todo[k:k + 50]
    got = yf_download([ysym(t) for t in chunk])
    if not got and len(chunk) > 1:                                # nothing at all: wait, retry, then check Yahoo answers
        time.sleep(60)
        got = yf_download([ysym(t) for t in chunk])
        if not got and not yf_download([ysym(BENCH)]):
            save_missing()
            raise RuntimeError("Yahoo is not answering (rate limit or outage). Wait an hour and rerun this cell; "
                               "what was downloaded is saved.")
    for t in chunk:
        df = got.get(ysym(t))
        if df is not None:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
            df.reset_index().to_parquet(ppath(t) + ".tmp", index=False)
            os.replace(ppath(t) + ".tmp", ppath(t))
            missing.pop(t, None)
        elif not os.path.exists(ppath(t)):                        # a refresh that comes back empty keeps the old file
            missing[t] = {"tries": missing.get(t, {"tries": 0})["tries"] + 1, "last": today}
    save_missing()
    print(f"  {min(k + 50, len(todo)):,}/{len(todo):,} | with data: {sum(ysym(t) in got for t in chunk)}/{len(chunk)}")
    time.sleep(2)

def load_close(t):
    df = read_cache(t)
    if df is None:
        return None
    d = pd.to_datetime(df["date"], errors="coerce")
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    df = df.assign(date=d.dt.normalize().astype("datetime64[ns]")).dropna(subset=["date"])
    df = df.drop_duplicates("date", keep="last").set_index("date").sort_index()
    close = pd.to_numeric(df["Close"], errors="coerce")
    vol = pd.to_numeric(df["Volume"], errors="coerce") if "Volume" in df else pd.Series(np.nan, index=df.index)
    return close, vol

b = load_close(BENCH)
assert b is not None, f"No prices for the benchmark {BENCH}: rerun this cell later."
bclose = b[0].where(b[0] > 0).dropna()
CAL = bclose.index                                                # trading calendar
bad_prints = {}

def stock_frame(t):
    """Abnormal returns on the trading calendar. Each return runs from the stock's previous valid close (`span` = trading
    days it covers) and XBI's return is taken over the same days; NaN on days without a valid close."""
    x = load_close(t)
    if x is None:
        return None
    close, vol = x
    close = close[close.index.isin(CAL)]
    vol = vol.reindex(close.index)
    close = close.where(close > 0)
    close = close.where(~((vol == 0) & (close == close.shift(1))))        # no trade: Yahoo repeats the last close
    c = close.dropna()
    if len(c) < 2:
        return None
    lr = np.log(c / c.shift(1))
    nxt = lr.shift(-1)
    bad = ((lr.abs() > np.log(3)) & (nxt.abs() > np.log(3)) & (np.sign(lr) == -np.sign(nxt))
           & ((lr + nxt).abs() < np.log(1.25)))                    # one-day bad print: 3x away and back the next day
    if bad.any():
        bad_prints[t] = int(bad.sum())
        c = c[~bad]
    pos = pd.Series(CAL.get_indexer(c.index), index=c.index)
    span = pos.diff()
    bc = bclose.reindex(c.index)
    ar = ((c / c.shift(1) - 1) - (bc / bc.shift(1) - 1)).where(span <= MAX_GAP)
    out = pd.DataFrame({"ar": ar.reindex(CAL), "span": span.reindex(CAL), "volume": vol.reindex(CAL), "close": c.reindex(CAL)},
                       index=CAL)
    out["traded"] = out.index.isin(c.index)
    return out

PX = {t: stock_frame(t) for t in tickers}
PX = {t: f for t, f in PX.items() if f is not None and f["traded"].sum() > 30}
gave_up = sorted(t for t in tickers if missing.get(t, {}).get("tries", 0) >= 3)
print(f"Tickers with prices: {len(PX):,} of {len(tickers):,} | Yahoo has nothing after 3 tries: {len(gave_up):,} | "
      f"still to retry on another day: {sum(1 for t in tickers if t in missing and t not in gave_up):,}")
print(f"One-day bad prints removed: {sum(bad_prints.values()):,} in {len(bad_prints):,} tickers | calendar {CAL[0]:%Y-%m-%d} to {CAL[-1]:%Y-%m-%d}")


# %% [markdown]
# ## Step 3 — Event windows
# Day 0 is the first trading day on or after the event: the 8-K's market day for readouts (Notebook 3 already moved after-close filings to the next session), the posting date for stops. When a window ends on a day the stock didn't trade (a halt), the return that closes the gap is added.


# %% EVENTS
PATH_DAYS = range(-10, 21)
E0, E1 = EST_WINDOW

def event_stats(t, day):
    px = PX.get(t)
    if px is None:
        return None, None
    i0 = int(CAL.searchsorted(day))
    if i0 >= len(CAL) or (CAL[i0] - day).days > 7:
        return None, None
    ar, span, vol, live, close = (px[c].to_numpy() for c in ("ar", "span", "volume", "traded", "close"))
    n = len(CAL)

    def seg(x, a, b):
        lo, hi = max(i0 + a, 0), min(i0 + b + 1, n)
        return x[lo:hi] if lo < hi else x[:0]

    if not (seg(live, -5, -1).any() and seg(live, 0, 3).any()):
        return None, None                                         # not trading around the event (not listed yet, delisted, long halt)

    def car(a, b, need=1, close_gap=True):
        lo, hi = max(i0 + a, 0), min(i0 + b, n - 1)
        if hi < lo:
            return np.nan
        x = ar[lo:hi + 1]
        tot, k = float(np.nansum(x)), int(np.isfinite(x).sum())
        if close_gap and not live[hi]:                                          # window ends inside a halt or gap: add the return that closes it
            nxt = np.flatnonzero(live[hi + 1:hi + 1 + MAX_GAP])
            if len(nxt) and np.isfinite(ar[hi + 1 + nxt[0]]):
                tot, k = tot + ar[hi + 1 + nxt[0]], k + 1
        return tot if k >= need else np.nan

    est, est_span = seg(ar, E0, E1), seg(span, E0, E1)
    est = est[np.isfinite(est) & (est_span == 1)]
    post = seg(ar, 2, 21)
    vb = seg(vol, E0, E1)
    vb = vb[np.isfinite(vb)]
    ve = seg(vol, -1, 1)
    ve = ve[np.isfinite(ve)]
    base_v = np.median(vb) if len(vb) >= 20 else np.nan
    w = seg(ar, -1, 1)
    cb, vb2 = seg(close, E0, E1), seg(vol, E0, E1)
    dv = cb * vb2
    dv = dv[np.isfinite(dv) & (dv > 0)]
    pre_close = seg(close, -10, -2)
    pre_close = pre_close[np.isfinite(pre_close)]
    s = {"day0": CAL[i0], "ar_m1": ar[i0 - 1] if i0 >= 1 else np.nan, "ar0": ar[i0], "span0": span[i0],
         "car_1_1": car(-1, 1), "car_1_0": car(-1, 0), "car01": car(0, 1), "car05": car(0, 5, need=3),
         "pre_vol": est.std(ddof=1) if len(est) >= MIN_EST_DAYS else np.nan,
         "post_vol": np.nanstd(post, ddof=1) if np.isfinite(post).sum() >= 15 else np.nan,
         "abn_volume": ve.mean() / base_v if len(ve) and base_v > 0 else np.nan,
         "max_abs_ar": float(np.nanmax(np.abs(w))) if np.isfinite(w).any() else np.nan,
         "car_m10_m2": car(-10, -2, need=5, close_gap=False), "car_2_20": car(2, 20, need=10),
         "price_pre": float(pre_close[-1]) if len(pre_close) else np.nan,
         "log_dollar_vol": float(np.log(np.median(dv))) if len(dv) >= 20 else np.nan,
         "days_listed": int(seg(live, -250, -1).sum()),
         "post_complete": bool(i0 + 21 < n and seg(live, 15, 21).any())}
    s["vol_ratio"] = s["post_vol"] / s["pre_vol"] if s["pre_vol"] > 0 else np.nan
    path = {k: ar[i0 + k] for k in PATH_DAYS if 0 <= i0 + k < n}
    return s, path

def run_events(ids, tks, days, kind):
    rows, paths = [], []
    for eid, t, d in zip(ids, tks, days):
        if not isinstance(t, str):
            continue
        s, p = event_stats(t, d)
        if s:
            rows.append({"event_id": eid, **s})
            paths += [{"kind": kind, "event_id": eid, "t": k, "ar": v} for k, v in p.items()]
    return pd.DataFrame(rows), pd.DataFrame(paths, columns=["kind", "event_id", "t", "ar"])

rs, rp = run_events(ro["accession"], ro["ticker"], ro["event_date"], "readout")
re_ = ro.merge(rs, left_on="accession", right_on="event_id", how="left") if len(rs) else ro.assign(car_1_1=np.nan)
re_["has_prices"] = re_["car_1_1"].notna()
assert re_["has_prices"].any(), "No readout has prices around its date: check Step 2's output (Yahoo or the benchmark)."
re_["abs_car"] = re_["car_1_1"].abs()
re_["abs_ar0"] = re_["ar0"].abs()
re_["_pos"] = CAL.searchsorted(re_["event_date"])
re_ = re_.sort_values(["cik", "_pos"]).reset_index(drop=True)
gp = re_.groupby("cik")["_pos"]
re_["overlap"] = (gp.diff() <= 5) | (gp.diff(-1).abs() <= 5)      # another readout from the firm within 5 trading days
re_["recent_ipo"] = re_["days_listed"].lt(60)
re_["extreme"] = re_["max_abs_ar"].gt(2.0)
re_["clean"] = re_["has_prices"] & ~re_["overlap"] & ~re_["earnings_day"] & ~re_["extreme"] & ~re_["recent_ipo"]

rpos = re_.groupby("cik")["_pos"].apply(np.asarray).to_dict()
ev_pos = CAL.searchsorted(ev["registry_date"])
ev["near_readout"] = [bool(len(rpos.get(c, [])) and np.abs(rpos[c] - p).min() <= 1) for c, p in zip(ev["cik"], ev_pos)]
ss, sp = run_events(ev["event_id"], ev["ticker"], ev["registry_date"], "stop")
se = ev.merge(ss, on="event_id", how="left") if len(ss) else ev.assign(car01=np.nan)
se["has_prices"] = se["car01"].notna()
de_pos = CAL.searchsorted(de["first_date"])
de["near_readout"] = [bool(len(rpos.get(c, [])) and np.abs(rpos[c] - p).min() <= 1) for c, p in zip(de["cik"], de_pos)]
ds_, dp = run_events(de["event_id"], de["ticker"], de["first_date"], "disclosure")
dse = de.merge(ds_, on="event_id", how="left") if len(ds_) else de.assign(car_1_1=np.nan)
dse["has_prices"] = dse["car_1_1"].notna()

re_.drop(columns="_pos").to_parquet(os.path.join(FEAT_DIR, f"readout_events{SUFFIX}.parquet"), index=False)
se.to_parquet(os.path.join(FEAT_DIR, f"stop_events{SUFFIX}.parquet"), index=False)
dse.to_parquet(os.path.join(FEAT_DIR, f"stop_disclosure_events{SUFFIX}.parquet"), index=False)
paths = pd.concat([rp, sp, dp], ignore_index=True)
paths.to_parquet(os.path.join(FEAT_DIR, f"event_paths{SUFFIX}.parquet"), index=False)

in_run = re_["ticker"].isin(tickers)
print(f"Readouts with prices: {re_['has_prices'].sum():,} of {in_run.sum():,} with a ticker in this run "
      f"({re_.loc[in_run, 'has_prices'].mean():.0%}); no ticker at all: {re_['ticker'].isna().sum():,}")
h = re_[re_["has_prices"]]
print(f"  flagged: other readout within 5 days {h['overlap'].mean():.0%} | earnings 8-K {h['earnings_day'].mean():.0%} | "
      f"listed < 60 days {h['recent_ipo'].mean():.0%} | |AR| > 200% in days -1..+1 {h['extreme'].sum():,} | "
      f"day 0 return spans a gap {(h['span0'] > 1).mean():.1%} | post window incomplete {(~h['post_complete'].astype(bool)).mean():.0%}")
print(f"Stop events with prices: {se['has_prices'].sum():,} of {len(se):,} | near a readout: {se['near_readout'].sum():,} | "
      f"8-K within a day: {se['same_day_8k'].sum():,}")
print(f"First-disclosure events with prices: {dse['has_prices'].sum():,} of {len(dse):,} | near a readout: {dse['near_readout'].sum():,}")


# %% [markdown]
# ## Step 4 — OCC options volume around recent readouts
# Daily contracts per stock from OCC's volume query (it keeps about two years). Shares the cache with the earlier scratch cell (`occ/stock_daily_volume.csv`). Safe to stop and rerun: it continues where it stopped.


# %% OCC
DAILY = os.path.join(OCC_DIR, "stock_daily_volume.csv")
DONE = os.path.join(OCC_DIR, "stock_daily_done.txt")
CHECKED = os.path.join(OCC_DIR, "cache_v2_checked.txt")
PROBE = os.path.join(OCC_DIR, "oldest_day.json")
ACCT = {"C": "customer", "F": "firm", "M": "market_maker"}
SIDE = {"C": "call", "P": "put"}
OCC_COLS = ["ticker", "date"] + [f"{s}_{a}" for s in SIDE.values() for a in ACCT.values()] + ["contracts"]

def occ_day(ticker, day):
    """One stock-day: a row of contracts, {} when OCC has no record (no options traded), None when OCC gave no proper answer."""
    t = occ_fetch(dict(reportDate=day.strftime("%Y%m%d"), format="csv", volumeQueryType="O", symbolType="U", symbol=ticker,
                       reportType="D", productKind="OSTK", accountType="ALL", porc="BOTH"))
    if t is None:
        return None
    if not t.startswith("quantity"):
        return {}
    try:
        df = pd.read_csv(io.StringIO(t), index_col=False, dtype=str)
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
        out = {"ticker": ticker, "date": day.isoformat()}
        for (side, acct), q in df.groupby(["porc", "actype"])["quantity"].sum().items():
            out[f"{SIDE.get(side, side)}_{ACCT.get(acct, acct)}"] = int(q)
    except Exception:
        return None
    out["contracts"] = sum(v for k, v in out.items() if k not in ("ticker", "date")) / 2   # each trade counted on both sides
    return out if out["contracts"] > 0 else {}

def save_row(row):
    pd.DataFrame([row]).reindex(columns=OCC_COLS).to_csv(DAILY, mode="a", header=not os.path.exists(DAILY), index=False)

def read_daily():
    if not os.path.exists(DAILY):
        return pd.Series(dtype=float)
    v = pd.read_csv(DAILY, dtype=str, on_bad_lines="skip")
    if not {"ticker", "date", "contracts"} <= set(v.columns):
        return pd.Series(dtype=float)
    v["contracts"] = pd.to_numeric(v["contracts"], errors="coerce")
    v = v.dropna(subset=["ticker", "date", "contracts"]).drop_duplicates(["ticker", "date"], keep="last")
    return v.set_index(["ticker", "date"])["contracts"]

def fix_old_cache():
    """Once per cache made by an earlier version. The first versions appended rows whose columns didn't match the header
    (repaired here: contracts is always half the sum of the side counts written before it), and every earlier version
    could save a day as 'no options traded' when the request failed, the day was outside OCC's window, or OCC hadn't
    published it yet. Days with volume are kept; every 'no options' day is asked again."""
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    rows, bad = [], 0
    if os.path.exists(DAILY):
        with open(DAILY, newline="") as f:
            rd = csv.reader(f)
            header = next(rd, [])
            for fl in rd:
                if len(fl) < 3:
                    continue
                if header == OCC_COLS and len(fl) == len(OCC_COLS):           # this version's layout: keep the row
                    row = dict(zip(OCC_COLS, fl))
                    if pd.notna(pd.to_numeric(row["contracts"], errors="coerce")):
                        rows.append(row)
                        continue
                num = [0.0 if x in ("", "nan") else pd.to_numeric(x, errors="coerce") for x in fl[2:]]
                for j in (len(num) - 1, len(num) - 2):              # contracts is last, or before n_exchanges (scratch cell)
                    if j >= 1 and num[j] > 0 and abs(num[j] - sum(num[:j]) / 2) < 1e-6:
                        rows.append({"ticker": fl[0], "date": fl[1], "contracts": num[j]})
                        break
                else:
                    bad += 1
        os.replace(DAILY, f"{DAILY}.old_{stamp}")
    pd.DataFrame(rows, columns=OCC_COLS).drop_duplicates(["ticker", "date"], keep="last").to_csv(DAILY, index=False)
    have = {f"{r['ticker']} {r['date'].replace('-', '')}" for r in rows}
    asked = set()
    if os.path.exists(DONE):
        asked = {k.strip() for k in open(DONE) if k.strip()}
        os.replace(DONE, f"{DONE}.old_{stamp}")
    with open(DONE, "w") as f:
        f.write("".join(k + "\n" for k in sorted(have)))
    print(f"OCC cache from an earlier version checked: {len(have):,} stock-days with volume kept | {len(asked - have):,} days "
          f"saved as 'no options' will be asked again once | {bad} unreadable rows (asked again)")
    with open(CHECKED, "w") as f:
        f.write(f"v2 {stamp}\n")

def occ_window_start():
    """First day OCC still has (about two years back): AAPL is asked every 5th trading day from two years back
    until data appears. Checked once a day."""
    if os.path.exists(PROBE):
        p = json.load(open(PROBE))
        if p.get("checked") == str(TODAY.date()):
            return pd.Timestamp(p["oldest"])
    cands = CAL[(CAL >= TODAY - pd.DateOffset(years=2)) & (CAL <= TODAY - pd.Timedelta(days=30))]
    for day in cands[::5][:40]:
        row = occ_day("AAPL", day.date())
        if row is None:
            raise RuntimeError("OCC is not answering: rerun this cell later.")
        if row:
            with open(PROBE, "w") as f:
                json.dump({"checked": str(TODAY.date()), "oldest": str(day.date())}, f)
            return day
    raise RuntimeError("OCC returned no AAPL volume in the last two years: its query may have changed. Check the OCC volume query.")

occ_events = pd.DataFrame()
if RUN_OCC:
    marker = open(CHECKED).read().strip() if os.path.exists(CHECKED) else ""
    if not marker.startswith("v2"):
        if os.path.exists(DAILY) or os.path.exists(DONE):
            fix_old_cache()
        else:
            with open(CHECKED, "w") as f:
                f.write("v2 new\n")
    for p in (DAILY, DONE):
        ensure_newline(p)
    oldest = occ_window_start()
    latest = CAL[CAL <= TODAY - pd.Timedelta(days=3)][-1]          # OCC may not have published the last few days
    lst = re_[re_["has_prices"] & re_["ticker"].notna() & (re_["day0"] >= oldest - pd.Timedelta(days=21))]
    lst = lst.sort_values("day0", ascending=False)
    if DRY_RUN:
        lst = lst.head(DRY_RUN_OCC)
    print(f"OCC window from {oldest:%Y-%m-%d} to {latest:%Y-%m-%d} | readouts: {len(lst):,}")
    done = {k.strip() for k in open(DONE)} if os.path.exists(DONE) else set()
    t0, n_req, n_fail, fail_run = time.time(), 0, 0, 0
    for k, (tk, d) in enumerate(zip(lst["ticker"], lst["day0"]), 1):
        i0 = CAL.searchsorted(d)
        days = [x.date() for x in CAL[max(0, i0 - OCC_BEFORE):i0 + OCC_AFTER + 1] if oldest <= x <= latest]
        failed = False
        for day in days:
            key = f"{tk} {day:%Y%m%d}"
            if key in done:
                continue
            row = occ_day(tk, day)
            n_req += 1
            if row is None:                                       # no proper answer: not saved, asked again on the next run
                failed = True
                break
            if row:
                save_row(row)                                     # volume saved before the day is marked as asked
            done.add(key)
            with open(DONE, "a") as f:
                f.write(key + "\n")
        n_fail += failed
        fail_run = fail_run + 1 if failed else 0
        if fail_run >= 5:
            raise RuntimeError("OCC gave no proper answer for 5 readouts in a row (down or blocking). Rerun this cell later; "
                               "everything so far is saved.")
        if k % 25 == 0 or k == len(lst):
            print(f"  readout {k}/{len(lst)} | {n_req:,} requests | about {(time.time() - t0) / k * (len(lst) - k) / 60:.0f} min left")
    if n_fail:
        print(f"{n_fail:,} readouts got no proper answer for some day: rerun this cell later to fill them")

    vol = read_daily()
    rows = []
    for acc, tk, d in zip(lst["accession"], lst["ticker"], lst["day0"]):
        i0 = CAL.searchsorted(d)
        rel = {}
        for k in range(-OCC_BEFORE, OCC_AFTER + 1):
            if 0 <= i0 + k < len(CAL) and oldest <= CAL[i0 + k] <= latest:
                day = CAL[i0 + k]
                if f"{tk} {day:%Y%m%d}" in done:                   # asked: no record means no options traded
                    rel[k] = float(vol.get((tk, f"{day:%Y-%m-%d}"), 0.0))
        rows.append({"accession": acc, "ticker": tk, **{f"c{k:+d}": v for k, v in rel.items()}})
    occ_events = pd.DataFrame(rows)
    if len(occ_events):
        def window_mean(ks, need):                                # mean contracts over days ks, if at least `need` were asked
            cols = [f"c{k:+d}" for k in ks if f"c{k:+d}" in occ_events]
            if not cols:
                return pd.Series(np.nan, index=occ_events.index)
            return occ_events[cols].mean(axis=1).where(occ_events[cols].notna().sum(axis=1) >= need)
        occ_events["opt_base"] = window_mean(range(-10, -5), 3)
        occ_events["opt_pre"] = window_mean(range(-5, -1), 3)    # ends at day -2: day -1 is in the event window
        occ_events["opt_event"] = window_mean(range(-1, 2), 2)
        occ_events["log_opt_pre"] = np.log1p(occ_events["opt_pre"]) - np.log1p(occ_events["opt_base"])
        occ_events["log_opt_event"] = np.log1p(occ_events["opt_event"]) - np.log1p(occ_events["opt_base"])
        occ_events["opt_ok"] = occ_events["opt_base"] >= OCC_MIN_BASE
        occ_events.to_parquet(os.path.join(FEAT_DIR, f"occ_events{SUFFIX}.parquet"), index=False)
        print(f"OCC: {len(occ_events):,} readouts | base window complete: {occ_events['opt_base'].notna().mean():.0%} | "
              f"options traded on days -10..-6: {(occ_events['opt_base'] > 0).mean():.0%} | "
              f"at least {OCC_MIN_BASE} contracts/day: {occ_events['opt_ok'].mean():.0%}")
else:
    print("OCC step skipped (RUN_OCC = False)")


# %% [markdown]
# ## Step 5 — First tests and figures
# Check this cell's output. t-stats are clustered by firm and by event date. Pilot numbers: survivorship (delisted firms missing from Yahoo) and the screening-rule readout labels are the main caveats.


# %% TESTS
print(f"Run: {'DRY RUN' if DRY_RUN else 'FULL'} | tickers with prices {len(PX):,} of {len(tickers):,}")

print("\n=== Coverage (survivorship check): readouts by year ===")
c = re_[re_["ticker"].isin(tickers) | re_["ticker"].isna()]
cov = c.groupby("year").agg(readouts=("accession", "size"), no_ticker=("ticker", lambda s: s.isna().mean()),
                            with_prices=("has_prices", "mean"))
print(cov.round(2).to_string())

r = re_[re_["has_prices"]].copy()
r["log_pre_vol"] = np.log(r["pre_vol"].where(r["pre_vol"] > 0))
top = r[r["readout_kind"] == "topline_or_endpoint"]
TONES = ["positive", "negative", "mixed", "unclear"]
print("\n=== Readouts: abnormal return, days -1..+1 (stock - XBI) ===")
for tone in TONES:
    print(f"  topline/endpoint, {tone:8s}: {mean_t(top[top['readout_tone'] == tone], 'car_1_1')}")
print(f"  data updates (all)          : {mean_t(r[r['readout_kind'] == 'data_update'], 'car_1_1')}")
print(f"Median |car(-1,+1)|: topline/endpoint {top['abs_car'].median():.1%} | data update "
      f"{r.loc[r['readout_kind'] == 'data_update', 'abs_car'].median():.1%} | median abnormal volume (days -1..+1) "
      f"{r['abn_volume'].median():.1f}x | volatility after vs before (median ratio) {r['vol_ratio'].median():.2f}")

print("\n=== Timing check: topline readouts, mean abnormal return by 8-K timing ===")
tt = top[top["readout_tone"].isin(["positive", "negative"])]
if len(tt):
    print(tt.groupby(["readout_tone", "timing"]).agg(n=("car_1_1", "size"), day_m1=("ar_m1", "mean"), day0=("ar0", "mean"),
                                                     car_1_1=("car_1_1", "mean")).round(3).to_string())

print("\n=== Topline readouts before, at and after the event (mean, t clustered) ===")
for tone in TONES:
    x = top[top["readout_tone"] == tone]
    if len(x) > 2:
        print(f"  {tone:8s} n = {len(x):,} | days -10..-2 {mean_t(x, 'car_m10_m2', short=True)} | days -1..+1 "
              f"{mean_t(x, 'car_1_1', short=True)} | days +2..+20 {mean_t(x, 'car_2_20', short=True)}")

print("\n=== Robustness: topline readouts ===")
checks = [("pre-open 8-Ks only, days -1..+1", top[top["timing"] == "pre_open"], "car_1_1"),
          ("clean events only, days -1..+1", top[top["clean"]], "car_1_1"),
          ("all, days -1..0", top, "car_1_0"),
          ("all, day 0 only", top, "ar0")]
for label, sub, col in checks:
    for tone in ["positive", "negative"]:
        print(f"  {label:32s} {tone:8s}: {mean_t(sub[sub['readout_tone'] == tone], col)}")
print("  (clean = no other readout from the firm within 5 trading days, no earnings 8-K, listed 60+ days, |AR| <= 200%)")

d = r.copy()
d["y"] = winsor(d["abs_car"])
d["earnings_day"] = d["earnings_day"].astype(int)
d = d.dropna(subset=["y", "readout_kind", "readout_tone", "log_pre_vol", "log_dollar_vol", "timing", "earnings_day", "year",
                     "cik", "day0"])

def cat(col, ref):
    return f"C({col}, Treatment(reference={ref!r}))" if ref in set(d[col]) else f"C({col})"

if len(d) > 50 and d["cik"].nunique() >= 10:
    f = (f"y ~ {cat('readout_kind', 'data_update')} + {cat('readout_tone', 'unclear')} + log_pre_vol + log_dollar_vol + "
         f"{cat('timing', 'pre_open')} + earnings_day + C(year)")
    g = cluster_groups(d["cik"], d["day0"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.ols(f, d).fit(**({"cov_type": "cluster", "cov_kwds": {"groups": g}} if g is not None else {}))
    tab = pd.DataFrame({"coef": m.params, "se": m.bse, "p": m.pvalues})
    tab.index = [re.sub(r"C\((\w+)(?:, Treatment\(reference='[^']*'\))?\)\[T\.([^\]]+)\]", r"\1 = \2", i) for i in tab.index]
    print(f"\n=== |car(-1,+1)| (clipped at {WINSOR:.0%}/{1 - WINSOR:.0%}) on readout labels, clustered by firm and date ===")
    print(tab.loc[[i for i in tab.index if not i.startswith("year = ")]].round(4).to_string())
    print(f"R2 = {m.rsquared:.3f}, n = {int(m.nobs):,} | reference levels: kind = data_update, tone = unclear, timing = pre_open "
          f"(where present) | year fixed effects")

print(f"\n=== Registry stops (posting day) | disclosure from {source} ===")
s = se[se["has_prices"]]
for grp in ["registry first or only", "filing first", "mixed batch"]:
    x = s[s["group"] == grp]
    if len(x):
        print(f"  {grp:24s} car01   {mean_t(x, 'car01')}")
        print(f"  {'':24s} car_1_1 {mean_t(x, 'car_1_1')}")
        print(f"  {'':24s} car05   {mean_t(x, 'car05')}")
x = s[(s["group"] == "registry first or only") & ~s["same_day_8k"] & ~s["near_readout"] & (s["firm_lead_trials"] <= 20)]
print(f"  registry first or only, no 8-K or readout within a day, firms with 1-20 trials:\n"
      f"  {'':24s} car01   {mean_t(x, 'car01')}\n  {'':24s} car05   {mean_t(x, 'car05')}")
if sd is not None:
    for det in ["never disclosed (within a year)", "disclosed after the registry"]:
        x = s[s["registry_detail"] == det]
        print(f"  registry first: {det:31s} car01 {mean_t(x, 'car01', short=True)} | car05 {mean_t(x, 'car05', short=True)} | n = {len(x):,}")
m = s[s["late_stage"] & s["material_reason"] & (s["firm_lead_trials"] <= 20)]
print("  Phase 3 stops for safety or efficacy, firms with 1-20 trials:")
for grp in ["registry first or only", "filing first"]:
    x = m[m["group"] == grp]
    print(f"  {grp:24s} car01   {mean_t(x, 'car01')}")
print("  why stopped (stop events with prices): " + ", ".join(f"{k} {v:,}" for k, v in
      s["reason"].str.split(";").explode().value_counts().items()))

if len(dse) and dse["has_prices"].any():
    q = dse[dse["has_prices"] & ~dse["near_readout"]]
    print("\n=== Stops: reaction around the filing that first discloses the stop (days -1..+1; filings next to a readout left out) ===")
    for w in ["before the registry", "after the registry"]:
        print(f"  first filing {w:20s} {mean_t(q[q['when'] == w], 'car_1_1')}")
    print(f"  first filing is an 8-K                {mean_t(q[q['channel'].str.contains('8-K')], 'car_1_1')}")
    print(f"  first filing is a 10-Q/10-K or other  {mean_t(q[~q['channel'].str.contains('8-K')], 'car_1_1')}")
    x = q[q["late_stage"] & q["material_reason"] & (q["firm_lead_trials"] <= 20)]
    print(f"  Phase 3, safety or efficacy, 1-20 trials {mean_t(x, 'car_1_1')}")
    print(f"  next to a readout (same firm, within a day): {mean_t(dse[dse['has_prices'] & dse['near_readout']], 'car_1_1')}")

if len(occ_events):
    o = occ_events.merge(r[["accession", "cik", "day0", "abs_car", "car_1_1"]], on="accession")
    ok = o[o["opt_ok"]]
    print(f"\n=== Options volume (OCC): {len(o):,} readouts | at least {OCC_MIN_BASE} contracts/day on days -10..-6: {len(ok):,} ===")
    print(f"  log(contracts days -5..-2 / days -10..-6): {mean_t(ok, 'log_opt_pre', pct=False)}")
    print(f"  log(contracts days -1..+1 / days -10..-6): {mean_t(ok, 'log_opt_event', pct=False)}")
    x = ok.dropna(subset=["log_opt_pre", "abs_car"])
    if len(x) > 10:
        g = cluster_groups(x["cik"])
        fit = smf.ols("abs_car ~ log_opt_pre", x).fit(**({"cov_type": "cluster", "cov_kwds": {"groups": g}} if g is not None else {}))
        print(f"  build-up before the readout vs |car(-1,+1)|: correlation {x[['log_opt_pre', 'abs_car']].corr().iloc[0, 1]:.2f} | "
              f"slope {fit.params['log_opt_pre']:.3f} (t = {fit.tvalues['log_opt_pre']:.1f}"
              f"{'' if g is not None else ', not clustered'}) | n = {len(x):,}")

# ---- figures ----
import matplotlib.pyplot as plt
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "legend.frameon": False, "axes.axisbelow": True})

def car_path(kind, ids, t0=-10, t1=20):
    """Mean cumulative abnormal return (%) with each day's returns clipped at the WINSOR tails across events."""
    p = paths[(paths["kind"] == kind) & paths["event_id"].isin(set(ids)) & paths["t"].between(t0, t1)]
    if p.empty:
        return None, 0
    w = p.pivot_table(index="event_id", columns="t", values="ar", aggfunc="first").apply(winsor)
    return w.fillna(0).cumsum(axis=1).mean() * 100, len(w)

def finish(ax, xlabel, title, name):
    ax.axvline(0, color=INK2, linewidth=1, linestyle="--")
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean cumulative abnormal return (%)")
    ax.set_title(title, loc="left")
    ax.legend(fontsize=9, ncol=2)
    ax.figure.savefig(os.path.join(FIG_DIR, f"{name}{SUFFIX}.png"), dpi=200, bbox_inches="tight")
    plt.show()

fig, ax = plt.subplots(figsize=(8, 3.6))
for tone, col in zip(TONES, COL):
    m, n = car_path("readout", top.loc[top["readout_tone"] == tone, "accession"])
    if m is not None:
        ax.plot(m.index, m.values, color=col, linewidth=2, label=f"{tone} (n = {n:,})")
finish(ax, "Trading days from the readout 8-K", "Topline / primary-endpoint readouts: stock minus XBI", "fig6_readout_car")

fig, ax = plt.subplots(figsize=(8, 3.6))
for tone, col in zip(["positive", "negative"], COL):
    for timing, ls in [("pre_open", "-"), ("after_close", "--")]:
        m, n = car_path("readout", top.loc[(top["readout_tone"] == tone) & (top["timing"] == timing), "accession"])
        if m is not None:
            ax.plot(m.index, m.values, color=col, linewidth=2, linestyle=ls, label=f"{tone}, {timing.replace('_', '-')} 8-K (n = {n:,})")
finish(ax, "Trading days from the readout 8-K", "Topline readouts by 8-K timing: after-close 8-Ks move on day -1",
       "fig6b_readout_car_by_timing")

fig, ax = plt.subplots(figsize=(8, 3.6))
for grp, col in zip(["registry first or only", "filing first"], COL):
    m, n = car_path("stop", s.loc[s["group"] == grp, "event_id"])
    if m is not None:
        ax.plot(m.index, m.values, color=col, linewidth=2, label=f"{grp} (n = {n:,})")
finish(ax, "Trading days from the registry posting", "Trial stops posted on ClinicalTrials.gov: stock minus XBI", "fig7_stop_car")

if len(occ_events) and occ_events["opt_ok"].any():
    q = occ_events[occ_events["opt_ok"]]
    cols = [f"c{k:+d}" for k in range(-OCC_BEFORE, OCC_AFTER + 1) if f"c{k:+d}" in q]
    lq = np.log1p(q[cols])
    rel = lq.sub(lq[[c for c in cols if -10 <= int(c[1:]) <= -6]].mean(axis=1), axis=0)   # base days average to 0
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar([int(c[1:]) for c in cols], rel.mean().values, color=COL[0], edgecolor=SURF, linewidth=1.5)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Trading days from the readout 8-K")
    ax.set_ylabel("Mean log(1 + contracts),\nrelative to days -10..-6")
    ax.set_title(f"Options volume around readouts (OCC, n = {len(q):,} with {OCC_MIN_BASE}+ contracts/day before)", loc="left")
    fig.savefig(os.path.join(FIG_DIR, f"fig8_occ_volume{SUFFIX}.png"), dpi=200, bbox_inches="tight")
    plt.show()


# %% [markdown]
# ## What next
# - Check Step 5's output.
# - After the Notebook 3b v3.1 full run, rerun this notebook from the top: the stop split switches to Notebook 3b by itself (prices and OCC are cached).
# - Next: Notebook 5, the text measure (what in a readout's wording predicts the size of the move), built on `readout_events.parquet`.
# - Before the paper version: a price source that keeps delisted firms (survivorship).

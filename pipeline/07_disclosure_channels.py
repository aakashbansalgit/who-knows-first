"""Notebook 7: Disclosure channels and what happens after

Pipeline stage of the who-knows-first replication code. Cells are separated by '# %%' markers, so the file
runs top to bottom as a script and opens as a notebook in Jupyter, VS Code or (after tools/make_notebooks.py) Colab.
See README.md for inputs, outputs and the run order.
"""


# %% [markdown]
# # Notebook 7: Disclosure channels and what happens after
#
# The paper's question here: when a trial stops, does it matter **where** the news first appears (an 8-K, a 10-Q or 10-K, the registry only, or nowhere), and do prices catch up later when the channel is quiet?
#
# **Step 2. Drift after stops.** Abnormal returns over days +2..+20, +2..+60 and +2..+120 after (a) the registry posting, by whether the firm disclosed the stop first, later or never, and (b) the firm's first disclosure, by channel. Three versions of each number:
# - `raw`: stock minus XBI, Notebook 4's rules;
# - `vs own placebo dates`: the same window on random dates for the same stock (no event nearby) is subtracted, because small biotechs drift down against XBI anyway (cash burn, dilution);
# - `beta-adjusted`: stock minus beta x XBI, beta from days -250..-30.
# Long windows are clustered by firm and event month. Events whose stock stops trading inside the window (acquired, delisted, long halt) are counted, since that is where failures can hide.
#
# **Step 3. Which channel do firms choose?** Per stop: 8-K first, 10-Q/10-K first, other filing first, or never; by stated reason (safety and efficacy vs the rest), phase, sponsor size. Logits clustered by firm.
#
# **Step 4. Timing.** Friday filings for bad vs good news (the DellaVigna and Pollet test), and how long the registry lags the firm when the firm speaks first.
#
# **Step 5. The vendors' claim.** Registry outcome edits at firms with few trials (concentrated pipelines): do they predict failure or move the stock? Needs Notebook 6's outputs.
#
# **Step 6. Reconciling the readout average.** Notebook 4's Table 1 reports raw means; Notebook 6 clips tails at 1%/99%. This step prints both, and how much of the raw mean comes from the largest moves.
#
# **Step 7. Hand-coding samples.** 200 readouts and 100 disclosure matches as CSVs with empty columns to fill in.
#
# **Needs:** Notebook 3b v3.1 (`stop_disclosure.parquet`, `stop_disclosure_hits.parquet`), Notebook 4 v2 (`stop_events.parquet`, `stop_disclosure_events.parquet`, `readout_events.parquet`, cached prices), Notebook 6 for Step 5. No GPU, no downloads.
#
# **Output** (in `MyDrive/biopharma_tiv/features/nb7/`): `drift_registry.parquet`, `drift_disclosure.parquet`, `channel_choice.parquet`, `validate_readouts.csv`, `validate_matches.csv`, figures `fig13_drift.png`, `fig14_channels.png`.


# %% CONFIG
# ==== Settings ====
import os
PROJECT_DIR = os.environ.get("WKF_PROJECT_DIR", "/content/drive/MyDrive/biopharma_tiv")
BENCH = "XBI"
MAX_GAP = 5                         # same price rules as Notebook 4
WINDOWS = [(0, 1), (2, 20), (2, 60), (2, 120)]   # trading days after the event
PRE = (-60, -2)                     # pre-trend check
BETA_EST = (-250, -30)              # beta estimation window (trading days)
MIN_BETA_DAYS = 100
N_PLACEBO = 5                       # random dates per event for the same stock
PLACEBO_SPAN = 750                  # placebo dates within +-3 years of the event
PLACEBO_EXCL = 20                   # no other event of the firm within this many trading days of a placebo date
WINSOR = 0.01
N_BOOT = 1000                       # firm resamples for the intervals
SMALL_LEAD = 20                     # "small sponsor": leads at most this many trials
CONC_LEAD = 5                       # "concentrated pipeline": leads at most this many trials
N_VALID_READOUTS = 200
N_VALID_MATCHES = 100


# %% [markdown]
# ## Step 0: Setup


# %% SETUP
import os, re, glob, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

if PROJECT_DIR.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
FEAT_DIR = os.path.join(PROJECT_DIR, "features")
PRICE_DIR = os.path.join(PROJECT_DIR, "prices")
CTG_DIR = os.path.join(PROJECT_DIR, "ctgov")
NB6_DIR = os.path.join(FEAT_DIR, "nb6")
NB7_DIR = os.path.join(FEAT_DIR, "nb7")
FIG_DIR = os.path.join(FEAT_DIR, "figures")
for d in (NB7_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)
rng = np.random.default_rng(2026)

def as_bool(s):
    return pd.Series(s).astype("boolean").fillna(False).astype(bool)

def col(df, c, default):
    """df[c], or a column of `default` when the input file is older and lacks it."""
    return df[c] if c in df else pd.Series(default, index=df.index)

def as_day(s):
    d = pd.to_datetime(s, errors="coerce")
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    return d.dt.normalize().astype("datetime64[ns]")

def winsor(s, q=None):
    q = WINSOR if q is None else q
    s = pd.Series(s, dtype=float)
    if not q or s.notna().sum() < 20:
        return s
    lo, hi = s.quantile([q, 1 - q])
    return s.clip(lo, hi)

def cluster_groups(firm, date=None):
    f = pd.factorize(pd.Series(firm).astype(str))[0]
    if len(set(f)) < 10:
        return None
    if date is None:
        return f
    d = pd.factorize(pd.Series(date).astype(str))[0]
    return np.column_stack([f, d]) if len(set(d)) >= 10 else f

def stat(df, c, date_col="day0"):
    """Mean, t (clustered by firm and date_col), median, n, firms."""
    if c not in df or not len(df):
        return None
    d = df[[c, "cik", date_col]].dropna()
    n = len(d)
    if n <= 2:
        return {"mean": d[c].mean() if n else np.nan, "t": np.nan, "median": np.nan, "n": n, "firms": d["cik"].nunique()}
    y = d[c].astype(float).to_numpy()
    t = np.nan
    if y.std() > 0:
        g = cluster_groups(d["cik"], d[date_col])
        kw = {"cov_type": "cluster", "cov_kwds": {"groups": g}} if g is not None else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = float(np.asarray(sm.OLS(y, np.ones(n)).fit(**kw).tvalues)[0])
    return {"mean": y.mean(), "t": t, "median": float(np.median(y)), "n": n, "firms": d["cik"].nunique()}

def short(s):
    if s is None or s["n"] == 0:
        return "n = 0"
    tt = "" if np.isnan(s["t"]) else f" (t = {s['t']:.1f})"
    return f"{s['mean']:+.2%}{tt}"

def boot_ci(df, c, n_boot=None):
    """95% interval for the mean of df[c], resampling firms."""
    n_boot = N_BOOT if n_boot is None else n_boot
    d = df[[c, "cik"]].dropna()
    if d["cik"].nunique() < 10:
        return (np.nan, np.nan)
    sums = d.groupby("cik")[c].agg(["sum", "size"])
    s, k = sums["sum"].to_numpy(), sums["size"].to_numpy()
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    m = s[idx].sum(axis=1) / k[idx].sum(axis=1)
    return tuple(np.percentile(m, [2.5, 97.5]))

def wlab(a, b):
    return f"{a:+d}..{b:+d}"

def wname(a, b):
    return f"{a}_{b}".replace("-", "m")

PHASE_RX = re.compile(r"PHASE([1-4])")
def phase_level(p):
    nums = [int(x) for x in PHASE_RX.findall(str(p or "").upper().replace(" ", "").replace("_", ""))]
    return max(nums) if nums else np.nan

REASONS = [("safety", r"safety|adverse|toxicit|death|clinical hold|serious"),
           ("efficacy", r"futil|efficacy|lack of (?:benefit|response|activity)|did not meet|interim analys|endpoint|"
                        r"data monitoring|\bdsmb\b|\bdmc\b|\bidmc\b|unlikely to"),
           ("enrollment", r"enrol|recruit|accru"),
           ("business", r"business|strateg|sponsor|portfolio|priorit|fund|financ|commercial|company decision|resourc|acqui|"
                        r"merger|partner")]

def reason_class(x):
    t = str(x or "").strip().lower()
    if not t or t in ("nan", "none", "<na>"):
        return "not given"
    return next((k for k, rx in REASONS if re.search(rx, t)), "other")

def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)

# ---- prices: Notebook 4's rules, cached files only ----
def read_px(t):
    try:
        df = pd.read_parquet(os.path.join(PRICE_DIR, f"{t}.parquet"))
    except Exception:
        return None
    if not len(df) or not {"date", "Close"} <= set(df.columns):
        return None
    df = df.assign(date=as_day(df["date"])).dropna(subset=["date"]).drop_duplicates("date", keep="last")
    df = df.set_index("date").sort_index()
    close = pd.to_numeric(df["Close"], errors="coerce")
    vol = pd.to_numeric(df["Volume"], errors="coerce") if "Volume" in df else pd.Series(np.nan, index=df.index)
    return close, vol

b = read_px(BENCH)
assert b is not None, f"No cached prices for {BENCH} in {PRICE_DIR}: run Notebook 4 first."
bclose = b[0].where(b[0] > 0).dropna()
CAL = bclose.index
NCAL = len(CAL)

def stock_frame(t):
    x = read_px(t)
    if x is None:
        return None
    close, vol = x
    close = close[close.index.isin(CAL)]
    vol = vol.reindex(close.index)
    close = close.where(close > 0)
    close = close.where(~((vol == 0) & (close == close.shift(1))))
    c = close.dropna()
    if len(c) < 30:
        return None
    lr = np.log(c / c.shift(1))
    nxt = lr.shift(-1)
    bad = (lr.abs() > np.log(3)) & (nxt.abs() > np.log(3)) & (np.sign(lr) == -np.sign(nxt)) & ((lr + nxt).abs() < np.log(1.25))
    c = c[~bad]
    span = pd.Series(CAL.get_indexer(c.index), index=c.index).diff()
    bc = bclose.reindex(c.index)
    r = (c / c.shift(1) - 1).where(span <= MAX_GAP)
    rb = (bc / bc.shift(1) - 1).where(span <= MAX_GAP)
    return {"r": r.reindex(CAL).to_numpy(), "rb": rb.reindex(CAL).to_numpy(), "span": span.reindex(CAL).to_numpy(),
            "ar": (r - rb).reindex(CAL).to_numpy(), "live": CAL.isin(c.index)}

_PX = {}
def px_of(t):
    if not isinstance(t, str) or not t:
        return None
    if t not in _PX:
        _PX[t] = stock_frame(t)
    return _PX[t]

def wsum(x, live, i0, a, b_):
    """Sum of x over trading days i0+a..i0+b_ (Notebook 4's halt rule). NaN if the window runs past the data or has
    fewer than half its days."""
    lo, hi = i0 + a, i0 + b_
    if lo < 0 or hi > NCAL - 1:
        return np.nan
    seg = x[lo:hi + 1]
    tot, k = float(np.nansum(seg)), int(np.isfinite(seg).sum())
    if not live[hi]:
        nx = np.flatnonzero(live[hi + 1:hi + 1 + MAX_GAP])
        if len(nx) and np.isfinite(x[hi + 1 + nx[0]]):
            tot, k = tot + x[hi + 1 + nx[0]], k + 1
    return tot if k >= max(1, (b_ - a + 1) // 2) else np.nan

def beta_at(f, i0):
    lo, hi = i0 + BETA_EST[0], i0 + BETA_EST[1]
    if lo < 0:
        return np.nan
    r, rb, sp = f["r"][lo:hi + 1], f["rb"][lo:hi + 1], f["span"][lo:hi + 1]
    m = np.isfinite(r) & np.isfinite(rb) & (sp == 1)
    if m.sum() < MIN_BETA_DAYS:
        return np.nan
    v = np.var(rb[m], ddof=1)
    return float(np.clip(np.cov(r[m], rb[m], ddof=1)[0, 1] / v, -1, 4)) if v > 0 else np.nan

ALLW = WINDOWS + [PRE]
WN = [wname(a, b_) for a, b_ in WINDOWS]
LONG = WINDOWS[-1]

def windows_at(f, i0):
    beta = beta_at(f, i0)
    bar = f["r"] - beta * f["rb"] if np.isfinite(beta) else None
    out = {"beta": beta}
    for a, b_ in ALLW:
        k = wname(a, b_)
        out[f"car_{k}"] = wsum(f["ar"], f["live"], i0, a, b_)
        out[f"bcar_{k}"] = wsum(bar, f["live"], i0, a, b_) if bar is not None else np.nan
    return out

def first_pos(day):
    i0 = int(CAL.searchsorted(day))
    if i0 >= NCAL or (CAL[i0] - day).days > 7:
        return None
    return i0

def event_row(t, day):
    f = px_of(t)
    if f is None or pd.isna(day):
        return None
    i0 = first_pos(day)
    if i0 is None:
        return None
    live = f["live"]
    if not (live[max(0, i0 - 5):i0].any() and live[i0:i0 + 4].any()):
        return None
    out = {"day0": CAL[i0], "i0": i0, **windows_at(f, i0)}
    hi = i0 + LONG[1]
    out["complete"] = hi <= NCAL - 1
    out["ended_early"] = bool(hi <= NCAL - 1 and not live[max(i0, hi - 9):hi + 1].any())
    seg = f["ar"][i0 + 2:min(hi, NCAL - 1) + 1]
    out[f"trunc_{wname(2, LONG[1])}"] = float(np.nansum(seg)) if np.isfinite(seg).any() else np.nan
    return out

EVPOS = {}                       # ticker -> trading-day positions of every known event (filled in Step 1)
EVPOS_STOP = {}                  # ticker -> positions of stops and stop disclosures only
PLACEBO_POS = {}                 # event_id -> placebo positions (for the figure)

def placebo_row(eid, t, i0):
    f = px_of(t)
    maxb = max(b_ for _, b_ in WINDOWS)
    lo, hi = max(i0 - PLACEBO_SPAN, -PRE[0] + 1, -BETA_EST[0]), min(i0 + PLACEBO_SPAN, NCAL - 1 - maxb)
    if f is None or hi <= lo:
        return {}
    cand = np.arange(lo, hi + 1)
    cand = cand[f["live"][cand] & (np.abs(cand - i0) > maxb)]
    ev, evs = EVPOS.get(t), EVPOS_STOP.get(t)
    rule = "strict"
    if len(cand):
        near = np.zeros(len(cand), bool)
        if ev is not None and len(ev):                             # any event of the firm within PLACEBO_EXCL days
            near = (np.abs(ev[None, :] - cand[:, None]) <= PLACEBO_EXCL).any(axis=1)
        far = np.ones(len(cand), bool)
        if evs is not None and len(evs):                           # no stop or stop disclosure within the window length either side
            far = ~(np.abs(evs[None, :] - cand[:, None]) <= maxb).any(axis=1)
        strict = cand[~near & far]
        cand, rule = (strict, "strict") if len(strict) >= 2 else (cand[~near], "loose")
    if not len(cand):
        return {}
    picks = rng.choice(cand, size=min(N_PLACEBO, len(cand)), replace=False)
    PLACEBO_POS[eid] = np.sort(picks)
    rows = pd.DataFrame([windows_at(f, int(j)) for j in picks])
    out = {"n_placebo": len(picks), "placebo_rule": rule}
    for a, b_ in ALLW:
        k = wname(a, b_)
        out[f"pcar_{k}"] = rows[f"car_{k}"].mean()
        out[f"pbcar_{k}"] = rows[f"bcar_{k}"].mean()
    return out

def with_windows(df, id_col, tick_col, date_col):
    rows = []
    for eid, t, d in zip(df[id_col], df[tick_col], df[date_col]):
        o = event_row(t, d)
        if o:
            rows.append({id_col: eid, **o, **placebo_row(eid, t, o["i0"])})
    w = pd.DataFrame(rows)
    if not len(w):
        return df.iloc[0:0].copy()
    out = df.merge(w, on=id_col, how="inner")
    for a, b_ in ALLW:
        k = wname(a, b_)
        out[f"xcar_{k}"] = out[f"car_{k}"] - col(out, f"pcar_{k}", np.nan)
        out[f"xbcar_{k}"] = out[f"bcar_{k}"] - col(out, f"pbcar_{k}", np.nan)
    out["month"] = out["day0"].dt.to_period("M").astype(str)
    return out
print("Saving to:", NB7_DIR, "| price calendar", f"{CAL[0]:%Y-%m-%d} to {CAL[-1]:%Y-%m-%d}")


# %% [markdown]
# ## Step 1: Load events and compute the long windows
# Registry-date events (one per firm per posting day, from Notebook 4) and first-disclosure events (one per firm per filing date). Placebo dates are drawn for the same stock: no stop or stop disclosure of the firm within 120 trading days either side (so the placebo window doesn't contain the effect being measured, nor the drift after an earlier stop), and no readout within 20 days. Stocks with fewer than two such dates fall back to keeping only 20 days clear of any event (`placebo_rule` = loose).


# %% LOAD
def need(path, step):
    assert os.path.exists(path), f"Missing {path}: run {step} first."
    return pd.read_parquet(path)

se = need(os.path.join(FEAT_DIR, "stop_events.parquet"), "Notebook 4 v2")
dse = need(os.path.join(FEAT_DIR, "stop_disclosure_events.parquet"), "Notebook 4 v2")
sd = need(os.path.join(FEAT_DIR, "stop_disclosure.parquet"), "Notebook 3b v3.1")
ro = need(os.path.join(FEAT_DIR, "readout_events.parquet"), "Notebook 4 v2")
assert "registry_detail" in se and "first_date" in dse, "Notebook 4's files are older than v2: rerun Notebook 4 v2."
se["registry_date"], dse["first_date"], ro["event_date"] = as_day(se["registry_date"]), as_day(dse["first_date"]), as_day(ro["event_date"])
for d in (se, dse, ro):
    d["cik"] = pd.to_numeric(d["cik"], errors="coerce")

# every event of each firm, for the placebo exclusion
allev = pd.concat([ro[["ticker", "event_date"]].rename(columns={"event_date": "d"}),
                   se[["ticker", "registry_date"]].rename(columns={"registry_date": "d"}),
                   dse[["ticker", "first_date"]].rename(columns={"first_date": "d"})], ignore_index=True)
allev = allev.assign(stop=np.arange(len(allev)) >= len(ro)).dropna()
allev["pos"] = CAL.searchsorted(allev["d"])
EVPOS.update({t: np.unique(g.to_numpy()) for t, g in allev.groupby("ticker")["pos"]})
stopev = allev[allev["stop"]]
EVPOS_STOP.update({t: np.unique(g.to_numpy()) for t, g in stopev.groupby("ticker")["pos"]})

KEEP_S = ["event_id", "cik", "ticker", "registry_date", "group", "registry_detail", "never_disclosed", "late_stage",
          "material_reason", "reason", "firm_lead_trials", "same_day_8k", "near_readout", "n_trials", "log_dollar_vol"]
S = se[[c for c in KEEP_S if c in se]].copy()
S["registry_detail"] = col(S, "registry_detail", "").fillna("").astype(str)
S["sgroup"] = np.select([S["group"].eq("filing first"), S["registry_detail"].str.startswith("never"),
                         S["registry_detail"].str.startswith("disclosed after")],
                        ["filing first", "registry first, never disclosed", "registry first, disclosed later"], "mixed or unknown")
S = with_windows(S, "event_id", "ticker", "registry_date")

KEEP_D = ["event_id", "cik", "ticker", "first_date", "channel", "when", "late_stage", "material_reason", "firm_lead_trials",
          "near_readout", "n_trials"]
D = dse[[c for c in KEEP_D if c in dse]].copy()
ch = col(D, "channel", "").fillna("").astype(str)
D["chan"] = np.select([ch.str.contains("8-K"), ch.str.contains(r"10-Q|10-K|20-F|40-F")], ["8-K", "10-Q/10-K"], "other filing")
D = with_windows(D, "event_id", "ticker", "first_date")
for X in (S, D):
    for c in ("material_reason", "late_stage", "near_readout", "same_day_8k"):
        if c in X:
            X[c] = as_bool(X[c]).to_numpy()
    X["small"] = pd.to_numeric(col(X, "firm_lead_trials", np.nan), errors="coerce").le(SMALL_LEAD).to_numpy()
S.to_parquet(os.path.join(NB7_DIR, "drift_registry.parquet"), index=False)
D.to_parquet(os.path.join(NB7_DIR, "drift_disclosure.parquet"), index=False)
print(f"Registry-date stop events with prices: {len(S):,} of {len(se):,} | first-disclosure events with prices: {len(D):,} of {len(dse):,}")
print(f"Full {wlab(*LONG)} window available: registry {S['complete'].mean():.0%}, disclosure {D['complete'].mean():.0%} "
      f"| placebo dates per event (median): {S['n_placebo'].median() if 'n_placebo' in S else 0:.0f} | placebo windows free of "
      f"the firm's stops (+-{LONG[1]} days): {col(S, 'placebo_rule', '').eq('strict').mean():.0%} (else only +-{PLACEBO_EXCL} days kept clear)")
print("Registry groups:", S["sgroup"].value_counts().to_dict())
print("Disclosure channels:", D["chan"].value_counts().to_dict())


# %% [markdown]
# ## Step 2: Drift after stops
# For each group and window: `raw` (stock minus XBI), `vs placebo` (minus the same stock's average over random dates), `beta-adj` (stock minus beta x XBI). Day 0..+1 is clustered by firm and day; longer windows by firm and month. A negative `vs placebo` drift after quiet channels, and none after 8-Ks, would mean the market misses news that is public but not prominent.


# %% DRIFT
def show(X, gcol, groups, title):
    print(f"\n=== {title} ===")
    for g in groups:
        x = X if g == "all" else X[X[gcol] == g]
        if not len(x):
            continue
        ee = x["ended_early"].mean() if "ended_early" in x else np.nan
        print(f"-- {g}: n = {len(x):,}, firms = {x['cik'].nunique():,}, stock stops trading inside {wlab(*LONG)}: {ee:.0%}")
        for (a, b_), k in zip(ALLW, WN + [wname(*PRE)]):
            dc = "day0" if b_ <= 1 else "month"
            print(f"   days {wlab(a, b_):9s} raw {short(stat(x, 'car_' + k, dc)):>18s} | vs placebo "
                  f"{short(stat(x, 'xcar_' + k, dc)):>18s} | beta-adj vs placebo {short(stat(x, 'xbcar_' + k, dc)):>18s}")

SG = ["registry first, never disclosed", "registry first, disclosed later", "filing first"]
show(S, "sgroup", SG, "Registry posting date: drift by who spoke first")
show(S[S["small"] & ~as_bool(col(S, "near_readout", False))], "sgroup", SG,
     f"Same, sponsors leading <= {SMALL_LEAD} trials, no readout within a day")
show(S[S["material_reason"]], "sgroup", SG, "Same, stated reason safety or efficacy")
DG = ["8-K", "10-Q/10-K", "other filing"]
show(D, "chan", DG, "First disclosure in a filing: drift by channel")
show(D[D["small"]], "chan", DG, f"Same, sponsors leading <= {SMALL_LEAD} trials")
show(D[D["material_reason"]], "chan", DG, "Same, stated reason safety or efficacy")

print("\n=== Firm-bootstrap 95% intervals, drift vs placebo ===")
for X, gcol, groups in [(S, "sgroup", SG), (D, "chan", DG)]:
    for g in groups:
        x = X[X[gcol] == g]
        for a, b_ in WINDOWS[2:]:
            k = wname(a, b_)
            lo, hi = boot_ci(x, "xcar_" + k)
            ci = "fewer than 10 firms" if np.isnan(lo) else f"{lo:+.2%} to {hi:+.2%}"
            print(f"  {g:34s} days {wlab(a, b_):9s} {x['xcar_' + k].mean():+.2%} ({ci}), n = {x['xcar_' + k].notna().sum():,}")

res_drift = {}
print("\n=== Regression: drift vs placebo on group, clustered by firm ===")
for X, gcol, base, lab in [(S, "sgroup", "filing first", "registry date"), (D, "chan", "8-K", "disclosure date")]:
    for k in WN[2:]:
        y = "xcar_" + k
        z = X[X[gcol].isin(SG + DG)].dropna(subset=[y]).copy()
        z[y] = winsor(z[y])
        if len(z) < 40 or z["cik"].nunique() < 10 or z[gcol].nunique() < 2:
            continue
        f = f"{y} ~ C({gcol}, Treatment(reference='{base}')) + small + material_reason"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_ = smf.ols(f, z.assign(small=z["small"].astype(int), material_reason=z["material_reason"].astype(int))).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(z["cik"])[0]})
        res_drift[(lab, k)] = m_
        keep = [i for i in m_.params.index if i != "Intercept"]
        print(f"-- {lab}, days {k.replace('_', '..+')} (n = {len(z):,}, tails clipped at {WINSOR:.0%}; base group: {base})")
        print(pd.DataFrame({"coef": m_.params[keep], "se": m_.bse[keep], "p": m_.pvalues[keep]}).round(4).to_string())


# %% [markdown]
# ## Step 3: Which channel do firms choose?
# One row per in-sample stop (Notebook 3b). `first channel` is the earliest filing with stop wording within a year either side of the registry posting. Logits clustered by firm.


# %% CHANNEL
s = sd[as_bool(col(sd, "in_sample", True))].copy()
s["registry_date"] = as_day(s["registry_date"])
s["cik"] = pd.to_numeric(s["cik"], errors="coerce")
s["reason"] = col(s, "why_stopped", "").map(reason_class)
s["material"] = s["reason"].isin(["safety", "efficacy"]).astype(int)
s["phase_level"] = col(s, "phase", None).map(phase_level)
cur = sorted(glob.glob(os.path.join(CTG_DIR, "industry_trials_current*.parquet")), key=os.path.getmtime)
if cur and s["phase_level"].isna().any():
    c_ = pd.read_parquet(cur[-1])
    pcol = next((c for c in ("phases", "phase") if c in c_), None)
    if pcol:
        s["phase_level"] = s["phase_level"].fillna(s["nct_id"].map(dict(zip(c_["nct_id"], c_[pcol].map(phase_level)))))
s["late"] = (s["phase_level"] >= 3).astype(int)
s["small"] = pd.to_numeric(col(s, "firm_lead_trials", np.nan), errors="coerce").le(SMALL_LEAD).astype(int)
s["files_8k"] = pd.to_numeric(col(s, "firm_8ks_year", 1), errors="coerce").fillna(0).gt(0).astype(int)
s["era"] = pd.cut(s["registry_date"].dt.year, [1990, 2015, 2020, 2100], labels=["to 2015", "2016-2020", "2021+"]).astype(str)
fc = col(s, "first_channel", "none").fillna("none").astype(str)
s["outcome"] = np.select([fc.eq("8-K"), fc.str.contains(r"10-Q|10-K|20-F|40-F"), fc.eq("none")],
                         ["8-K first", "10-Q/10-K first", "never"], "other filing first")
s["never"] = (s["outcome"] == "never").astype(int)
s["first_8k"] = (s["outcome"] == "8-K first").astype(int)
s["before"] = col(s, "timing", "none").eq("before").astype(int)
s[["stop_id", "nct_id", "cik", "registry_date", "reason", "material", "phase_level", "late", "small", "files_8k", "era",
   "outcome", "before"] if "stop_id" in s else ["nct_id", "cik", "registry_date", "reason", "outcome"]].to_parquet(
    os.path.join(NB7_DIR, "channel_choice.parquet"), index=False)

ORDER = ["8-K first", "10-Q/10-K first", "other filing first", "never"]
print(f"In-sample stops: {len(s):,} at {s['cik'].nunique():,} firms")
for by, lab in [("reason", "stated reason"), ("late", "phase 3 or later (1) vs earlier or unknown (0)"),
                ("small", f"sponsor leads <= {SMALL_LEAD} trials (1)"), ("era", "era")]:
    tab = pd.crosstab(s[by], s["outcome"], normalize="index").reindex(columns=ORDER, fill_value=0)
    tab["n"] = s[by].value_counts()
    print(f"\n=== First channel by {lab} (share of stops) ===\n" + tab.round(2).to_string())

res_channel = {}
print("\n=== Logits, clustered by firm (odds ratios) ===")
for name, y, sub in [("never disclosed", "never", s), ("8-K first (disclosed stops)", "first_8k", s[s["never"] == 0]),
                     ("disclosed before the registry", "before", s)]:
    z = sub.dropna(subset=[y, "material", "late", "small", "files_8k"])
    if len(z) < 80 or z[y].nunique() < 2 or z[y].sum() < 10 or z["cik"].nunique() < 10:
        print(f"  {name}: too few stops")
        continue
    rhs = "material + late + small + C(era)" + (" + files_8k" if z["files_8k"].nunique() > 1 else "")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_ = smf.logit(f"{y} ~ {rhs}", z).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(z["cik"])[0]})
        res_channel[name] = m_
        keep = [i for i in m_.params.index if i in ("material", "late", "small", "files_8k")]
        print(f"-- {name} (n = {len(z):,}, yes = {int(z[y].sum()):,})")
        print(pd.DataFrame({"odds ratio": np.exp(m_.params[keep]), "coef": m_.params[keep], "p": m_.pvalues[keep]}).round(3).to_string())
    except Exception as e:
        print(f"  {name}: logit did not fit ({type(e).__name__}: {str(e)[:120]})")


# %% [markdown]
# ## Step 4: Timing
# (a) Day of the week of the filing: firms that want less attention file bad news on Fridays (DellaVigna and Pollet 2009). (b) When the firm speaks first, how many days later the registry posts the stop.


# %% TIMING
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri"]
r8 = ro[(ro["readout_kind"] == "topline_or_endpoint") & ro["readout_tone"].isin(["positive", "negative", "mixed"])].copy()
fday = as_day(r8["acceptance_et"]) if "acceptance_et" in r8 else r8["event_date"]
r8["dow"] = fday.dt.dayofweek
r8["fri_after_close"] = (r8["dow"] == 4) & col(r8, "timing", "").eq("after_close")
print("=== Filing weekday: readout 8-Ks (filing date" + (", Eastern time" if "acceptance_et" in r8 else ", market day") + ") ===")
tab = pd.crosstab(r8["readout_tone"], r8["dow"].map(dict(enumerate(DOW + ["Sat", "Sun"]))), normalize="index")
print(tab.reindex(columns=DOW, fill_value=0).round(2).assign(n=r8["readout_tone"].value_counts()).to_string())
print("Friday after the close:", r8.groupby("readout_tone")["fri_after_close"].mean().round(3).to_dict())
d8 = dse.copy()
d8["dow"] = d8["first_date"].dt.dayofweek
ch = col(d8, "channel", "").fillna("").astype(str)
d8 = d8[ch.str.contains("8-K")]
d8["kind"] = np.where(as_bool(col(d8, "material_reason", False)).to_numpy(), "stop 8-K, safety/efficacy", "stop 8-K, other reason")
if len(d8):
    print("\n=== Filing weekday: 8-Ks that first disclose a stop ===")
    tab = pd.crosstab(d8["kind"], d8["dow"].map(dict(enumerate(DOW + ["Sat", "Sun"]))), normalize="index")
    print(tab.reindex(columns=DOW, fill_value=0).round(2).assign(n=d8["kind"].value_counts()).to_string())
fri = [("readouts, positive", r8["readout_tone"].eq("positive"), r8), ("readouts, negative", r8["readout_tone"].eq("negative"), r8)]
fri += [(k, d8["kind"].eq(k), d8) for k in sorted(d8["kind"].unique())]
print("\nFriday share with 95% interval:")
for lab, mk, X in fri:
    k, n = int((X.loc[mk, "dow"] == 4).sum()), int(mk.sum())
    lo, hi = wilson(k, n)
    print(f"  {lab:30s} {k / n if n else np.nan:5.0%} ({lo:.0%} to {hi:.0%}), n = {n:,}")

bf = s[s["before"] == 1].copy()
bf["lag"] = -pd.to_numeric(col(bf, "days_first", np.nan), errors="coerce")
if len(bf):
    print(f"\n=== Registry lag when the firm disclosed first: days from first filing to the registry posting (n = {len(bf):,}) ===")
    g = bf.groupby("reason")["lag"].agg(n="size", median="median", over_30=lambda x: (x > 30).mean(), over_90=lambda x: (x > 90).mean())
    print(g.round(2).to_string())
    print(f"All: median {bf['lag'].median():.0f} days | over 30 days {(bf['lag'] > 30).mean():.0%} | over 90 days {(bf['lag'] > 90).mean():.0%}")
    print("(a lag over 30 days can mean a late registry update, or a firm announcing a plan before the status changed)")


# %% [markdown]
# ## Step 5: The registry-signal claim
# Vendors say registry outcome edits at firms with concentrated pipelines (few trials) are the strongest signal. Concentrated = the firm leads at most `CONC_LEAD` trials. (a) Failure rate of the next topline readout. (b) Stock move when the edit is posted, and drift after, vs placebo dates. Uses Notebook 6's `readout_flags.parquet` and `flag_events.parquet`.


# %% CONC
fl_path, fe_path = os.path.join(NB6_DIR, "readout_flags.parquet"), os.path.join(NB6_DIR, "flag_events.parquet")
reg_path = os.path.join(FEAT_DIR, "registry_events.parquet")
if os.path.exists(fl_path) and os.path.exists(fe_path) and os.path.exists(reg_path):
    reg = pd.read_parquet(reg_path, columns=["cik", "firm_lead_trials"])
    reg["cik"] = pd.to_numeric(reg["cik"], errors="coerce")
    lead = reg.dropna().groupby("cik")["firm_lead_trials"].median()
    fl = pd.read_parquet(fl_path)
    fl["cik"] = pd.to_numeric(fl["cik"], errors="coerce")
    fl["conc"] = fl["cik"].map(lead).le(CONC_LEAD)
    b2 = fl[fl["outcome"].isin(["success", "failure"])].copy()
    b2["fail"] = (b2["outcome"] == "failure").astype(int)
    print(f"=== (a) Failure rate of topline readouts, by outcome edit in the year before and pipeline size (n = {len(b2):,}) ===")
    for cl, cv in [("concentrated", True), ("not concentrated", False)]:
        for el, ev_ in [("outcome edit", 1), ("no outcome edit", 0)]:
            mk = (b2["conc"] == cv) & (col(b2, "has_outcome_edit", 0) == ev_)
            k, n = int(b2.loc[mk, "fail"].sum()), int(mk.sum())
            lo, hi = wilson(k, n)
            print(f"  {cl:17s} {el:16s} {k / n if n else np.nan:5.0%} failed ({k} of {n}; {lo:.0%} to {hi:.0%})")
    fe = pd.read_parquet(fe_path)
    fe["cik"] = pd.to_numeric(fe["cik"], errors="coerce")
    fe["registry_date"] = as_day(fe["registry_date"])
    fe = fe[fe["types"].str.contains("outcome_edit", na=False) & ~as_bool(col(fe, "near_readout", False))].copy()
    fe["conc"] = fe["cik"].map(lead).le(CONC_LEAD).to_numpy()
    fe["event_id"] = "f" + fe["cik"].astype("Int64").astype(str) + "_" + fe["registry_date"].dt.strftime("%Y%m%d")
    fe = fe.drop(columns=[c for c in fe.columns if c.startswith(("car_", "day0"))]).drop_duplicates("event_id")
    FE = with_windows(fe, "event_id", "ticker", "registry_date")
    print(f"\n=== (b) Outcome edits posted, no readout within a day ({len(FE):,} with prices) ===")
    for cl, cv in [("concentrated", True), ("not concentrated", False)]:
        x = FE[FE["conc"] == cv]
        print(f"  {cl:17s} n = {len(x):,} | days +0..+1 {short(stat(x, 'car_0_1'))} | days +2..+20 vs placebo "
              f"{short(stat(x, 'xcar_2_20', 'month'))} | days +2..+60 vs placebo {short(stat(x, 'xcar_2_60', 'month'))}")
else:
    FE = pd.DataFrame()
    print("Notebook 6's readout_flags.parquet / flag_events.parquet not found: run Notebook 6 first, then rerun this cell.")


# %% [markdown]
# ## Step 6: Reconciling the readout average
# Notebook 4's readout table reports raw means of days -1..+1; Notebook 6 clips the top and bottom 1% of topline readouts first. If the two differ a lot, a few very large winners drive the raw mean.


# %% RECON
t1 = ro[as_bool(col(ro, "has_prices", True)) & (ro["readout_kind"] == "topline_or_endpoint")].dropna(subset=["car_1_1"]).copy()
x = t1["car_1_1"]
lo, hi = x.quantile([WINSOR, 1 - WINSOR])
allp = ro[as_bool(col(ro, "has_prices", True))]["car_1_1"].dropna()
plo, phi = allp.quantile([WINSOR, 1 - WINSOR])
recon = {"raw": x.mean(), "clipped_topline": x.clip(lo, hi).mean(), "clipped_pooled": x.clip(plo, phi).mean(),
         "median": x.median(), "without_top10": x.drop(x.nlargest(10).index).mean(), "n": len(x)}
print(f"=== All topline readouts (n = {len(x):,}), days -1..+1 ===")
print(f"  raw mean (Notebook 4's table)          {recon['raw']:+.2%}")
print(f"  tails clipped within topline (NB6)     {recon['clipped_topline']:+.2%}   (clip at {lo:+.1%} / {hi:+.1%})")
print(f"  tails clipped over all readouts        {recon['clipped_pooled']:+.2%}")
print(f"  without the 10 largest moves           {recon['without_top10']:+.2%}")
print(f"  median                                 {recon['median']:+.2%}")
print(f"  10 largest moves: {', '.join(f'{v:+.0%}' for v in x.nlargest(10))}")
print("\n=== By tone: raw vs clipped within topline ===")
for tone in ["positive", "negative", "mixed", "unclear"]:
    y = t1.loc[t1["readout_tone"] == tone, "car_1_1"]
    if len(y):
        print(f"  {tone:9s} n = {len(y):5,} raw {y.mean():+7.2%} | clipped {y.clip(lo, hi).mean():+7.2%} | median {y.median():+7.2%}")


# %% [markdown]
# ## Step 7: Hand-coding samples
# Two CSVs to fill in by hand (open the filing link, read, type Y/N). Readouts are stratified by label so rare labels are checked too; matches oversample names shared by many trials, the riskiest kind.


# %% VALID
def edgar_url(cik, acc):
    try:
        return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{str(acc).replace('-', '')}/"
    except Exception:
        return ""

def strat(df, key, n_total, rs=0):
    if not len(df):
        return df
    w = df[key].value_counts()
    share = (w / w.sum()) ** 0.5                 # square-root allocation: rare labels get more than their share
    alloc = (share / share.sum() * n_total).round().astype(int).clip(lower=1)
    parts = [g.sample(min(len(g), int(alloc.get(k, 0))), random_state=rs) for k, g in df.groupby(key)]
    return pd.concat(parts).head(n_total)

rv = ro.copy()
rv["label"] = np.where(rv["readout_kind"] == "topline_or_endpoint", "topline " + rv["readout_tone"].astype(str), "data update")
vr = strat(rv, "label", N_VALID_READOUTS)
vr = pd.DataFrame({"accession": vr["accession"], "ticker": col(vr, "ticker", ""), "event_date": vr["event_date"].dt.date,
                   "label_kind": vr["readout_kind"], "label_tone": vr["readout_tone"],
                   "headline": col(vr, "headline", "").fillna("").astype(str).str[:300],
                   "snippet": col(vr, "readout_snippet", "").fillna("").astype(str).str[:600],
                   "filing_url": [edgar_url(c, a) for c, a in zip(vr["cik"], vr["accession"])]})
for c in ["is a trial readout (Y/N)", "topline or data update", "true tone (positive/negative/mixed/unclear)", "notes"]:
    vr[c] = ""
vr.sample(frac=1, random_state=1).to_csv(os.path.join(NB7_DIR, "validate_readouts.csv"), index=False)

hp = os.path.join(FEAT_DIR, "stop_disclosure_hits.parquet")
if os.path.exists(hp):
    h = pd.read_parquet(hp)
    h = h[as_bool(col(h, "stop_hit", False))].copy()
    ins = sd[as_bool(col(sd, "in_sample", True))]
    if "stop_id" in h and "stop_id" in ins:
        h = h[h["stop_id"].isin(ins["stop_id"])].sort_values(["stop_id", "days"]).drop_duplicates("stop_id")
        h = h.merge(ins[["stop_id", "nct_id", "cik"] + [c for c in ("ticker", "firm", "why_stopped") if c in ins]], on="stop_id", how="left")
    h["risk"] = np.where(pd.to_numeric(col(h, "name_trials", 1), errors="coerce").fillna(1) > 5, "name shared by 6+ trials",
                         col(h, "match_kind", "").astype(str))
    vm = strat(h, "risk", N_VALID_MATCHES)
    vm = pd.DataFrame({"nct_id": col(vm, "nct_id", ""), "ctgov_url": "https://clinicaltrials.gov/study/" + col(vm, "nct_id", "").astype(str),
                       "firm": col(vm, "firm", ""), "ticker": col(vm, "ticker", ""), "why_stopped": col(vm, "why_stopped", ""),
                       "form": col(vm, "form", ""), "filing_date": col(vm, "filing_date", ""), "days_vs_registry": col(vm, "days", np.nan),
                       "match_kind": col(vm, "match_kind", ""), "matched": col(vm, "matched", ""), "risk": vm["risk"],
                       "snippet": col(vm, "snippet", "").fillna("").astype(str).str[:600],
                       "filing_url": [edgar_url(c, a) for c, a in zip(col(vm, "cik", np.nan), col(vm, "accession", ""))]})
    for c in ["about this trial (Y/N)", "says it stopped or will stop (Y/N)", "notes"]:
        vm[c] = ""
    vm.sample(frac=1, random_state=1).to_csv(os.path.join(NB7_DIR, "validate_matches.csv"), index=False)
    print(f"validate_matches.csv: {len(vm)} rows | by risk: {vm['risk'].value_counts().to_dict()}")
else:
    print("stop_disclosure_hits.parquet not found (Notebook 3b v3.1): no match sample.")
print(f"validate_readouts.csv: {len(vr)} rows | by label: {vr['label_kind'].value_counts().to_dict()}")
print("Files in", NB7_DIR)


# %% [markdown]
# ## Step 8: Figures
# Check this notebook's printed output and both figures.


# %% FIGS
import matplotlib.pyplot as plt
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "legend.frameon": False, "axes.axisbelow": True})
T0, T1 = -20, LONG[1]
DAYS = np.arange(T0, T1 + 1)

def path_matrix(ticks, positions):
    rows = []
    for t, i0 in zip(ticks, positions):
        f = px_of(t)
        if f is None:
            continue
        v = np.full(len(DAYS), np.nan)
        lo, hi = int(i0) + T0, min(int(i0) + T1, NCAL - 1)
        if lo < 0 or hi < lo:
            continue
        v[:hi - lo + 1] = f["ar"][lo:hi + 1]
        rows.append(v)
    if not rows:
        return None
    m = pd.DataFrame(rows).apply(winsor)
    return m.mean(axis=0).fillna(0).cumsum().to_numpy() * 100

lines = [("registry posting, never disclosed", S[S["sgroup"] == "registry first, never disclosed"]),
         ("first disclosed in an 8-K", D[D["chan"] == "8-K"]),
         ("first disclosed in a 10-Q/10-K", D[D["chan"] == "10-Q/10-K"])]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
for (lab, X), c in zip(lines, COL):
    if not len(X):
        continue
    ev_ = path_matrix(X["ticker"], X["i0"])
    pt, pp = [], []
    for eid, t in zip(X["event_id"], X["ticker"]):
        for j in PLACEBO_POS.get(eid, []):
            pt.append(t)
            pp.append(j)
    pl = path_matrix(pt, pp) if pp else None
    if ev_ is not None:
        ax.plot(DAYS, ev_, color=c, linewidth=2.2, label=f"{lab} (n = {len(X):,})")
    if pl is not None:
        ax.plot(DAYS, pl, color=c, linewidth=1.2, linestyle="--", alpha=0.6)
ax.axvline(0, color=INK2, linewidth=1)
ax.axhline(0, color=GRID, linewidth=1)
ax.set_xlabel("Trading days from the event")
ax.set_ylabel("Cumulative abnormal return vs XBI (%)")
ax.plot([], [], color=INK2, linewidth=1.2, linestyle="--", label="dashed: same stocks on placebo dates")
ax.set_title("Drift after trial stops, by where the news first appeared", loc="left")
ax.legend(fontsize=9, loc="lower left")
fig.savefig(os.path.join(FIG_DIR, "fig13_drift.png"), dpi=200, bbox_inches="tight")
plt.show()

if len(s):
    tab = pd.crosstab(s["reason"], s["outcome"], normalize="index").reindex(columns=ORDER, fill_value=0) * 100
    nn = s["reason"].value_counts().reindex(tab.index)
    tab = tab.loc[nn.sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(tab) + 1.4))
    left = np.zeros(len(tab))
    shade = [COL[0], COL[2], COL[3], "#c9c8c3"]
    for k, c in zip(ORDER, shade):
        ax.barh(np.arange(len(tab)), tab[k], left=left, color=c, edgecolor=SURF, linewidth=2, label=k)
        left += tab[k].to_numpy()
    ax.set_yticks(np.arange(len(tab)))
    ax.set_yticklabels([f"{r} (n = {n:,})" for r, n in zip(tab.index, nn.loc[tab.index])])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of stops (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Where a trial stop is first disclosed, by the stated reason", loc="left")
    ax.legend(ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.savefig(os.path.join(FIG_DIR, "fig14_channels.png"), dpi=200, bbox_inches="tight")
    plt.show()


# %% [markdown]
# ## What next
# - Check the printed output and the two figures.
# - If quiet channels (registry only, 10-Q/10-K) are followed by negative drift vs placebo and 8-Ks are not, that is the limited-attention result and the paper's new headline. If not, the materiality story stands.
# - Fill in the two validation CSVs (about 3 to 4 hours for 300 rows); the error rates in the paper come from them.

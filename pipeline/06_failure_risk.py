"""Notebook 6: The risk of a drug failing

Pipeline stage of the who-knows-first replication code. Cells are separated by '# %%' markers, so the file
runs top to bottom as a script and opens as a notebook in Jupyter, VS Code or (after tools/make_notebooks.py) Colab.
See README.md for inputs, outputs and the run order.
"""


# %% [markdown]
# # Notebook 6: The risk of a drug failing
#
# **Part A. What success probability do prices imply?** A topline readout is close to a two-outcome bet: the stock jumps by R_up if the trial works and falls by R_down if it fails. If the price before the readout is fair, the probability of success the market assigns is
#
# p_implied = -R_down / (R_up - R_down)
#
# computed from the average move after successes and after failures in a group of readouts. Compared with the share that actually succeeded (`realized`). A positive gap means prices were more pessimistic than outcomes turned out: either the market underestimates success, or investors are paid a premium for holding the binary risk. The two can't be told apart from returns alone. Groups: era, phase, firm size, and recent years only (less survivorship). Intervals come from resampling firms.
#
# **Part B. Do registry red flags predict failure, and does the market notice them?** In the year before a readout, the trial's ClinicalTrials.gov record can show warning signs (from Notebook 3's registry changes):
# - `pcd_later`: primary completion date pushed back 3+ months,
# - `enrollment_down`: enrollment target cut 20%+,
# - `outcome_edit`: primary outcomes changed.
# `pcd_earlier` and `enrollment_up` are kept as a contrast. Three tests: (1) are flagged trials more likely to fail (controlling for phase, firm size, volatility and the stock's own run-up before the readout); (2) does the stock move when a flag is posted; (3) do failures after a flag still surprise the market as much as failures without one.
#
# **Outcome labels.** Notebook 3's tone (positive = success, negative = failure; mixed and unclear kept separate), checked against a dictionary label from the release text (success phrases such as "met its primary endpoint", failure phrases such as "did not meet"). If Notebook 5's LLM answers exist, P(met) is reported too. Results are also shown for the readouts where both labels agree.
#
# **Needs:** Notebook 3 (`readouts.parquet`, `registry_events.parquet`), Notebook 4 v2 (`readout_events.parquet` and the price files). Notebook 5's texts are used if present. No GPU, no downloads; a few minutes on a normal runtime.
#
# **Output** (in `MyDrive/biopharma_tiv/features/nb6/`): `implied_pos.csv`, `readout_flags.parquet`, `flag_events.parquet`, figures `fig11_implied_pos.png`, `fig12_flags.png`.


# %% CONFIG
# ==== Settings ====
import os
PROJECT_DIR = os.environ.get("WKF_PROJECT_DIR", "/content/drive/MyDrive/biopharma_tiv")
BENCH = "XBI"
FLAG_WINDOW = 365        # days before a readout in which a registry red flag counts
FLAG_GAP = 7             # a flag must be posted at least this many days before the readout
WINSOR = 0.01            # tails clipped before averaging moves
N_BOOT = 1000            # resamples (by firm) for the intervals
MAX_GAP = 5              # same price rules as Notebook 4


# %% [markdown]
# ## Step 0: Setup


# %% SETUP
import os, re, glob, json, warnings, datetime as dt
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
FIG_DIR = os.path.join(FEAT_DIR, "figures")
for d in (NB6_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)
rng = np.random.default_rng(2026)

def as_bool(s):
    return pd.Series(s).astype("boolean").fillna(False).astype(bool)

def col(df, c, default):
    """df[c], or a column of `default` when the input file is older and lacks it."""
    return df[c] if c in df else pd.Series(default, index=df.index)

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

def mean_t(df, col, pct=True):
    if col not in df or not len(df):
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
    t = float(np.asarray(sm.OLS(y, np.ones(n)).fit(**({"cov_type": "cluster", "cov_kwds": {"groups": g}} if g is not None else {})).tvalues)[0])
    return (f"mean {fmt(y.mean())} (t = {t:.1f}) | median {fmt(np.median(y))} | n = {n:,}, firms = {d['cik'].nunique():,}"
            + ("" if g is not None else " (fewer than 10 firms: t not clustered)"))

def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)

# ---- prices: Notebook 4's rules (cached files only, nothing is downloaded) ----
def read_px(t):
    p = os.path.join(PRICE_DIR, f"{t}.parquet")
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None
    if not len(df) or not {"date", "Close"} <= set(df.columns):
        return None
    d = pd.to_datetime(df["date"], errors="coerce")
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    df = df.assign(date=d.dt.normalize().astype("datetime64[ns]")).dropna(subset=["date"]).drop_duplicates("date", keep="last")
    df = df.set_index("date").sort_index()
    close = pd.to_numeric(df["Close"], errors="coerce")
    vol = pd.to_numeric(df["Volume"], errors="coerce") if "Volume" in df else pd.Series(np.nan, index=df.index)
    return close, vol

b = read_px(BENCH)
assert b is not None, f"No cached prices for {BENCH} in {PRICE_DIR}: run Notebook 4 first."
bclose = b[0].where(b[0] > 0).dropna()
CAL = bclose.index

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
    ar = ((c / c.shift(1) - 1) - (bc / bc.shift(1) - 1)).where(span <= MAX_GAP)
    return pd.DataFrame({"ar": ar.reindex(CAL), "traded": CAL.isin(c.index)}, index=CAL)

_PX = {}

def car_at(t, day, windows):
    """Abnormal returns summed over each (a, b) window of trading days around `day` (Notebook 4's rules)."""
    if not isinstance(t, str):
        return None
    if t not in _PX:
        _PX[t] = stock_frame(t)
    px = _PX[t]
    if px is None:
        return None
    i0 = int(CAL.searchsorted(day))
    if i0 >= len(CAL) or (CAL[i0] - day).days > 7:
        return None
    ar, live, n = px["ar"].to_numpy(), px["traded"].to_numpy(), len(CAL)
    if not (live[max(0, i0 - 5):i0].any() and live[i0:i0 + 4].any()):
        return None
    out = {"day0": CAL[i0]}
    for a, b_ in windows:
        lo, hi = max(i0 + a, 0), min(i0 + b_, n - 1)
        x = ar[lo:hi + 1]
        tot, k = float(np.nansum(x)), int(np.isfinite(x).sum())
        if not live[hi]:
            nx = np.flatnonzero(live[hi + 1:hi + 1 + MAX_GAP])
            if len(nx) and np.isfinite(ar[hi + 1 + nx[0]]):
                tot, k = tot + ar[hi + 1 + nx[0]], k + 1
        out[f"car_{a}_{b_}".replace("-", "m")] = tot if k >= max(1, (b_ - a + 1) // 2) else np.nan
    return out
print("Saving to:", NB6_DIR, "| price calendar", f"{CAL[0]:%Y-%m-%d} to {CAL[-1]:%Y-%m-%d}")


# %% [markdown]
# ## Step 1: Readouts and outcome labels
# Topline / primary-endpoint readouts with prices. Outcome from Notebook 3's tone, checked against a dictionary label from the release text (Notebook 5's text if it exists, else Notebook 3's headline and matched sentence).


# %% LOAD
ev = pd.read_parquet(os.path.join(FEAT_DIR, "readout_events.parquet"))
need_cols = {"car_m10_m2", "log_dollar_vol", "phase", "same_day_accessions"}
assert need_cols <= set(ev.columns), "readout_events.parquet is from an older Notebook 4: run Notebook 4 v2 first."
ev["event_date"] = pd.to_datetime(ev["event_date"])
ev = ev[as_bool(ev["has_prices"])].copy()
ev["year"] = ev["event_date"].dt.year
top = ev[ev["readout_kind"] == "topline_or_endpoint"].copy()
top["outcome"] = top["readout_tone"].map({"positive": "success", "negative": "failure", "mixed": "mixed"}).fillna("unclear")

SUCCESS = re.compile(r"\bmet\s+(?:its|the|both|all|each)\s+(?:\w+\s+){0,3}?(?:co-?)?primary\b|\bachieved\s+(?:its\s+|the\s+)?"
                     r"(?:primary|statistical(?:ly)?\s+significan)|\bstatistically\s+significant\s+(?:\w+\s+){0,2}?"
                     r"(?:improvement|reduction|increase|benefit|difference|effect)|\bpositive\s+(?:topline|top-line|pivotal|phase)", re.I)
FAILURE = re.compile(r"\bdid\s+not\s+(?:meet|achieve|reach|demonstrate|show)\b|\bfailed\s+to\s+(?:meet|achieve|reach|demonstrate|show)\b|"
                     r"\bnot\s+statistically\s+significant\b|\bno\s+(?:statistically\s+)?significant\s+(?:difference|improvement|"
                     r"benefit|effect)|\bmissed\s+(?:its|the)\s+primary|\bdiscontinu\w+\s+(?:the\s+)?(?:program|development)", re.I)
txt_path = os.path.join(FEAT_DIR, "nb5", "texts.parquet")
texts = pd.read_parquet(txt_path, columns=["accession", "text"]) if os.path.exists(txt_path) else pd.DataFrame(columns=["accession", "text"])
top = top.merge(texts.drop_duplicates("accession"), on="accession", how="left")
fallback = top.get("headline", pd.Series("", index=top.index)).fillna("").astype(str) + " " + \
           top.get("readout_snippet", pd.Series("", index=top.index)).fillna("").astype(str)
src = top["text"].where(top["text"].fillna("").str.len() >= 200, fallback)
s_hit, f_hit = src.str.contains(SUCCESS), src.str.contains(FAILURE)
top["dict_label"] = np.select([s_hit & ~f_hit, f_hit & ~s_hit, s_hit & f_hit], ["success", "failure", "mixed"], "none")
top["agree"] = top["outcome"].isin(["success", "failure"]) & (top["outcome"] == top["dict_label"])
print(f"Topline readouts with prices: {len(top):,} | release text from Notebook 5 for {top['text'].notna().mean():.0%}, "
      f"else Notebook 3's headline and sentence")
print("Notebook 3 tone vs dictionary label (rows: tone; share of each tone):")
print(pd.crosstab(top["outcome"], top["dict_label"], normalize="index").round(2).to_string())
print(f"Clear readouts (success or failure by tone): {top['outcome'].isin(['success', 'failure']).sum():,} | "
      f"both labels agree: {top['agree'].sum():,}")

llm_files = sorted(glob.glob(os.path.join(FEAT_DIR, "nb5", "llm_*_p3*.parquet")), key=os.path.getmtime)
llm_files = [f for f in llm_files if "dryrun" not in f and "unmasked" not in f and f.endswith(".parquet") and os.path.isfile(f)]
if llm_files:
    llm = pd.read_parquet(llm_files[-1], columns=["accession", "p_met"])
    top = top.merge(llm, on="accession", how="left")
    print(f"LLM P(met) from {os.path.basename(llm_files[-1])}: {top['p_met'].notna().mean():.0%} of readouts")
else:
    top["p_met"] = np.nan

lo, hi = top["car_1_1"].quantile([WINSOR, 1 - WINSOR])
top["car"] = top["car_1_1"].clip(lo, hi)                                     # main: days -1..+1, tails clipped
w = top["car_m10_m2"].fillna(0) + top["car_1_1"]
lo, hi = w.quantile([WINSOR, 1 - WINSOR])
top["car_long"] = w.clip(lo, hi)                                             # from day -10: includes the run-up


# %% [markdown]
# ## Step 2: Part A, market-implied probability of success
# For each group: average move after successes (R_up) and after failures (R_down), the share that succeeded (`realized`), the implied probability, and the gap (realized minus implied). `mixed as failure` counts mixed results with the failures. 95% intervals from resampling firms.


# %% POS
def implied(df, col, mixed_as_failure=False):
    bad = ["failure", "mixed"] if mixed_as_failure else ["failure"]
    s = df.loc[df["outcome"] == "success", col].dropna().to_numpy()
    f = df.loc[df["outcome"].isin(bad), col].dropna().to_numpy()
    if len(s) < 5 or len(f) < 5:
        return None
    rs, rf = s.mean(), f.mean()
    if rs <= rf:
        return None
    q = len(s) / (len(s) + len(f))
    p = -rf / (rs - rf)
    return rs, rf, q, p, len(s), len(f)

def boot(df, col, mixed_as_failure=False):
    base = implied(df, col, mixed_as_failure)
    if base is None:
        return None
    firms = df["cik"].unique()
    idx = {c: np.flatnonzero((df["cik"] == c).to_numpy()) for c in firms}
    ps, gaps = [], []
    for _ in range(N_BOOT):
        ii = np.concatenate([idx[c] for c in rng.choice(firms, len(firms))])
        r = implied(df.iloc[ii], col, mixed_as_failure)
        if r is not None:
            ps.append(r[3])
            gaps.append(r[2] - r[3])
    (rs, rf, q, p, ns, nf) = base
    return {"n_success": ns, "n_failure": nf, "R_up": rs, "R_down": rf, "realized": q, "implied": p, "gap": q - p,
            "implied_lo": np.percentile(ps, 2.5), "implied_hi": np.percentile(ps, 97.5),
            "gap_lo": np.percentile(gaps, 2.5), "gap_hi": np.percentile(gaps, 97.5)}

top["era"] = pd.cut(top["year"], [2009, 2015, 2020, 2030], labels=["2010-2015", "2016-2020", "2021-2025"]).astype(str)
top["size"] = pd.qcut(top["log_dollar_vol"].rank(method="first"), 3, labels=["small", "mid", "large"]).astype(str) \
    if top["log_dollar_vol"].notna().sum() >= 30 else "unknown"
top.loc[top["log_dollar_vol"].isna(), "size"] = "unknown"
top["phase_g"] = top["phase"].map(lambda p: "phase 3" if p == 3 else "phase 2" if p == 2 else "phase 1" if p == 1 else "phase unknown")
GROUPS = [("all", top), ("both labels agree", top[top["agree"] | ~top["outcome"].isin(["success", "failure"])]),
          ("clean events", top[as_bool(col(top, "clean", True))])]
GROUPS += [(f"era {e}", top[top["era"] == e]) for e in ["2010-2015", "2016-2020", "2021-2025"]]
GROUPS += [("2018 and later", top[top["year"] >= 2018])]
GROUPS += [(f"{p}", top[top["phase_g"] == p]) for p in ["phase 3", "phase 2", "phase 1", "phase unknown"]]
GROUPS += [(f"{s_} firms (dollar volume)", top[top["size"] == s_]) for s_ in ["small", "mid", "large"]]

rows = []
SHORT = {"all", "2018 and later", "clean events"}
for ycol, wlab, maf in [("car", "days -1..+1", False), ("car", "days -1..+1", True), ("car_long", "days -10..+1", False)]:
    for name, g in GROUPS:
        if (maf or ycol == "car_long") and name not in SHORT:
            continue
        r = boot(g, ycol, maf)
        if r:
            rows.append({"window": wlab, "mixed_as_failure": maf, "group": name, **r})
pos = pd.DataFrame(rows)
pos.to_csv(os.path.join(NB6_DIR, "implied_pos.csv"), index=False)

def show(sub, title):
    print(f"\n=== {title} ===")
    print(f"  {'group':28s} {'n up':>5s} {'n down':>6s} {'R_up':>7s} {'R_down':>7s} {'realized':>9s} {'implied':>8s} "
          f"{'gap (95% interval)':>24s}")
    for r in sub.itertuples():
        print(f"  {r.group:28s} {r.n_success:5,} {r.n_failure:6,} {r.R_up:+7.1%} {r.R_down:+7.1%} {r.realized:9.0%} "
              f"{r.implied:8.0%} {r.gap:+8.0%} ({r.gap_lo:+.0%} to {r.gap_hi:+.0%})")

show(pos[(pos["window"] == "days -1..+1") & ~pos["mixed_as_failure"]], "Implied vs realized success, days -1..+1 (mixed left out)")
show(pos[(pos["window"] == "days -1..+1") & pos["mixed_as_failure"] & pos["group"].isin(["all", "2018 and later", "clean events"])],
     "Same, mixed results counted as failures")
show(pos[(pos["window"] == "days -10..+1") & ~pos["mixed_as_failure"] & pos["group"].isin(["all", "2018 and later", "clean events"])],
     "Same, window from day -10 (price before any run-up)")
print("\n  gap > 0: prices were more pessimistic than outcomes (or a premium for holding the binary risk); gap < 0: too optimistic.")
x = top[top["outcome"].isin(["success", "failure", "mixed", "unclear"])]
print(f"  Average move across all topline readouts (should be about 0 if priced fairly with no premium): {mean_t(x, 'car')}")
if top["p_met"].notna().sum() >= 30:
    from scipy.stats import spearmanr
    z = top.dropna(subset=["p_met"])
    print(f"  LLM P(met) vs return: rank correlation {spearmanr(z['p_met'], z['car'])[0]:+.2f} (n = {len(z):,})")


# %% [markdown]
# ## Step 3: Part B, registry red flags before readouts
# Readouts linked to trials (Notebook 3's `nct_ids`). A flag counts if posted between `FLAG_WINDOW` days and `FLAG_GAP` days before the readout, on a record that wasn't stale. Only readouts at least a year after the registry archive starts, and only trials in Notebook 2's industry list, so a missing flag means no flag.


# %% FLAGS
reg = pd.read_parquet(os.path.join(FEAT_DIR, "registry_events.parquet"))
reg["registry_date"] = pd.to_datetime(reg["registry_date"], errors="coerce")
snap = pd.to_datetime(reg["snap"], errors="coerce") if "snap" in reg else reg["registry_date"]
ARCH0 = snap.min()
RED, CONTRAST = ["pcd_later", "enrollment_down", "outcome_edit"], ["pcd_earlier", "enrollment_up"]
rf = reg[reg["event_type"].isin(RED + CONTRAST) & ~as_bool(col(reg, "stale_record", False)) & reg["registry_date"].notna()]
cur_path = sorted(glob.glob(os.path.join(CTG_DIR, "industry_trials_current*.parquet")))
covered = set(pd.read_parquet(cur_path[-1], columns=["nct_id"])["nct_id"]) if cur_path else set(reg["nct_id"])

lk = top[["accession", "cik", "event_date", "nct_ids"]].copy()
lk["nct_id"] = lk["nct_ids"].fillna("").astype(str).str.split(";")
lk = lk.explode("nct_id")
lk = lk[lk["nct_id"].str.match(r"NCT\d{8}", na=False) & lk["nct_id"].isin(covered)]
m = lk.merge(rf[["nct_id", "event_type", "registry_date"]], on="nct_id", how="left")
days_before = (m["event_date"] - m["registry_date"]).dt.days
m = m[days_before.between(FLAG_GAP, FLAG_WINDOW)]
cnt = (m.pivot_table(index="accession", columns="event_type", values="nct_id", aggfunc="size", fill_value=0) if len(m)
       else pd.DataFrame(index=pd.Index([], name="accession")))
for c in RED + CONTRAST:
    if c not in cnt:
        cnt[c] = 0
fl = top[top["accession"].isin(set(lk["accession"])) & (top["event_date"] >= ARCH0 + pd.Timedelta(days=365))].copy()
fl = fl.merge(cnt[RED + CONTRAST].reset_index(), on="accession", how="left")
fl[RED + CONTRAST] = fl[RED + CONTRAST].fillna(0)
for c in RED + CONTRAST:
    fl[f"has_{c}"] = (fl[c] > 0).astype(int)
fl["any_flag"] = (fl[RED].sum(axis=1) > 0).astype(int)
fl["n_flags"] = fl[[f"has_{c}" for c in RED]].sum(axis=1)
fl.to_parquet(os.path.join(NB6_DIR, "readout_flags.parquet"), index=False)
print(f"Registry archive from {ARCH0:%Y-%m} | topline readouts linked to a covered trial, a year or more into the archive: {len(fl):,}")
print("Share with each flag in the year before: " + " | ".join(f"{c} {fl[f'has_{c}'].mean():.0%}" for c in RED + CONTRAST)
      + f" | any red flag {fl['any_flag'].mean():.0%}")

b_ = fl[fl["outcome"].isin(["success", "failure"])].copy()
b_["fail"] = (b_["outcome"] == "failure").astype(int)
print(f"\n=== Failure rate (success vs failure by tone; n = {len(b_):,}) ===")
for lab, mask in [("no red flag", b_["any_flag"] == 0), ("any red flag", b_["any_flag"] == 1)] + \
                 [(f"  {c}", b_[f"has_{c}"] == 1) for c in RED] + [(f"  contrast: {c}", b_[f"has_{c}"] == 1) for c in CONTRAST]:
    k, n = int(b_.loc[mask, "fail"].sum()), int(mask.sum())
    lo_, hi_ = wilson(k, n)
    print(f"  {lab:28s} {k / n if n else np.nan:6.0%} failed ({k:,} of {n:,}; 95% interval {lo_:.0%} to {hi_:.0%})")

b_["log_pre_vol"] = np.log(b_["pre_vol"].where(b_["pre_vol"] > 0))
b_["phase_g"] = b_["phase"].map(lambda p: "phase 3" if p == 3 else "phase 2" if p == 2 else "other")
z = b_.dropna(subset=["log_pre_vol", "log_dollar_vol", "car_m10_m2"])
res_logit = {}
if len(z) >= 60 and z["fail"].sum() >= 10 and z["any_flag"].sum() >= 5 and z["cik"].nunique() >= 10:
    print(f"\n=== Logit: P(failure), clustered by firm (n = {len(z):,}, failures {int(z['fail'].sum()):,}) ===")
    for name, rhs in [("any red flag", "any_flag"), ("each flag", " + ".join(f"has_{c}" for c in RED))]:
        f = f"fail ~ {rhs} + C(phase_g) + log_dollar_vol + log_pre_vol + car_m10_m2"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = smf.logit(f, z).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(z["cik"])[0]})
            res_logit[name] = mod
            keep = [i for i in mod.params.index if i.startswith(("any_flag", "has_", "car_m10_m2", "log_"))]
            tab = pd.DataFrame({"odds ratio": np.exp(mod.params[keep]), "coef": mod.params[keep], "se": mod.bse[keep], "p": mod.pvalues[keep]})
            print(f"  -- {name}\n" + tab.round(3).to_string())
        except Exception as e:
            print(f"  {name}: logit did not fit ({type(e).__name__}: {str(e)[:120]})")
    print("  (car_m10_m2 = the stock's move in days -10..-2: the market's own read before the readout)")
else:
    print("Too few linked readouts or flags for the logit.")


# %% [markdown]
# ## Step 4: Part B, does the market notice the flags?
# (1) The stock's move on the days a red flag is posted (posting date, the firm leads the trial, an 8-K filer with 1-100 trials, no 8-K or readout within a trading day). (2) The readout move for failures and successes with and without a flag before: if flags were priced, failures after a flag should fall less and successes after a flag should rise more.


# %% PRICED
fe = reg[reg["event_type"].isin(RED + CONTRAST) & ~as_bool(col(reg, "stale_record", False)) & (reg["date_source"] == "posted")
         & (col(reg, "owner_source", "sponsor_at_time") != "collaborator") & reg["cik"].notna()
         & pd.to_numeric(col(reg, "firm_8ks_year", 1), errors="coerce").fillna(0).gt(0)
         & pd.to_numeric(col(reg, "firm_lead_trials", 1), errors="coerce").fillna(0).between(1, 100)
         & ~as_bool(col(reg, "same_day_8k", False))].copy()
fe["cik"] = fe["cik"].astype(int)
fe["registry_date"] = fe["registry_date"].dt.normalize()
fe = fe.groupby(["cik", "registry_date"]).agg(types=("event_type", lambda s: ";".join(sorted(set(s)))),
                                             n=("nct_id", "nunique")).reset_index()
uni = sorted(glob.glob(os.path.join(PROJECT_DIR, "edgar", "universe_*.parquet")), key=os.path.getmtime)
tick = ev.dropna(subset=["ticker"]).drop_duplicates("cik").set_index("cik")["ticker"].to_dict()
if uni:
    u = pd.read_parquet(uni[-1], columns=["cik", "tickers"])
    for c, t in zip(pd.to_numeric(u["cik"], errors="coerce"), u["tickers"]):
        if pd.notna(c) and int(c) not in tick and isinstance(t, str) and t.strip():
            tick[int(c)] = min([x.strip().upper() for x in re.split(r"[;,\s]+", t) if x.strip()], key=len)
fe["ticker"] = fe["cik"].map(tick)
rpos = {c: np.asarray(CAL.searchsorted(g)) for c, g in ev.groupby("cik")["event_date"]}
fe_pos = CAL.searchsorted(fe["registry_date"])
fe["near_readout"] = [bool(len(rpos.get(c, [])) and np.abs(rpos[c] - p).min() <= 1) for c, p in zip(fe["cik"], fe_pos)]
rows = []
for r in fe.itertuples():
    o = car_at(r.ticker, r.registry_date, [(0, 1), (0, 5), (-1, 1)])
    if o:
        rows.append({"cik": r.cik, "registry_date": r.registry_date, **o})
fe = fe.merge(pd.DataFrame(rows, columns=["cik", "registry_date", "day0", "car_0_1", "car_0_5", "car_m1_1"]),
              on=["cik", "registry_date"], how="left")
fe.to_parquet(os.path.join(NB6_DIR, "flag_events.parquet"), index=False)
q = fe[fe["car_0_1"].notna() & ~fe["near_readout"]]
print(f"=== Stock move when a flag is posted (no 8-K or readout within a day; {len(q):,} firm-days with prices) ===")
for c in RED + CONTRAST:
    x = q[q["types"].str.contains(c)]
    print(f"  {c:16s} days 0..+1 {mean_t(x, 'car_0_1')}")
    print(f"  {'':16s} days 0..+5 {mean_t(x, 'car_0_5')}")

print("\n=== Readout move with and without a red flag before (days -1..+1, tails clipped) ===")
for oc in ["failure", "success", "mixed"]:
    for lab, v in [("no flag", 0), ("red flag", 1)]:
        x = fl[(fl["outcome"] == oc) & (fl["any_flag"] == v)]
        print(f"  {oc:8s} {lab:9s} {mean_t(x, 'car')}")
z2 = fl[fl["outcome"].isin(["success", "failure"])].assign(fail=lambda d_: (d_["outcome"] == "failure").astype(int))
z2 = z2.dropna(subset=["log_dollar_vol", "car"])
if len(z2) >= 60 and z2["cik"].nunique() >= 10 and z2["any_flag"].sum() >= 5:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mm = smf.ols("car ~ fail * any_flag + log_dollar_vol + C(year)", z2).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(z2["cik"])[0]})
    keep = [i for i in mm.params.index if "flag" in i or i == "fail"]
    print("  Regression of the readout move (clustered by firm):")
    print(pd.DataFrame({"coef": mm.params[keep], "se": mm.bse[keep], "p": mm.pvalues[keep]}).round(4).to_string())
    print("  (fail:any_flag > 0 would mean failures after a flag fall less, i.e. the flag was partly priced)")


# %% [markdown]
# ## Step 5: Figures
# Check this notebook's printed output and the two figures.


# %% FIGS
import matplotlib.pyplot as plt
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "legend.frameon": False, "axes.axisbelow": True})
sub = pos[(pos["window"] == "days -1..+1") & ~pos["mixed_as_failure"]].reset_index(drop=True)
if len(sub):
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(sub) + 1.2))
    y = np.arange(len(sub))[::-1]
    ax.errorbar(sub["implied"] * 100, y, xerr=[(sub["implied"] - sub["implied_lo"]) * 100, (sub["implied_hi"] - sub["implied"]) * 100],
                fmt="o", color=COL[0], markersize=8, capsize=0, linewidth=2, label="implied by prices (95% interval)")
    ax.plot(sub["realized"] * 100, y, "D", color=COL[1], markersize=8, label="realized share of successes")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["group"])
    ax.set_xlabel("Probability of success (%)")
    ax.set_title("What prices imply vs what happened: topline readouts", loc="left")
    ax.legend(fontsize=9, loc="lower left")
    fig.savefig(os.path.join(FIG_DIR, "fig11_implied_pos.png"), dpi=200, bbox_inches="tight")
    plt.show()
if len(b_):
    labs = ["no red flag", "any red flag"] + RED
    masks = [b_["any_flag"] == 0, b_["any_flag"] == 1] + [b_[f"has_{c}"] == 1 for c in RED]
    rates, los, his, ns = [], [], [], []
    for mk in masks:
        k, n = int(b_.loc[mk, "fail"].sum()), int(mk.sum())
        lo_, hi_ = wilson(k, n)
        rates.append(k / n * 100 if n else np.nan)
        los.append(lo_ * 100)
        his.append(hi_ * 100)
        ns.append(n)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = np.arange(len(labs))
    ax.bar(x, rates, color=[COL[2]] + [COL[1]] * (len(labs) - 1), edgecolor=SURF, linewidth=2)
    ax.errorbar(x, rates, yerr=[np.array(rates) - np.array(los), np.array(his) - np.array(rates)], fmt="none", ecolor=INK2, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n = {n:,})" for l, n in zip(labs, ns)])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Readouts that failed (%)")
    ax.set_title("Registry red flags in the year before a topline readout", loc="left")
    fig.savefig(os.path.join(FIG_DIR, "fig12_flags.png"), dpi=200, bbox_inches="tight")
    plt.show()


# %% [markdown]
# ## What next
# - Check the output.
# - If flags predict failure and the stock barely moves when they are posted, that is the registry-information-ignored result, alongside the stop findings.
# - With Notebook 5's LLM run, P(met) replaces the tone label as the outcome (fewer mislabeled mixed results).

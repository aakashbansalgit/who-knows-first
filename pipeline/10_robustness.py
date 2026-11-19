"""Notebook 10: Robustness and statistical checks

Pipeline stage of the who-knows-first replication code. Cells are separated by '# %%' markers, so the file
runs top to bottom as a script and opens as a notebook in Jupyter, VS Code or (after tools/make_notebooks.py) Colab.
See README.md for inputs, outputs and the run order.
"""


# %% [markdown]
# # Notebook 10: Robustness and statistical checks
#
# Every check here reuses files the earlier notebooks saved. Nothing is downloaded and no GPU is needed. It takes about 5 to 15 minutes on a normal runtime (most of it is Step 3, which reads the cached price files).
#
# | Step | Check | Paper section |
# |---|---|---|
# | 1 | Survivorship in the readout sample: are failures more common among readouts without prices? | 5, Table 4 |
# | 2 | How the implied probability of success depends on the treatment of extreme moves | 5, Table 4 |
# | 3 | The run-up before readouts: successes minus failures, against placebo dates | 5, 8 |
# | 4 | The registry versus the firm: material stops only, and the later firm disclosure of registry-first stops | 6 |
# | 5 | Confidence intervals and minimum detectable effects for every null result | 6, 7, 8, 9 |
# | 6 | Exit logits: rare events (Firth), linear probability, weighting by firm, acquisitions as a competing exit | 7 |
# | 7 | Model diagnostics: heteroskedasticity, heavy tails, influential points, collinearity, clustering | all |
# | 8 | Extras: stop 8-Ks bundled with restructurings, the most extreme moves to verify, a sample to check the disclosure search's recall | 5, 6 |
# | 9 | Summary table, with a Holm correction for the positive findings | |
#
# **Needs** (in `MyDrive/biopharma_tiv/features/`): `readout_events.parquet`, `stop_events.parquet`, `stop_disclosure_events.parquet`, `stop_disclosure.parquet` (Notebooks 3b and 4), `nb7/drift_registry.parquet`, `nb8/exit_outcomes_registry.parquet`, `nb8/exit_outcomes_stops.parquet`, `nb8/history/` (Notebook 8), `nb9/readout_insiders.parquet` (Notebook 9), and the cached prices in `MyDrive/biopharma_tiv/prices/`. A missing input skips its step with a message.
#
# **Output** (in `features/nb10/`): one CSV per step and `summary.csv`.


# %% CONFIG
# ==== Settings ====
import os
PROJECT_DIR = os.environ.get("WKF_PROJECT_DIR", "/content/drive/MyDrive/biopharma_tiv")
BENCH = "XBI"
MAX_GAP = 5              # same price rules as Notebooks 4 and 6
WINSOR = 0.01            # the paper's clipping
N_BOOT = 1000            # resamples by firm
N_PLACEBO = 5            # placebo dates per readout (Step 3)
POST_START = "2023-06-01"
MAIN_TAG = "m60_m1"      # Notebook 9's main window, days -60..-1
SMALL_LEAD = 20
RECALL_N = 50            # never-disclosed stops to check by hand (Step 8)
SEED = 2026


# %% [markdown]
# ## Step 0: Setup


# %% SETUP
import os, re, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from patsy import dmatrices

if PROJECT_DIR.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
FEAT_DIR = os.path.join(PROJECT_DIR, "features")
PRICE_DIR = os.path.join(PROJECT_DIR, "prices")
NB10_DIR = os.path.join(FEAT_DIR, "nb10")
os.makedirs(NB10_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)
warnings.filterwarnings("ignore")
RESULTS = []             # one row per headline check, collected for Step 9

def as_bool(s):
    return pd.Series(s).astype("boolean").fillna(False).astype(bool)

def col(df, c, default):
    return df[c] if c in df else pd.Series(default, index=df.index)

def load(rel, nb):
    p = os.path.join(FEAT_DIR, rel)
    if not os.path.exists(p):
        print(f"  (missing {rel}: run {nb} first; skipping what needs it)")
        return None
    return pd.read_parquet(p)

def as_day(s):
    d = pd.to_datetime(s, errors="coerce")
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    return d.dt.normalize().astype("datetime64[ns]")

def clip(s, q=WINSOR):
    s = pd.Series(s, dtype=float)
    if not q or s.notna().sum() < 20:
        return s
    lo, hi = s.quantile([q, 1 - q])
    return s.clip(lo, hi)

def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)

def groups(df, firm="cik", date=None):
    f = pd.factorize(df[firm].astype(str))[0]
    if date is None or date not in df:
        return f
    d = pd.factorize(df[date].astype(str))[0]
    return np.column_stack([f, d]) if len(set(d)) >= 10 else f

def mean_ci(df, y, date="day0"):
    """Mean with a 95% interval from standard errors clustered by firm and date (two-way)."""
    d = df.dropna(subset=[y]).copy()
    n = len(d)
    if n < 3:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "t": np.nan, "p": np.nan, "median": np.nan, "n": n, "firms": 0}
    m = sm.OLS(d[y].astype(float).to_numpy(), np.ones(n)).fit(cov_type="cluster", cov_kwds={"groups": groups(d, "cik", date)})
    b, se = float(m.params[0]), float(m.bse[0])
    return {"mean": b, "lo": b - 1.96 * se, "hi": b + 1.96 * se, "t": b / se if se > 0 else np.nan,
            "p": float(m.pvalues[0]), "median": float(d[y].median()), "n": n, "firms": d["cik"].nunique()}

def fmt_ci(r, pct=True):
    if not r or np.isnan(r["mean"]):
        return f"n = {r['n'] if r else 0}"
    f = (lambda v: f"{v:+.2%}") if pct else (lambda v: f"{v:+.3f}")
    return (f"mean {f(r['mean'])} [{f(r['lo'])}, {f(r['hi'])}] t = {r['t']:.1f} | median {f(r['median'])} | "
            f"n = {r['n']:,}, firms = {r['firms']:,}")

def firm_boot(d, stat, n=N_BOOT):
    """Point estimate and 95% interval of stat(sub-frame), resampling firms with replacement."""
    d = d.reset_index(drop=True)
    firms = d["cik"].unique()
    idx = {c: np.flatnonzero(d["cik"].to_numpy() == c) for c in firms}
    base = stat(d)
    bs = []
    for _ in range(n):
        ii = np.concatenate([idx[c] for c in rng.choice(firms, len(firms))])
        v = stat(d.iloc[ii])
        if v is not None and np.isfinite(v):
            bs.append(v)
    if len(bs) < 50:
        return base, np.nan, np.nan, np.nan
    bs = np.asarray(bs)
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return base, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), p

def fit(formula, df, kind="ols", cluster="cik", weights=None):
    z = df.copy()
    g = pd.factorize(z[cluster].astype(str))[0]
    if kind == "logit":
        return smf.logit(formula, z).fit(disp=0, cov_type="cluster", cov_kwds={"groups": g})
    if weights is not None:
        return smf.wls(formula, z, weights=z[weights]).fit(cov_type="cluster", cov_kwds={"groups": g})
    return smf.ols(formula, z).fit(cov_type="cluster", cov_kwds={"groups": g})

def term(m, name):
    k = [i for i in m.params.index if i == name or i.startswith(name)]
    if not k:
        return None
    k = k[0]
    b, se = float(m.params[k]), float(m.bse[k])
    return {"b": b, "se": se, "lo": b - 1.96 * se, "hi": b + 1.96 * se, "p": float(m.pvalues[k])}

def firth_logit(formula, df, maxit=200, tol=1e-8):
    """Firth's penalized-likelihood logit: unbiased in small samples and finite under separation. Wald intervals."""
    y, X = dmatrices(formula, df, return_type="dataframe")
    yv, Xv = y.to_numpy().ravel(), X.to_numpy()
    b = np.zeros(Xv.shape[1])

    def pll(b):
        p = np.clip(1 / (1 + np.exp(-Xv @ b)), 1e-12, 1 - 1e-12)
        info = Xv.T @ (Xv * (p * (1 - p))[:, None])
        sign, logdet = np.linalg.slogdet(info)
        return np.sum(yv * np.log(p) + (1 - yv) * np.log(1 - p)) + 0.5 * logdet

    cur = pll(b)
    for _ in range(maxit):
        p = 1 / (1 + np.exp(-Xv @ b))
        w = p * (1 - p)
        info = Xv.T @ (Xv * w[:, None])
        inv = np.linalg.pinv(info)
        h = np.einsum("ij,jk,ik->i", Xv, inv, Xv) * w
        step = inv @ (Xv.T @ (yv - p + h * (0.5 - p)))
        lam = 1.0
        while lam > 1e-4:
            new = pll(b + lam * step)
            if new >= cur - 1e-10:
                break
            lam /= 2
        b, prev, cur = b + lam * step, cur, pll(b + lam * step)
        if np.max(np.abs(lam * step)) < tol:
            break
    p = 1 / (1 + np.exp(-Xv @ b))
    se = np.sqrt(np.diag(np.linalg.pinv(Xv.T @ (Xv * (p * (1 - p))[:, None]))))
    return pd.DataFrame({"b": b, "se": se}, index=X.columns)

def or_txt(b, se):
    return f"{np.exp(b):.2f} [{np.exp(b - 1.96 * se):.2f}, {np.exp(b + 1.96 * se):.2f}]"

def pp_txt(r):
    return f"{r['b'] * 100:+.1f} pts [{r['lo'] * 100:+.1f}, {r['hi'] * 100:+.1f}], p = {r['p']:.2f}"

# ---- prices, Notebook 4's rules (cached files only) ----
def read_px(t):
    try:
        df = pd.read_parquet(os.path.join(PRICE_DIR, f"{t}.parquet"))
    except Exception:
        return None
    if not len(df) or not {"date", "Close"} <= set(df.columns):
        return None
    d = as_day(df["date"])
    df = df.assign(date=d).dropna(subset=["date"]).drop_duplicates("date", keep="last").set_index("date").sort_index()
    close = pd.to_numeric(df["Close"], errors="coerce")
    vol = pd.to_numeric(df["Volume"], errors="coerce") if "Volume" in df else pd.Series(np.nan, index=df.index)
    return close, vol

_b = read_px(BENCH)
CAL = None
if _b is None:
    print(f"No cached {BENCH} prices in {PRICE_DIR}: Step 3 will be skipped.")
else:
    bclose = _b[0].where(_b[0] > 0).dropna()
    CAL = bclose.index

_PX = {}
def px_arrays(t):
    if t in _PX:
        return _PX[t]
    out = None
    x = read_px(t) if isinstance(t, str) and CAL is not None else None
    if x is not None:
        close, vol = x
        close = close[close.index.isin(CAL)]
        vol = vol.reindex(close.index)
        close = close.where(close > 0)
        close = close.where(~((vol == 0) & (close == close.shift(1))))
        c = close.dropna()
        if len(c) >= 30:
            lr = np.log(c / c.shift(1))
            nxt = lr.shift(-1)
            bad = (lr.abs() > np.log(3)) & (nxt.abs() > np.log(3)) & (np.sign(lr) == -np.sign(nxt)) & ((lr + nxt).abs() < np.log(1.25))
            c = c[~bad]
            span = pd.Series(CAL.get_indexer(c.index), index=c.index).diff()
            bc = bclose.reindex(c.index)
            ar = ((c / c.shift(1) - 1) - (bc / bc.shift(1) - 1)).where(span <= MAX_GAP)
            out = (ar.reindex(CAL).to_numpy(), CAL.isin(c.index))
    _PX[t] = out
    return out

def wsum(ar, i0, a, b):
    lo, hi = max(i0 + a, 0), min(i0 + b, len(ar) - 1)
    if hi < lo:
        return np.nan
    x = ar[lo:hi + 1]
    k = int(np.isfinite(x).sum())
    return float(np.nansum(x)) if k >= max(1, (b - a + 1) // 2) else np.nan
print("Saving to:", NB10_DIR)


# %% [markdown]
# ## Step 1: Survivorship in the readout sample
# Prices exist only for firms that still trade. If failed readouts are more common among firms that later disappeared, the realized success rate in Table 4 (computed on readouts with prices) is too high. The tone labels exist for every readout, priced or not, so the realized rate can be computed on all of them.


# %% SURV
RO = load("readout_events.parquet", "Notebook 4")
if RO is not None:
    RO["event_date"] = as_day(RO["event_date"])
    RO["cik"] = pd.to_numeric(RO["cik"], errors="coerce")
    RO["has_prices"] = as_bool(RO["has_prices"]).to_numpy()
    if "day0" in RO:
        RO["day0"] = as_day(RO["day0"]).fillna(RO["event_date"])
    else:
        RO["day0"] = RO["event_date"]
    RO["year"] = RO["event_date"].dt.year
    TA = RO[RO["readout_kind"] == "topline_or_endpoint"].copy()
    TA["outcome"] = TA["readout_tone"].map({"positive": "success", "negative": "failure", "mixed": "mixed"}).fillna("unclear")
    print("=== Topline readouts by tone, with and without prices ===")
    print(pd.crosstab(TA["outcome"], TA["has_prices"].map({True: "with prices", False: "no prices"}), margins=True).to_string())
    rows = []
    for lab, x in [("with prices (Table 4)", TA[TA["has_prices"]]), ("no prices", TA[~TA["has_prices"]]), ("all", TA)]:
        s, f = int((x["outcome"] == "success").sum()), int((x["outcome"] == "failure").sum())
        lo, hi = wilson(s, s + f)
        rows.append({"readouts": lab, "successes": s, "failures": f, "success share": s / (s + f) if s + f else np.nan, "lo": lo, "hi": hi})
    sv = pd.DataFrame(rows)
    print("\n=== Success share, positive / (positive + negative) ===")
    print(sv.to_string(index=False, float_format=lambda v: f"{v:.1%}"))
    z = TA[TA["outcome"].isin(["success", "failure"])].assign(fail=lambda d: (d["outcome"] == "failure").astype(int),
                                                              priced=lambda d: d["has_prices"].astype(int))
    try:
        m = fit("priced ~ fail + C(year)", z, "logit")
        r = term(m, "fail")
        print(f"\nLogit, has prices on failure (year effects, clustered by firm): odds ratio {or_txt(r['b'], r['se'])}, p = {r['p']:.2f}")
        print("  (below 1: failed readouts are less likely to have prices, so Table 4's realized rate is biased up)")
    except Exception as e:
        print(f"  logit did not fit: {e}")
    print("\nBy era (success share with prices | without):")
    TA["era"] = pd.cut(TA["year"], [2009, 2015, 2020, 2026], labels=["2010-2015", "2016-2020", "2021-2025"])
    for era, x in TA.groupby("era", observed=True):
        a = x[x["has_prices"]]; b_ = x[~x["has_prices"]]
        sh = lambda d: (d["outcome"] == "success").sum() / max(1, d["outcome"].isin(["success", "failure"]).sum())
        print(f"  {era}: {sh(a):.0%} (n = {a['outcome'].isin(['success', 'failure']).sum():,}) | {sh(b_):.0%} "
              f"(n = {b_['outcome'].isin(['success', 'failure']).sum():,})")
    sv.to_csv(os.path.join(NB10_DIR, "survivorship_readouts.csv"), index=False)
    REAL_ALL = float(sv.loc[sv["readouts"] == "all", "success share"].iloc[0])
    REAL_PX = float(sv.loc[sv["readouts"] == "with prices (Table 4)", "success share"].iloc[0])
    RESULTS.append({"check": "Realized success share, all readouts vs priced", "estimate": f"{REAL_ALL:.1%} vs {REAL_PX:.1%}",
                    "interval": "", "p": np.nan, "family": "", "reading": "a big drop means Table 4's gap is partly survivorship"})


# %% [markdown]
# ## Step 2: The implied probability of success and the extreme moves
# p = -R_down / (R_up - R_down) uses mean moves. The mean is the right object (the expected price), but ten moves above +300% shape R_up. This step shows the gap (realized minus implied) under different treatments of the tails, on the full sample, on clean events as defined in the paper (which drop any daily move above 200%), and on clean events without that filter. It also computes the probability within groups of similar readouts (phase and size) and averages them, since a pooled probability from pooled means can differ from the average of group probabilities.


# %% IMPLIED
if RO is not None:
    T = TA[TA["has_prices"] & TA["outcome"].isin(["success", "failure"]) & TA["car_1_1"].notna()].copy()
    base_q = TA.loc[TA["has_prices"], "car_1_1"]                                  # Notebook 6 clips on all priced topline readouts
    def qclip(s, q):
        if not q:
            return s
        lo, hi = base_q.quantile([q, 1 - q])
        return s.clip(lo, hi)
    def qtrim(s, q):                                                           # within each outcome, so trimming does not change the realized share
        g = T.loc[s.index, "outcome"]
        lo, hi = s.groupby(g).transform(lambda x: x.quantile(q)), s.groupby(g).transform(lambda x: x.quantile(1 - q))
        return s.where((s >= lo) & (s <= hi))
    T["clean_nox"] = (as_bool(col(T, "overlap", False)) | as_bool(col(T, "earnings_day", False)) |
                      as_bool(col(T, "recent_ipo", False))).map(lambda v: not v).to_numpy()
    T["clean"] = as_bool(col(T, "clean", False)).to_numpy()

    def gap_stat(stat):
        def f(d):
            s, fl = d.loc[d["outcome"] == "success", "y"].dropna(), d.loc[d["outcome"] == "failure", "y"].dropna()
            if len(s) < 5 or len(fl) < 5:
                return np.nan
            rs, rf = stat(s), stat(fl)
            if rs <= rf:
                return np.nan
            ns_, nf_ = (d["outcome"] == "success").sum(), (d["outcome"] == "failure").sum()
            return ns_ / (ns_ + nf_) - (-rf / (rs - rf))
        return f

    variants = [("raw means", lambda s: s, np.mean), ("clip 0.5%", lambda s: qclip(s, 0.005), np.mean),
                ("clip 1% (paper)", lambda s: qclip(s, 0.01), np.mean), ("clip 2.5%", lambda s: qclip(s, 0.025), np.mean),
                ("clip 5%", lambda s: qclip(s, 0.05), np.mean), ("trim 1% within outcome", lambda s: qtrim(s, 0.01), np.mean),
                ("trim 5% within outcome", lambda s: qtrim(s, 0.05), np.mean), ("medians (not an expectation)", lambda s: s, np.median)]
    rows = []
    for samp, mask in [("all", np.ones(len(T), bool)), ("clean (paper)", T["clean"].to_numpy()),
                       ("clean without the 200% filter", T["clean_nox"].to_numpy())]:
        for name, tf, stat in variants:
            if samp != "all" and name not in ("raw means", "clip 1% (paper)", "trim 5% within outcome"):
                continue
            d = T[mask].assign(y=lambda x: tf(x["car_1_1"]))
            s, fl = d.loc[d["outcome"] == "success", "y"].dropna(), d.loc[d["outcome"] == "failure", "y"].dropna()
            rs, rf = stat(s), stat(fl)
            p = -rf / (rs - rf) if rs > rf else np.nan
            g, lo, hi, _ = firm_boot(d[["cik", "outcome", "y"]], gap_stat(stat))
            ns_, nf_ = (d["outcome"] == "success").sum(), (d["outcome"] == "failure").sum()
            rows.append({"sample": samp, "returns": name, "R_up": rs, "R_down": rf, "realized": ns_ / (ns_ + nf_),
                         "implied": p, "gap": g, "lo": lo, "hi": hi, "n": len(d)})
    IM = pd.DataFrame(rows)
    print("=== Implied vs realized probability of success, by treatment of the tails (95% interval, resampling firms) ===")
    print(IM.to_string(index=False, formatters={c: (lambda v: f"{v:+.1%}") for c in ["R_up", "R_down", "gap", "lo", "hi"]} |
                       {c: (lambda v: f"{v:.0%}") for c in ["realized", "implied"]}))
    IM.to_csv(os.path.join(NB10_DIR, "implied_sensitivity.csv"), index=False)
    if RO is not None and "REAL_ALL" in globals():
        r = IM[(IM["sample"] == "all") & (IM["returns"] == "clip 1% (paper)")].iloc[0]
        print(f"\nWith the realized rate from all readouts ({REAL_ALL:.0%}) instead of priced ones: gap {REAL_ALL - r['implied']:+.1%} "
              f"(paper: {r['gap']:+.1%})")

    # implied probability within groups, then averaged
    T["y"] = qclip(T["car_1_1"], 0.01)
    T["phase_g"] = pd.to_numeric(col(T, "phase", np.nan), errors="coerce").map(lambda v: "phase 3+" if v >= 3 else ("phase 2" if v == 2 else "other"))
    T["size_g"] = pd.qcut(T["log_dollar_vol"].rank(method="first"), 3, labels=["small", "mid", "large"]) if T["log_dollar_vol"].notna().sum() > 30 else "all"
    def strat(d):
        num, den = 0.0, 0
        for _, g_ in d.groupby(["phase_g", "size_g"], observed=True):
            s, fl = g_.loc[g_["outcome"] == "success", "y"], g_.loc[g_["outcome"] == "failure", "y"]
            if len(s) >= 5 and len(fl) >= 5 and s.mean() > fl.mean():
                num += (-fl.mean() / (s.mean() - fl.mean())) * len(g_)
                den += len(g_)
        return num / den if den else np.nan
    ps = strat(T)
    cover = T.groupby(["phase_g", "size_g"], observed=True).filter(lambda g_: (g_["outcome"] == "success").sum() >= 5 and (g_["outcome"] == "failure").sum() >= 5)
    pooled = IM[(IM["sample"] == "all") & (IM["returns"] == "clip 1% (paper)")]["implied"].iloc[0]
    print(f"\nImplied probability within phase x size groups, averaged: {ps:.0%} (pooled: {pooled:.0%}); "
          f"groups with 5+ successes and failures cover {len(cover) / len(T):.0%} of readouts; realized there "
          f"{(cover['outcome'] == 'success').mean():.0%}")
    r = IM[(IM["sample"] == "all") & (IM["returns"] == "clip 1% (paper)")].iloc[0]
    RESULTS.append({"check": "Implied-vs-realized gap, range across tail treatments (means only)",
                    "estimate": f"{IM.loc[(IM['sample'] == 'all') & ~IM['returns'].str.startswith('medians'), 'gap'].min():+.1%} to "
                                f"{IM.loc[(IM['sample'] == 'all') & ~IM['returns'].str.startswith('medians'), 'gap'].max():+.1%}",
                    "interval": f"paper {r['gap']:+.1%} [{r['lo']:+.1%}, {r['hi']:+.1%}]", "p": np.nan, "family": "",
                    "reading": "if the sign holds across treatments, 'roughly fair odds' stands"})


# %% [markdown]
# ## Step 3: The run-up before readouts
# A run-up before successes is evidence of leaks only if it is larger than before failures: speculators buy into known catalyst dates whatever the outcome. Windows end on day -2 (day -1 can contain the press release of an after-close 8-K). Each readout is also compared with the same stock on placebo dates (no readout of the firm within 30 trading days), because small biotechs drift against XBI in ordinary times.


# %% RUNUP
RU = None
if RO is not None and CAL is not None:
    R3 = TA[TA["has_prices"] & TA["outcome"].isin(["success", "failure", "mixed"])].copy()
    rpos = {c: np.sort(CAL.searchsorted(g.to_numpy())) for c, g in RO.groupby("cik")["day0"]}
    WIN = [(-5, -2), (-10, -2), (-20, -2)]
    out = []
    for r in R3.itertuples():
        a = px_arrays(r.ticker)
        if a is None:
            continue
        ar, live = a
        i0 = int(CAL.searchsorted(r.day0))
        if i0 >= len(CAL) or i0 < 25:
            continue
        rec = {"accession": r.accession}
        for w in WIN:
            rec[f"run_{-w[0]}"] = wsum(ar, i0, *w)
        cand = np.arange(max(i0 - 500, 30), min(i0 + 500, len(CAL) - 30))
        ev = rpos.get(r.cik, np.array([]))
        if len(ev):
            cand = cand[~(np.abs(ev[None, :] - cand[:, None]) <= 30).any(axis=1)]
        cand = cand[live[cand]]
        if len(cand) >= 2:
            pk = rng.choice(cand, size=min(N_PLACEBO, len(cand)), replace=False)
            for w in WIN:
                rec[f"plc_{-w[0]}"] = np.nanmean([wsum(ar, int(j), *w) for j in pk])
        out.append(rec)
    RU = R3.merge(pd.DataFrame(out), on="accession", how="inner")
    for w in WIN:
        k = -w[0]
        RU[f"xrun_{k}"] = RU[f"run_{k}"] - col(RU, f"plc_{k}", np.nan)
    print(f"Readouts with prices for the run-up: {len(RU):,} | check against Notebook 4 (days -10..-2): correlation "
          f"{RU[['run_10', 'car_m10_m2']].corr().iloc[0, 1]:.3f}")
    rows = []
    for k in [5, 10, 20]:
        for v, lab in [(f"run_{k}", "raw"), (f"xrun_{k}", "minus placebo")]:
            line = {"window": f"days -{k}..-2", "measure": lab}
            for oc in ["success", "failure", "mixed"]:
                m_ = mean_ci(RU[RU["outcome"] == oc], v)
                line[oc] = f"{m_['mean']:+.2%} (t {m_['t']:.1f})"
            d = RU[RU["outcome"].isin(["success", "failure"])][["cik", "outcome", v]].dropna()
            d["y"] = clip(d[v])
            g, lo, hi, p = firm_boot(d, lambda x: x.loc[x["outcome"] == "success", "y"].mean() - x.loc[x["outcome"] == "failure", "y"].mean())
            line["success minus failure (clipped)"] = f"{g:+.2%} [{lo:+.2%}, {hi:+.2%}]"
            line["p"] = p
            rows.append(line)
            if k == 10 and lab == "minus placebo":
                RESULTS.append({"check": "Run-up, successes minus failures, days -10..-2, minus placebo", "estimate": f"{g:+.2%}",
                                "interval": f"[{lo:+.2%}, {hi:+.2%}]", "p": p, "family": "positive findings",
                                "reading": "above zero: consistent with leaks; near zero: catalyst buying"})
    RT = pd.DataFrame(rows)
    print("\n=== Run-up before topline readouts (t clustered by firm and date; difference: 95% interval resampling firms) ===")
    print(RT.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    RT.to_csv(os.path.join(NB10_DIR, "runup.csv"), index=False)
    print("\nMedian run-up, days -10..-2: " + " | ".join(f"{oc} {RU.loc[RU['outcome'] == oc, 'run_10'].median():+.2%}"
                                                      for oc in ["success", "failure", "mixed"]))
else:
    print("Skipped: needs readout_events.parquet and the cached prices.")


# %% [markdown]
# ## Step 4: The registry versus the firm
# Stops that first appear on the registry may be the immaterial ones, so a zero reaction on the registry date could mean "immaterial news" rather than "the registry is ignored". Two tests: (a) the registry-date reaction for registry-first stops with a safety or efficacy reason, late-stage stops, and small sponsors with no other news that day; (b) for registry-first stops the firm discloses later, the reaction at that later disclosure. If the later firm filing moves the price while the earlier registry posting did not, the market ignored the registry.


# %% REGISTRY
SE = load("stop_events.parquet", "Notebook 4")
DSE = load("stop_disclosure_events.parquet", "Notebook 4")
if SE is not None:
    SE["cik"] = pd.to_numeric(SE["cik"], errors="coerce")
    SE = SE[as_bool(SE["has_prices"]).to_numpy()].copy()
    SE["day0"] = as_day(col(SE, "day0", SE["registry_date"]))
    det = col(SE, "registry_detail", "").fillna("").astype(str)
    rf = SE["group"].eq("registry first or only")
    mat, late = as_bool(col(SE, "material_reason", False)).to_numpy(), as_bool(col(SE, "late_stage", False)).to_numpy()
    small = pd.to_numeric(col(SE, "firm_lead_trials", np.nan), errors="coerce").le(SMALL_LEAD).to_numpy()
    quiet = ~as_bool(col(SE, "same_day_8k", False)).to_numpy() & ~as_bool(col(SE, "near_readout", False)).to_numpy()
    print("=== (a) Registry posting date, days 0..+1 (95% interval, clustered by firm and date) ===")
    REG = {}
    for lab, m_ in [("registry first or only", rf), ("  safety or efficacy reason", rf & mat), ("  business or other reason", rf & ~mat),
                    ("  phase 3 or later", rf & late), ("  small sponsor, no other news that day", rf & small & quiet),
                    ("  small sponsor, safety or efficacy, no other news", rf & small & quiet & mat),
                    ("  never disclosed by the firm", rf & det.str.startswith("never").to_numpy()),
                    ("  disclosed by the firm later", rf & det.str.startswith("disclosed after").to_numpy()),
                    ("filing first (already disclosed)", SE["group"].eq("filing first").to_numpy())]:
        REG[lab.strip()] = mean_ci(SE[m_], "car01")
        print(f"  {lab:52s} {fmt_ci(REG[lab.strip()])}")
    r = REG["registry first or only"]
    RESULTS.append({"check": "Registry-first posting, days 0..+1", "estimate": f"{r['mean']:+.2%}", "interval": f"[{r['lo']:+.2%}, {r['hi']:+.2%}]",
                    "p": r["p"], "family": "nulls", "reading": "a narrow interval around zero is an informative null"})
    r = REG["safety or efficacy reason"]
    RESULTS.append({"check": "Registry-first posting, safety/efficacy stops", "estimate": f"{r['mean']:+.2%}",
                    "interval": f"[{r['lo']:+.2%}, {r['hi']:+.2%}]", "p": r["p"], "family": "nulls",
                    "reading": "zero here too means the registry is ignored even for material stops"})
if DSE is not None:
    DSE["cik"] = pd.to_numeric(DSE["cik"], errors="coerce")
    DSE = DSE[as_bool(DSE["has_prices"]).to_numpy()].copy()
    DSE["day0"] = as_day(col(DSE, "day0", DSE["first_date"]))
    ch = col(DSE, "channel", "").fillna("").astype(str)
    DSE["chan"] = np.select([ch.str.contains("8-K"), ch.str.contains(r"10-Q|10-K|20-F|40-F")], ["8-K", "10-Q/10-K"], "other filing")
    print("\n=== (b) Reaction at the firm's first disclosure, days 0..+1, by whether the registry had already posted the stop ===")
    for when in ["before the registry", "after the registry"]:
        for chn in ["8-K", "10-Q/10-K"]:
            x = DSE[(col(DSE, "when", "") == when) & (DSE["chan"] == chn)]
            r = mean_ci(x, "car01")
            print(f"  {when:20s} {chn:10s} {fmt_ci(r)}")
            if when == "after the registry" and chn == "8-K":
                RESULTS.append({"check": "Firm 8-K after the registry already posted the stop, days 0..+1", "estimate": f"{r['mean']:+.2%}",
                                "interval": f"[{r['lo']:+.2%}, {r['hi']:+.2%}]", "p": r["p"], "family": "positive findings",
                                "reading": "negative: the 8-K still carried news the registry posting had not delivered"})
    print("  (after the registry: these are the 'registry first, disclosed later' stops; compare with their registry-date reaction in (a))")


# %% [markdown]
# ## Step 5: Intervals and minimum detectable effects for the nulls
# A null result is informative only if its interval rules out economically large effects. For each null in the paper this step prints the estimate, its 95% interval, and what the interval rules out. For the insider tests it also gives the minimum detectable difference in selling rates (80% power, 5% test).


# %% NULLS
NL = []
def add(test, est, lo, hi, p, unit, rules):
    NL.append({"test": test, "estimate": est, "lo": lo, "hi": hi, "p": p, "unit": unit, "the interval rules out": rules})

DR = load("nb7/drift_registry.parquet", "Notebook 7")
if DR is not None:
    DR["cik"] = pd.to_numeric(DR["cik"], errors="coerce")
    for w in ["2_60", "2_120"]:
        y = f"xcar_{w}"
        if y not in DR:
            continue
        z = DR[DR["sgroup"].isin(["filing first", "registry first, never disclosed"])].dropna(subset=[y]).copy()
        z["never"] = (z["sgroup"] == "registry first, never disclosed").astype(int)
        for lab, v in [("raw", z[y]), ("clipped 1%", clip(z[y]))]:
            m = smf.ols("v ~ never", z.assign(v=v)).fit(cov_type="cluster", cov_kwds={"groups": groups(z, "cik", "month")})
            r = term(m, "never")
            add(f"Survivor drift, never disclosed minus filing first, days +{w.replace('_', '..+')} ({lab})", r["b"], r["lo"], r["hi"], r["p"], "return",
                f"a quiet-stop penalty worse than {r['lo']:+.1%}")

E = load("nb9/readout_insiders.parquet", "Notebook 9")
if E is not None:
    E["cik"] = pd.to_numeric(E["cik"], errors="coerce")
    S2 = E[E["outcome"].isin(["success", "failure"])].dropna(subset=["log_dollar_vol"]).copy()
    for v, lab in [(f"sell_{MAIN_TAG}", "any sale"), (f"sell_noplan_{MAIN_TAG}", "non-plan sale"), (f"sell_plan_{MAIN_TAG}", "plan sale"),
                   (f"buy_{MAIN_TAG}", "purchase")]:
        if v not in S2 or S2[v].sum() < 10:
            continue
        try:
            m = fit(f"{v} ~ fail + base_sell + log_dollar_vol + C(year)", S2, "logit")
            r = term(m, "fail")
            add(f"Insiders: {lab} before failure vs success (odds ratio)", np.exp(r["b"]), np.exp(r["lo"]), np.exp(r["hi"]), r["p"], "odds ratio",
                f"odds ratios above {np.exp(r['hi']):.2f}")
            m2 = fit(f"{v} ~ fail + base_sell + log_dollar_vol + C(year)", S2)
            r2 = term(m2, "fail")
            add(f"Insiders: {lab}, failure minus success (linear probability)", r2["b"], r2["lo"], r2["hi"], r2["p"], "points",
                f"more than {r2['hi'] * 100:+.0f} points more {'buying' if lab == 'purchase' else 'selling'} before failures")
        except Exception as e:
            print(f"  {lab}: did not fit ({e})")
    v = f"sell_{MAIN_TAG}"
    if v in S2:
        nf, ns = int((S2["fail"] == 1).sum()), int((S2["fail"] == 0).sum())
        p0 = S2.loc[S2["fail"] == 0, v].mean()
        mde = (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * np.sqrt(p0 * (1 - p0) * (1 / nf + 1 / ns))
        print(f"Minimum detectable difference in the share with a sale (80% power, 5% test, ignoring clustering): {mde * 100:.0f} points "
              f"(base rate {p0:.0%}, {nf} failures, {ns} successes)")
        RESULTS.append({"check": "Insider selling: minimum detectable difference", "estimate": f"{mde * 100:.0f} points", "interval": "",
                        "p": np.nan, "family": "", "reading": "differences smaller than this would go unseen"})
        try:
            m = fit(f"{v} ~ fail * post + base_sell + log_dollar_vol + C(year)", S2)
            r = term(m, "fail:post")
            add("Insiders: change in the failure-success gap after the 2023 reform", r["b"], r["lo"], r["hi"], r["p"], "points",
                f"a reform effect beyond [{r['lo'] * 100:+.0f}, {r['hi'] * 100:+.0f}] points")
        except Exception as e:
            print(f"  reform: did not fit ({e})")

if RO is not None:
    F = TA[TA["has_prices"] & TA["outcome"].isin(["success", "failure", "mixed"])].copy()
    fday = as_day(F["acceptance_et"]) if "acceptance_et" in F else F["event_date"]
    F["fri"] = (fday.dt.dayofweek >= 4).astype(int)
    F["bad"] = F["outcome"].isin(["failure", "mixed"]).astype(int)
    F["log_abn_volume"] = np.log(F["abn_volume"].where(F["abn_volume"] > 0)) if "abn_volume" in F else np.nan
    G = F.dropna(subset=["log_dollar_vol"]).copy()
    for y, lab in [("car_1_1", "immediate reaction, days -1..+1"), ("car_2_20", "drift, days +2..+20"), ("log_abn_volume", "log abnormal volume")]:
        if y not in G or G[y].notna().sum() < 50:
            continue
        z = G.dropna(subset=[y]).copy()
        if y != "log_abn_volume":
            z[y] = clip(z[y])
        m = fit(f"{y} ~ bad * fri + log_dollar_vol + C(year)", z)
        r = term(m, "bad:fri")
        add(f"Friday: bad news x Friday, {lab}", r["b"], r["lo"], r["hi"], r["p"], "return" if y != "log_abn_volume" else "log",
            f"effects beyond [{r['lo']:+.1%}, {r['hi']:+.1%}]" if y != "log_abn_volume" else f"beyond [{r['lo']:+.2f}, {r['hi']:+.2f}]")
    try:
        m = fit("fri ~ bad + log_dollar_vol + C(year)", G, "logit")
        r = term(m, "bad")
        add("Friday: bad news filed on Fridays (odds ratio)", np.exp(r["b"]), np.exp(r["lo"]), np.exp(r["hi"]), r["p"], "odds ratio",
            f"odds ratios above {np.exp(r['hi']):.2f}")
    except Exception as e:
        print(f"  Friday choice: did not fit ({e})")

if NL:
    NT = pd.DataFrame(NL)
    def f_(r, v):
        return f"{v:.2f}" if r["unit"] in ("odds ratio", "log") else (f"{v * 100:+.1f}" if r["unit"] == "points" else f"{v:+.2%}")
    show = NT.assign(estimate=[f_(r, r["estimate"]) for _, r in NT.iterrows()],
                     interval=[f"[{f_(r, r['lo'])}, {f_(r, r['hi'])}]" for _, r in NT.iterrows()],
                     p=NT["p"].round(3))[["test", "estimate", "interval", "p", "the interval rules out"]]
    print("\n=== Nulls with 95% intervals ===")
    with pd.option_context("display.max_colwidth", 80, "display.width", 250):
        print(show.to_string(index=False))
    NT.to_csv(os.path.join(NB10_DIR, "nulls_ci.csv"), index=False)
    for _, r in NT.iterrows():
        if r["test"].startswith(("Friday: bad news x Friday, immediate", "Insiders: any sale before failure", "Insiders: change")):
            RESULTS.append({"check": r["test"], "estimate": f_(r, r["estimate"]), "interval": f"[{f_(r, r['lo'])}, {f_(r, r['hi'])}]",
                            "p": r["p"], "family": "nulls", "reading": f"rules out {r['the interval rules out']}"})


# %% [markdown]
# ## Step 6: Exit logits (Section 7)
# Failure exits are rare (about 50 within a year), and one group can have none, which biases ordinary logits and can make them blow up. Five versions of each comparison: the paper's logit; Firth's penalized logit (built for rare events and separation); a linear probability model; the same weighted so each firm counts once (never-disclosed stops concentrate at large sponsors, which rarely fail); and, for failures, leaving out events whose firm was acquired in the window (an acquired firm cannot later fail). Then a firm-level version: one row per firm, its share of never-disclosed stops against whether it exited by failure within two years of a stop.


# %% EXITS
EXR = []
def exit_block(X, gcol, base, other, lab):
    X = X[X[gcol].isin([base, other])].copy()
    X["grp"] = (X[gcol] == other).astype(int)
    X["cik"] = pd.to_numeric(X["cik"], errors="coerce")
    X["w"] = 1 / X.groupby("cik")["cik"].transform("size")
    ctr = [c for c in ["small", "material", "late"] if c in X and X[c].nunique() > 1]
    rhs = "grp" + "".join(f" + {c}" for c in ctr) + " + year"
    for y in ["fail_365", "notice_365", "fail_730"]:
        if y not in X:
            continue
        z = X.dropna(subset=[y] + ctr + ["year"]).copy()
        z[y] = z[y].astype(int)
        ev_ = int(z[y].sum())
        rec = {"sample": lab, "outcome": y, "n": len(z), "events": ev_, "events per parameter": round(ev_ / (len(ctr) + 3), 1),
               "share, base": z.loc[z["grp"] == 0, y].mean(), "share, other": z.loc[z["grp"] == 1, y].mean()}
        try:
            m = fit(f"{y} ~ {rhs}", z, "logit")
            r = term(m, "grp")
            rec["logit OR"] = or_txt(r["b"], r["se"]) + f", p = {r['p']:.2f}"
        except Exception as e:
            rec["logit OR"] = f"did not fit ({type(e).__name__})"
        try:
            fr = firth_logit(f"{y} ~ {rhs}", z)
            b, se = fr.loc["grp", "b"], fr.loc["grp", "se"]
            rec["Firth OR"] = or_txt(b, se) + f", p = {2 * stats.norm.sf(abs(b / se)):.2f}"
        except Exception as e:
            rec["Firth OR"] = f"did not fit ({type(e).__name__})"
        r = term(fit(f"{y} ~ {rhs}", z), "grp")
        rec["LPM"] = pp_txt(r)
        rw = term(fit(f"{y} ~ {rhs}", z, weights="w"), "grp")
        rec["LPM, each firm once"] = pp_txt(rw)
        acq = y.replace("fail", "acq")
        if y.startswith("fail") and acq in z:
            zz = z[z[acq].fillna(0) == 0]
            r = term(fit(f"{y} ~ {rhs}", zz), "grp")
            rec["LPM, acquired left out"] = pp_txt(r)
        EXR.append(rec)
        if y == "notice_365" and lab.startswith("per stop"):
            RESULTS.append({"check": "Listing-rule notice within a year, never minus 8-K first (per stop, each firm once)",
                            "estimate": f"{rw['b'] * 100:+.1f} pts", "interval": f"[{rw['lo'] * 100:+.1f}, {rw['hi'] * 100:+.1f}]",
                            "p": rw["p"], "family": "positive findings",
                            "reading": "still negative when each firm counts once: not driven by firms with many stops"})

XR = load("nb8/exit_outcomes_registry.parquet", "Notebook 8")
XS = load("nb8/exit_outcomes_stops.parquet", "Notebook 8")
if XR is not None:
    exit_block(XR, "sgroup", "filing first", "registry first, never disclosed", "registry date: never disclosed vs filing first")
if XS is not None:
    exit_block(XS, "outcome", "8-K first", "never", "per stop: never vs 8-K first")
if EXR:
    ET = pd.DataFrame(EXR)
    print("=== Exit comparisons under five specifications (95% intervals) ===")
    with pd.option_context("display.max_colwidth", 60, "display.width", 300):
        for _, r in ET.iterrows():
            print(f"\n-- {r['sample']} | {r['outcome']} | n = {r['n']:,}, events = {r['events']}, events per parameter = {r['events per parameter']} "
                  f"| share: base {r['share, base']:.1%}, other {r['share, other']:.1%}")
            for k in ["logit OR", "Firth OR", "LPM", "LPM, each firm once", "LPM, acquired left out"]:
                if k in r and isinstance(r[k], str):
                    print(f"   {k:24s} {r[k]}")
    ET.to_csv(os.path.join(NB10_DIR, "exits_robust.csv"), index=False)
    print("\n(rule of thumb: fewer than 10 events per parameter makes ordinary logits unreliable; Firth is the safer number then)")

if XS is not None:
    Z = XS.copy()
    Z["cik"] = pd.to_numeric(Z["cik"], errors="coerce")
    Z["never"] = (Z["outcome"] == "never").astype(float)
    FM = Z.groupby("cik").agg(never_share=("never", "mean"), n_stops=("never", "size"), fail_730=("fail_730", "max"),
                              small=("small", "max"), year=("year", "min")).dropna(subset=["fail_730"])
    FM["log_stops"] = np.log(FM["n_stops"])
    FM = FM.reset_index()
    try:
        m = smf.ols("fail_730 ~ never_share + log_stops + small + year", FM).fit(cov_type="HC3")
        r = term(m, "never_share")
        print(f"\nFirm level ({len(FM):,} firms, {int(FM['fail_730'].sum())} with a failure exit within two years of a stop): "
              f"going from no never-disclosed stops to all never-disclosed changes the failure probability by {pp_txt(r)} "
              f"(linear probability, controls: number of stops, small sponsor, first year; HC3 errors)")
    except Exception as e:
        print(f"  firm level: did not fit ({e})")


# %% [markdown]
# ## Step 7: Model diagnostics
# **What the models assume.** The paper uses (1) means of abnormal returns tested with standard errors clustered by firm and event date, (2) linear regressions (OLS) of returns or indicators on characteristics, (3) logits for binary outcomes, and (4) resampling firms (bootstrap) for intervals of nonlinear quantities. OLS coefficients do not need normal or equal-variance errors to be unbiased; what those assumptions affect is the standard errors. Clustered errors are robust to unequal variance (heteroskedasticity) and to correlation within a firm or a date, so the checks below ask: is heteroskedasticity present (it will be), do clustered errors differ from classical ones (they should), are a few observations driving coefficients, and are regressors too collinear to separate.


# %% DIAG
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
DG = []
if RO is not None:
    d = RO[RO["has_prices"]].copy()
    d["log_pre_vol"] = np.log(d["pre_vol"].where(d["pre_vol"] > 0))
    d["y"] = clip(d["abs_car"])
    d["earnings_day"] = as_bool(col(d, "earnings_day", False)).astype(int)
    d = d.dropna(subset=["y", "readout_kind", "readout_tone", "log_pre_vol", "log_dollar_vol", "timing", "year", "cik", "day0"])
    cat = lambda c, ref: f"C({c}, Treatment(reference={ref!r}))" if ref in set(d[c]) else f"C({c})"
    f = (f"y ~ {cat('readout_kind', 'data_update')} + {cat('readout_tone', 'unclear')} + log_pre_vol + log_dollar_vol + "
         f"{cat('timing', 'pre_open')} + earnings_day + C(year)")
    ols = smf.ols(f, d).fit()
    print(f"=== Model: |CAR(-1,+1)| on labels and controls (Appendix Table A1), n = {int(ols.nobs):,}, R2 = {ols.rsquared:.3f} ===")
    bp = het_breuschpagan(ols.resid, ols.model.exog)
    print(f"Heteroskedasticity, Breusch-Pagan (Koenker): LM = {bp[0]:.1f}, p = {bp[1]:.2g}  -> "
          + ("present: classical standard errors are wrong; the paper's clustered errors are robust to it" if bp[1] < 0.05 else "not detected"))
    jb = jarque_bera(ols.resid)
    print(f"Residuals: skewness {jb[2]:.2f}, kurtosis {jb[3]:.1f} (normal: 0 and 3), Jarque-Bera p = {jb[1]:.2g}  -> heavy tails; "
          "with n in the thousands the central limit theorem still covers the coefficient tests, and clipping limits the leverage of extremes")
    keep = [k for k in ols.params.index if not k.startswith("C(year)") and k != "Intercept"]
    se_tab = pd.DataFrame({"coef": ols.params[keep],
                           "SE classical": ols.bse[keep],
                           "SE HC3": smf.ols(f, d).fit(cov_type="HC3").bse[keep],
                           "SE cluster firm": smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": groups(d)}).bse[keep],
                           "SE cluster firm+date": smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": groups(d, "cik", "day0")}).bse[keep]})
    se_tab.index = [re.sub(r"C\((\w+)(?:, Treatment\(reference='[^']*'\))?\)\[T\.([^\]]+)\]", r"\1 = \2", i) for i in se_tab.index]
    print("\nStandard errors under different assumptions (two-way clustered is the paper's):")
    print(se_tab.round(4).to_string())
    ratio = (se_tab["SE cluster firm+date"] / se_tab["SE classical"]).median()
    print(f"Median ratio, two-way clustered to classical: {ratio:.2f} (above 1: classical errors overstate precision)")
    DG.append({"diagnostic": "Breusch-Pagan, |CAR| model", "value": f"p = {bp[1]:.2g}", "reading": "heteroskedastic; clustered SEs handle it"})
    DG.append({"diagnostic": "Clustered / classical SE, |CAR| model", "value": f"{ratio:.2f}", "reading": "how much clustering widens intervals"})

    X = ols.model.exog
    names = ols.model.exog_names
    idx = [i for i, n in enumerate(names) if not n.startswith("C(year)") and n != "Intercept"]
    vif = pd.Series([variance_inflation_factor(X, i) for i in idx], index=[names[i] for i in idx])
    vif.index = [re.sub(r"C\((\w+)(?:, Treatment\(reference='[^']*'\))?\)\[T\.([^\]]+)\]", r"\1 = \2", i) for i in vif.index]
    print(f"\nCollinearity, variance inflation factors (above 10 is a problem): max {vif.max():.1f} ({vif.idxmax()})")
    DG.append({"diagnostic": "Max VIF, |CAR| model", "value": f"{vif.max():.1f}", "reading": "below 10: regressors separable"})

    infl = OLSInfluence(ols)
    cd = pd.Series(infl.cooks_distance[0], index=d.index)
    cut = cd.quantile(0.99)
    d2 = d[cd <= cut]
    o2 = smf.ols(f, d2).fit(cov_type="cluster", cov_kwds={"groups": groups(d2, "cik", "day0")})
    comp = pd.DataFrame({"all": ols.params[keep], "without top 1% Cook's distance": o2.params[keep]})
    comp.index = se_tab.index
    print(f"\nInfluence: {int((cd > 4 / len(d)).sum()):,} observations above Cook's 4/n; dropping the most influential 1% "
          f"changes R2 from {ols.rsquared:.3f} to {o2.rsquared:.3f}. Coefficients:")
    print(comp.round(4).to_string())
    try:
        qr = smf.quantreg(f, d).fit(q=0.5, max_iter=5000)
        comp2 = pd.DataFrame({"OLS (mean)": ols.params[keep], "median regression": qr.params[keep]})
        comp2.index = se_tab.index
        print("\nMedian regression (robust to the tails) against OLS:")
        print(comp2.round(4).to_string())
    except Exception as e:
        print(f"  median regression did not run ({type(e).__name__})")

    sh = d.groupby("day0")["cik"].transform("size")
    print(f"\nCross-sectional dependence: {(sh > 1).mean():.0%} of readouts share their event day with another readout "
          f"(max {int(sh.max())} on one day); date clustering covers this.")

    if "F" in globals() and len(F):
        z = F.dropna(subset=["log_dollar_vol", "car_1_1"]).copy()
        z["car_1_1"] = clip(z["car_1_1"])
        o = smf.ols("car_1_1 ~ bad * fri + log_dollar_vol + C(year)", z).fit()
        bp2 = het_breuschpagan(o.resid, o.model.exog)
        oc = smf.ols("car_1_1 ~ bad * fri + log_dollar_vol + C(year)", z).fit(cov_type="cluster", cov_kwds={"groups": groups(z)})
        print(f"\n=== Model: Friday test, days -1..+1 === Breusch-Pagan p = {bp2[1]:.2g}; SE of bad x Friday: classical "
              f"{o.bse['bad:fri']:.4f}, clustered by firm {oc.bse['bad:fri']:.4f}")
        good, bad_ = z.loc[z["bad"] == 0, "car_1_1"], z.loc[z["bad"] == 1, "car_1_1"]
        print(f"Residual spread by news: good {o.resid[z['bad'] == 0].std():.3f}, bad {o.resid[z['bad'] == 1].std():.3f} "
              "(unequal variance across groups is the heteroskedasticity the clustered errors absorb)")

if XS is not None:
    z = XS[XS["outcome"].isin(["8-K first", "never"])].dropna(subset=["fail_365"]).copy()
    z["grp"] = (z["outcome"] == "never").astype(int)
    tab = pd.crosstab(z["grp"], z["fail_365"])
    print(f"\n=== Logit checks, per-stop failure within a year === events {int(z['fail_365'].sum())}; "
          f"cells: {tab.to_dict()} -> " + ("a zero cell means separation: use Firth (Step 6)" if (tab == 0).any().any() else "no empty cell"))
    try:
        ctr = [c for c in ["small", "material", "late"] if c in z and z[c].nunique() > 1]
        m = smf.logit("fail_365 ~ grp" + "".join(f" + {c}" for c in ctr) + " + year", z.dropna(subset=ctr + ["year"])).fit(disp=0)
        pr, yy = m.predict(), m.model.endog
        auc = stats.mannwhitneyu(pr[yy == 1], pr[yy == 0]).statistic / ((yy == 1).sum() * (yy == 0).sum())
        print(f"Discrimination (AUC) {auc:.2f}: these logits test group differences, not prediction, so a modest AUC is expected")
        DG.append({"diagnostic": "AUC, failure logit", "value": f"{auc:.2f}", "reading": "descriptive only"})
    except Exception as e:
        print(f"  logit check did not run ({e})")
if DG:
    pd.DataFrame(DG).to_csv(os.path.join(NB10_DIR, "diagnostics.csv"), index=False)


# %% [markdown]
# ## Step 8: Extras
# (a) **Stop 8-Ks bundled with other news.** Many 8-Ks that disclose a stop also announce a restructuring (Item 2.05) or management changes (Item 5.02). The -12.8% reaction is split by whether the same day's 8-K carries those items, from Notebook 8's cached filing histories.
# (b) **The most extreme readout moves**, to check by hand against news or EDGAR (reverse mergers and ticker changes can corrupt price histories).
# (c) **A recall sample for the disclosure search**: never-disclosed stops with EDGAR full-text search links, to check by hand whether a filing did disclose them (the hand-coded match sample only measures false matches, not misses).


# %% EXTRAS
HIST_DIR = os.path.join(FEAT_DIR, "nb8", "history")
if DSE is not None and os.path.isdir(HIST_DIR):
    k8 = DSE[DSE["chan"] == "8-K"].copy()
    k8["first_date"] = as_day(k8["first_date"])
    flags = []
    for r in k8.itertuples():
        p = os.path.join(HIST_DIR, f"{int(r.cik)}.parquet")
        it = ""
        if os.path.exists(p):
            h = pd.read_parquet(p)
            h["filingDate"] = as_day(h["filingDate"])
            x = h[h["form"].astype(str).str.startswith("8-K") & (h["filingDate"] - r.first_date).abs().le(pd.Timedelta(days=1))]
            it = ",".join(x["items"].fillna("").astype(str))
        flags.append(it)
    k8["items_day"] = flags
    k8["restructuring"] = k8["items_day"].str.contains(r"(?<![\d.])2\.05(?!\d)")
    k8["mgmt_change"] = k8["items_day"].str.contains(r"(?<![\d.])5\.02(?!\d)")
    k8["financing"] = k8["items_day"].str.contains(r"(?<![\d.])(?:1\.01|3\.02)(?!\d)")
    print("=== (a) Stop 8-Ks: reaction days 0..+1 by what else the 8-K reported ===")
    print(f"  all stop 8-Ks                          {fmt_ci(mean_ci(k8, 'car01'))}")
    for lab, m_ in [("with a restructuring (2.05)", k8["restructuring"]), ("with management changes (5.02)", k8["mgmt_change"]),
                    ("with neither", ~k8["restructuring"] & ~k8["mgmt_change"]),
                    ("with neither, nor a financing or agreement (1.01, 3.02)", ~k8["restructuring"] & ~k8["mgmt_change"] & ~k8["financing"])]:
        print(f"  {lab:38s} {fmt_ci(mean_ci(k8[m_], 'car01'))}")
    r = mean_ci(k8[~k8["restructuring"] & ~k8["mgmt_change"]], "car01")
    RESULTS.append({"check": "Stop 8-K reaction without restructuring or management items, days 0..+1", "estimate": f"{r['mean']:+.2%}",
                    "interval": f"[{r['lo']:+.2%}, {r['hi']:+.2%}]", "p": r["p"], "family": "positive findings",
                    "reading": "close to -12.8%: the stop itself, not bundled news, moves the price"})
else:
    print("(a) skipped: needs stop_disclosure_events.parquet and Notebook 8's history cache")

if RO is not None:
    T8 = TA[TA["has_prices"] & TA["car_1_1"].notna()].copy()
    acc = T8["accession"].astype(str).str.replace("-", "", regex=False)
    T8["filing"] = "https://www.sec.gov/Archives/edgar/data/" + T8["cik"].astype("Int64").astype(str) + "/" + acc + "/"
    xt = pd.concat([T8.nlargest(15, "car_1_1"), T8.nsmallest(10, "car_1_1")])[
        ["ticker", "event_date", "readout_tone", "car_1_1", "max_abs_ar", "price_pre", "filing"]]
    xt.to_csv(os.path.join(NB10_DIR, "extreme_moves.csv"), index=False)
    print("\n=== (b) Most extreme topline moves (saved to extreme_moves.csv; check each against news or the filing) ===")
    with pd.option_context("display.max_colwidth", 70, "display.width", 250):
        print(xt.assign(car_1_1=xt["car_1_1"].map(lambda v: f"{v:+.0%}"),
                        max_abs_ar=xt["max_abs_ar"].map(lambda v: f"{v:.0%}" if pd.notna(v) else "")).to_string(index=False))

SDF = load("stop_disclosure.parquet", "Notebook 3b")
if SDF is not None:
    s = SDF[as_bool(SDF["in_sample"]).to_numpy() & (SDF["timing"] == "none")].copy()
    s["registry_date"] = as_day(s["registry_date"])
    s["size"] = np.where(pd.to_numeric(s["firm_lead_trials"], errors="coerce") <= SMALL_LEAD, "small", "large")
    take = s.groupby("size", group_keys=False).apply(lambda g: g.sample(min(len(g), RECALL_N // 2), random_state=SEED))
    def link(name, cik, d):
        q = "%22" + str(name).strip().replace(" ", "%20") + "%22"
        return (f"https://www.sec.gov/edgar/search/#/q={q}&ciks={int(cik):010d}&dateRange=custom"
                f"&startdt={(d - pd.Timedelta(days=365)):%Y-%m-%d}&enddt={(d + pd.Timedelta(days=365)):%Y-%m-%d}")
    names = col(take, "names_searched", "").fillna("").astype(str)
    take["search_links"] = [" | ".join(link(n, c, d) for n in nm.split(", ")[:3] if n) for nm, c, d in
                            zip(names, take["cik"], take["registry_date"])]
    out = take[[c for c in ["stop_id", "nct_id", "firm", "cik", "registry_date", "why_stopped", "brief_title", "names_searched", "size",
                            "search_links"] if c in take]].copy()
    out["disclosed in a filing? (Y/N)"], out["form and date"], out["notes"] = "", "", ""
    out.to_csv(os.path.join(NB10_DIR, "recall_check.csv"), index=False)
    print(f"\n(c) Recall sample: {len(out)} never-disclosed stops saved to recall_check.csv with EDGAR full-text search links "
          f"(one per searched name, filtered to the firm and a year either side of the posting). Open each link and mark Y if a filing "
          f"says the trial stopped. The share of Y is the search's miss rate.")


# %% [markdown]
# ## Step 9: Summary
# The headline checks in one table. The positive findings get a Holm correction across their family (it controls the chance of any false positive among them).


# %% SUMMARY
SM = pd.DataFrame(RESULTS)
if len(SM):
    SM["Holm p (within family)"] = np.nan
    for fam, g in SM[SM["family"] == "positive findings"].dropna(subset=["p"]).groupby("family"):
        order = g["p"].sort_values().index
        m_ = len(order)
        adj, run = [], 0.0
        for i, ix in enumerate(order):
            run = max(run, min(1.0, (m_ - i) * g.loc[ix, "p"]))
            adj.append((ix, run))
        for ix, v in adj:
            SM.loc[ix, "Holm p (within family)"] = v
    SM.to_csv(os.path.join(NB10_DIR, "summary.csv"), index=False)
    with pd.option_context("display.max_colwidth", 70, "display.width", 300):
        print(SM.assign(p=SM["p"].round(3), **{"Holm p (within family)": SM["Holm p (within family)"].round(3)})[
            ["check", "estimate", "interval", "p", "Holm p (within family)", "reading"]].to_string(index=False))
print("\nSaved in", NB10_DIR, ":", sorted(os.listdir(NB10_DIR)))


# %% [markdown]
# ## Step 10: Is the run-up public news? Meeting presentations against topline releases
# Results presented at a medical meeting often have an abstract online days or weeks before the 8-K, so a run-up before them can be public news. Readouts are split by whether the release mentions an abstract, poster or meeting; a leak predicts a run-up before topline releases too. Then: how concentrated the run-up is. Needs Notebook 5's `nb5/texts.parquet` and Step 3.


# %% RUNUP_SPLIT
TX = load("nb5/texts.parquet", "Notebook 5")
if TX is not None:
    TX = TX.drop_duplicates("accession", keep="last")
if TX is not None and RU is not None:
    CONF_KW = re.compile(r"\b(?:abstracts?|posters?|late[- ]breaking|oral presentation|podium presentation|annual meeting|"
                         r"scientific sessions?|congress|symposium|(?:medical|scientific|international) conference)\b", re.I)
    CONF_AT = re.compile(r"\b[Pp]resent(?:ed|s|ing|ation|ations)?\s+(?:at|during)\s+(?:the\s+)?(?:\d{4}\s+)?[A-Z][A-Za-z]")
    raw = TX.set_index("accession")["text_raw"].fillna("").astype(str)
    meet = raw.map(lambda t: bool(CONF_KW.search(t) or CONF_AT.search(t))).rename("meeting").reset_index()
    R10 = RU.merge(meet, on="accession", how="inner")
    print(f"Run-up readouts with a release text: {len(R10):,} of {len(RU):,} | mention a meeting, abstract or poster: {R10['meeting'].mean():.0%}")
    print(pd.crosstab(R10["outcome"], R10["meeting"].map({True: "meeting", False: "topline release"})).to_string())
    sf = lambda z: z.loc[z["outcome"] == "success", "y"].mean() - z.loc[z["outcome"] == "failure", "y"].mean()
    rows = []
    for lab, m_ in [("topline release", ~R10["meeting"]), ("meeting, abstract or poster", R10["meeting"])]:
        x = R10[m_]
        for v, k in [("xrun_10", "days -10..-2, minus placebo"), ("xrun_5", "days -5..-2, minus placebo"), ("run_10", "days -10..-2, raw")]:
            line = {"readouts": lab, "window": k}
            for oc in ["success", "failure"]:
                r = mean_ci(x[x["outcome"] == oc], v)
                line[oc] = f"{r['mean']:+.2%} (n {r['n']})"
            d = x[x["outcome"].isin(["success", "failure"])][["cik", "outcome", v]].dropna()
            d["y"] = clip(d[v])
            g, lo, hi, p = firm_boot(d, sf)
            line["success minus failure"] = f"{g:+.2%} [{lo:+.2%}, {hi:+.2%}]"
            line["p"] = p
            rows.append(line)
            if k.startswith("days -10..-2, minus") and lab == "topline release":
                RESULTS.append({"check": "Run-up, successes minus failures, topline releases only (no meeting mentioned)", "estimate": f"{g:+.2%}",
                                "interval": f"[{lo:+.2%}, {hi:+.2%}]", "p": p, "family": "",
                                "reading": "still positive: the run-up is not explained by meeting abstracts made public earlier"})
    RS = pd.DataFrame(rows)
    print("\n=== Run-up, successes minus failures (clipped; 95% interval resampling firms) ===")
    print(RS.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    RS.to_csv(os.path.join(NB10_DIR, "runup_by_channel.csv"), index=False)
    z = R10[R10["outcome"].isin(["success", "failure"])].dropna(subset=["xrun_10"]).copy()
    z["y"], z["succ"], z["meet"] = clip(z["xrun_10"]), (z["outcome"] == "success").astype(int), z["meeting"].astype(int)
    r = term(fit("y ~ succ * meet", z), "succ:meet")
    print(f"\nMeeting minus topline, in the success-failure gap (days -10..-2 minus placebo; clustered by firm): {r['b']:+.2%} "
          f"[{r['lo']:+.2%}, {r['hi']:+.2%}], p = {r['p']:.2f}")

if RU is not None:
    print("\n=== How concentrated is the run-up? (days -10..-2 minus placebo) ===")
    d = RU[RU["outcome"].isin(["success", "failure"])][["cik", "outcome", "xrun_10"]].dropna().rename(columns={"xrun_10": "y"})
    s = d.loc[d["outcome"] == "success", "y"].sort_values(ascending=False)
    k10 = max(1, len(s) // 10)
    print(f"Successes: mean {s.mean():+.2%}, median {s.median():+.2%} | top 10% ({k10}) average {s.head(k10).mean():+.1%} | "
          f"mean without them {s.iloc[k10:].mean():+.2%}")
    def trimdiff(z, q=0.05):
        m_ = []
        for oc in ["success", "failure"]:
            v = z.loc[z["outcome"] == oc, "y"]
            a, b = v.quantile([q, 1 - q])
            m_.append(v[(v >= a) & (v <= b)].mean())
        return m_[0] - m_[1]
    for lab, f_ in [("means, clipped 1% (Step 3)", lambda z: sf(z.assign(y=clip(z["y"])))),
                    ("means, trimming 5% of each outcome's tails", trimdiff),
                    ("medians", lambda z: z.loc[z["outcome"] == "success", "y"].median() - z.loc[z["outcome"] == "failure", "y"].median())]:
        g, lo, hi, p = firm_boot(d, f_)
        print(f"  success minus failure, {lab:45s} {g:+.2%} [{lo:+.2%}, {hi:+.2%}], p = {p:.3f}")
        if lab.startswith("medians"):
            RESULTS.append({"check": "Run-up, successes minus failures, medians (days -10..-2 minus placebo)", "estimate": f"{g:+.2%}",
                            "interval": f"[{lo:+.2%}, {hi:+.2%}]", "p": p, "family": "",
                            "reading": "positive: the typical readout shows the gap too, not only the tails"})


# %% [markdown]
# ## Step 11: The implied probability with buy-and-hold returns
# A CAR adds up daily returns, so it can fall below -100% while a price cannot, and it understates compounding on big up moves. The two-outcome model is about the price change, so this step recomputes R_up, R_down and the gap with buy-and-hold abnormal returns over days -1 to +1 (the stock's price change from the close before day -1 to the close of day +1, minus XBI's), same windows and price rules.


# %% BHAR
_CL = {}
def px_close(t):
    """Cleaned closes on XBI's calendar (Notebook 4's rules), NaN where the stock did not trade."""
    if t in _CL:
        return _CL[t]
    out = None
    x = read_px(t) if isinstance(t, str) and CAL is not None else None
    if x is not None:
        close, vol = x
        close = close[close.index.isin(CAL)]
        vol = vol.reindex(close.index)
        close = close.where(close > 0)
        close = close.where(~((vol == 0) & (close == close.shift(1))))
        c = close.dropna()
        if len(c) >= 30:
            lr = np.log(c / c.shift(1))
            nxt = lr.shift(-1)
            bad = (lr.abs() > np.log(3)) & (nxt.abs() > np.log(3)) & (np.sign(lr) == -np.sign(nxt)) & ((lr + nxt).abs() < np.log(1.25))
            out = c[~bad].reindex(CAL).to_numpy()
    _CL[t] = out
    return out

def bhar(t, day0, a=-1, b=1):
    c = px_close(t)
    if c is None or pd.isna(day0):
        return np.nan
    i0 = int(CAL.searchsorted(day0))
    s, e = i0 + a - 1, i0 + b
    if s < 0 or e >= len(CAL):
        return np.nan
    js = next((j for j in range(s, max(s - MAX_GAP, -1), -1) if np.isfinite(c[j])), None)    # last close before the window
    je = next((j for j in range(e, min(e + MAX_GAP + 1, len(CAL))) if np.isfinite(c[j])), None)  # a halt extends to the next close
    if js is None or je is None:
        return np.nan
    return c[je] / c[js] - BREF[je] / BREF[js]

if RO is not None and CAL is not None:
    BREF = bclose.reindex(CAL).to_numpy()
    PT = TA[TA["has_prices"] & TA["car_1_1"].notna()][["accession", "ticker", "day0", "car_1_1"]].copy()
    PT["bh_1_1"] = [bhar(t, d) for t, d in zip(PT["ticker"], PT["day0"])]
    ok_ = PT[["car_1_1", "bh_1_1"]].dropna()
    print(f"Topline readouts: {len(PT):,} | buy-and-hold computed: {len(ok_):,} | correlation with CAR {ok_.corr().iloc[0, 1]:.3f}")
    print(f"Below -100%: CAR {int((ok_['car_1_1'] < -1).sum())}, buy-and-hold {int((ok_['bh_1_1'] < -1).sum())} | "
          f"min CAR {ok_['car_1_1'].min():+.0%}, min buy-and-hold {ok_['bh_1_1'].min():+.0%} | "
          f"max CAR {ok_['car_1_1'].max():+.0%}, max buy-and-hold {ok_['bh_1_1'].max():+.0%}")
    bq = PT["bh_1_1"].dropna()
    T11 = T.merge(PT[["accession", "bh_1_1"]], on="accession", how="left").dropna(subset=["bh_1_1"])
    def bclip(s, q):
        lo, hi = bq.quantile([q, 1 - q])
        return s.clip(lo, hi)
    def btrim(s, q, g):
        lo = s.groupby(g).transform(lambda x: x.quantile(q))
        hi = s.groupby(g).transform(lambda x: x.quantile(1 - q))
        return s.where((s >= lo) & (s <= hi))
    rows = []
    for samp, mask in [("all", np.ones(len(T11), bool)), ("clean (paper)", T11["clean"].to_numpy()),
                       ("clean without the 200% filter", T11["clean_nox"].to_numpy())]:
        for name in ["raw means", "clip 1% (paper)", "clip 2.5%", "trim 5% within outcome"]:
            if samp != "all" and name in ("clip 2.5%",):
                continue
            d = T11[mask].copy()
            d["y"] = {"raw means": d["bh_1_1"], "clip 1% (paper)": bclip(d["bh_1_1"], 0.01), "clip 2.5%": bclip(d["bh_1_1"], 0.025),
                      "trim 5% within outcome": btrim(d["bh_1_1"], 0.05, d["outcome"])}[name]
            s_, f_ = d.loc[d["outcome"] == "success", "y"].dropna(), d.loc[d["outcome"] == "failure", "y"].dropna()
            p = -f_.mean() / (s_.mean() - f_.mean()) if s_.mean() > f_.mean() else np.nan
            g, lo, hi, _ = firm_boot(d[["cik", "outcome", "y"]], gap_stat(np.mean))
            car = IM[(IM["sample"] == samp) & (IM["returns"] == name)]
            rows.append({"sample": samp, "returns": name, "R_up": s_.mean(), "R_down": f_.mean(), "implied": p, "gap": g, "lo": lo, "hi": hi,
                         "gap with CAR (Step 2)": car["gap"].iloc[0] if len(car) else np.nan, "n": len(d)})
    BH = pd.DataFrame(rows)
    print("\n=== Implied vs realized with buy-and-hold returns (gap = realized minus implied; 95% interval resampling firms) ===")
    print(BH.to_string(index=False, formatters={c: (lambda v: f"{v:+.1%}") for c in ["R_up", "R_down", "gap", "lo", "hi", "gap with CAR (Step 2)"]} |
                       {"implied": lambda v: f"{v:.0%}"}))
    BH.to_csv(os.path.join(NB10_DIR, "implied_bhar.csv"), index=False)
    r = BH[(BH["sample"] == "all") & (BH["returns"] == "clip 1% (paper)")].iloc[0]
    RESULTS.append({"check": "Implied-vs-realized gap with buy-and-hold returns, clip 1%", "estimate": f"{r['gap']:+.1%}",
                    "interval": f"[{r['lo']:+.1%}, {r['hi']:+.1%}]", "p": np.nan, "family": "",
                    "reading": f"CAR version {r['gap with CAR (Step 2)']:+.1%}; if close, Table 4 does not depend on summing returns"})


# %% [markdown]
# ## Step 12: Insiders the main test leaves out: 10% owners and gifts
# In biotech, 10% owners are often venture funds whose partners sit on the board; the fund files its own Form 4 as a 10% owner. Gifts of stock (code G) are a known way to shed shares before bad news, since the donor deducts the value at the gift date. Same design as Notebook 9: failure vs success, controls for the holder's usual activity (days -850 to -121), size and year, clustered by firm. Caveat: Notebook 9 kept Forms 4 only, and before the 2023 amendments gifts could wait for the annual Form 5, so gifts before 2023 are undercounted.


# %% INSIDERS2
IT = load("nb9/insider_trades.parquet", "Notebook 9")
if IT is not None and E is not None:
    IT["cik"] = pd.to_numeric(IT["cik"], errors="coerce")
    IT["trans_date"] = as_day(IT["trans_date"])
    IT["od"], IT["ten_pct"] = as_bool(IT["od"]).to_numpy(), as_bool(IT["ten_pct"]).to_numpy()
    fund = IT["ten_pct"] & ~IT["od"]
    out_ = col(IT, "ad", "").fillna("").astype(str).str.upper() != "A"
    KIND = {"10% owner sale": (IT["code"] == "S") & fund, "10% owner purchase": (IT["code"] == "P") & fund,
            "officer/director gift": (IT["code"] == "G") & IT["od"] & out_, "10% owner gift": (IT["code"] == "G") & fund & out_,
            "any sale, officers, directors or 10% owners": (IT["code"] == "S") & (IT["od"] | IT["ten_pct"])}
    for k, m_ in KIND.items():
        IT[k] = m_.to_numpy()
    X = IT[IT[list(KIND)].any(axis=1)]
    g_ = IT[IT["code"] == "G"]
    print("Gifts in the data by year (Forms 4 only): " + ", ".join(f"{y}: {n}" for y, n in g_["trans_date"].dt.year.value_counts().sort_index().items()))
    S12 = E[E["outcome"].isin(["success", "failure"])].dropna(subset=["log_dollar_vol"]).copy()
    S12["event_date"] = as_day(S12["event_date"])
    M12 = S12[["accession", "cik", "event_date"]].merge(X[["cik", "trans_date"] + list(KIND)], on="cik")
    M12["d"] = (M12["trans_date"] - M12["event_date"]).dt.days
    ten_firms = set(IT.loc[fund, "cik"])
    print(f"Readouts: {len(S12):,} | at firms with any 10% owner Form 4: {S12['cik'].isin(ten_firms).mean():.0%}")
    def flag(k, a, b):
        return S12["accession"].isin(set(M12.loc[M12[k] & M12["d"].between(a, b), "accession"])).astype(int).to_numpy()
    def count(k, a, b):
        return M12[M12[k] & M12["d"].between(a, b)].groupby("accession").size().reindex(S12["accession"]).fillna(0).to_numpy()
    rows = []
    for k in KIND:
        S12["y"], S12["base"] = flag(k, -60, -1), np.log1p(count(k, -850, -121))
        rec = {"activity": k}
        for oc in ["success", "failure"]:
            x = S12.loc[S12["outcome"] == oc, "y"]
            lo, hi = wilson(int(x.sum()), len(x))
            rec[oc] = f"{x.mean():.1%} ({lo:.1%}-{hi:.1%})"
        rec["readouts with it"] = int(S12["y"].sum())
        if S12["y"].sum() >= 10:
            try:
                m = fit("y ~ fail + base + log_dollar_vol + C(year)", S12, "logit")
                r = term(m, "fail")
                rec["odds ratio, failure"] = or_txt(r["b"], r["se"]) + f", p = {r['p']:.2f}"
            except Exception:
                fr = firth_logit("y ~ fail + base + log_dollar_vol + C(year)", S12)
                b, se = fr.loc["fail", "b"], fr.loc["fail", "se"]
                rec["odds ratio, failure"] = or_txt(b, se) + f", p = {2 * stats.norm.sf(abs(b / se)):.2f} (Firth)"
            r = term(fit("y ~ fail + base + log_dollar_vol + C(year)", S12), "fail")
            rec["failure minus success, LPM"] = pp_txt(r)
            for a in (-30, -120):
                S12["y2"] = flag(k, a, -1)
                r2 = term(fit("y2 ~ fail + base + log_dollar_vol + C(year)", S12), "fail")
                rec[f"LPM, days {a}..-1"] = f"{r2['b'] * 100:+.1f} pts (p = {r2['p']:.2f})"
            if k == "10% owner sale":
                RESULTS.append({"check": "10% owners selling before failure vs success (linear probability)", "estimate": f"{r['b'] * 100:+.1f} pts",
                                "interval": f"[{r['lo'] * 100:+.1f}, {r['hi'] * 100:+.1f}]", "p": r["p"], "family": "nulls",
                                "reading": "near zero: 10% owners (often venture funds) do not sell ahead of failures"})
            if k == "officer/director gift":
                RESULTS.append({"check": "Officer/director gifts before failure vs success (linear probability)", "estimate": f"{r['b'] * 100:+.1f} pts",
                                "interval": f"[{r['lo'] * 100:+.1f}, {r['hi'] * 100:+.1f}]", "p": r["p"], "family": "nulls",
                                "reading": "near zero: no gifting ahead of failures; positive: a leak channel"})
        rows.append(rec)
    IG = pd.DataFrame(rows)
    print("\n=== Share of topline readouts with the activity in days -60..-1, failure vs success (95% intervals) ===")
    with pd.option_context("display.max_colwidth", 60, "display.width", 300):
        for _, r in IG.iterrows():
            print(f"\n-- {r['activity']} | readouts with it: {r['readouts with it']} | success {r['success']} | failure {r['failure']}")
            for c_ in ["odds ratio, failure", "failure minus success, LPM", "LPM, days -30..-1", "LPM, days -120..-1"]:
                if c_ in r and isinstance(r[c_], str):
                    print(f"   {c_:28s} {r[c_]}")
    IG.to_csv(os.path.join(NB10_DIR, "insiders_ten_pct_gifts.csv"), index=False)
else:
    print("Skipped: needs nb9/insider_trades.parquet and nb9/readout_insiders.parquet")


# %% [markdown]
# ## Step 13: Friday by the press release's date
# The Friday test dates each readout by its 8-K's acceptance time. When the press release went out a day or more earlier, the news is that day's news. The dateline at the top of the release ("CAMBRIDGE, Mass., March 3, 2022") gives its date. Readouts whose release came out two or more days before the 8-K are also dropped in one variant, since the -1..+1 window misses them.


# %% FRIDAY_PR
if TX is not None and "F" in globals() and len(F):
    MON = (r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|"
           r"Nov(?:ember)?|Dec(?:ember)?)")
    DL = re.compile(MON + r"\.?\s+(\d{1,2}),?\s+((?:19|20)\d{2})")
    def pr_date(t):
        m = DL.search(str(t)[:600])
        if not m:
            return pd.NaT
        return pd.to_datetime(f"{m.group(1)[:3]} {int(m.group(2))} {m.group(3)}", format="%b %d %Y", errors="coerce")
    F13 = F.merge(TX[["accession", "text_raw"]], on="accession", how="left")
    F13["acc_day"] = as_day(F13["acceptance_et"]) if "acceptance_et" in F13 else F13["event_date"]
    F13["pr_day"] = as_day(F13["text_raw"].map(pr_date))
    F13["lag"] = (F13["acc_day"] - F13["pr_day"]).dt.days
    okd = F13["lag"].between(0, 7)
    print(f"Topline readouts: {len(F13):,} | dateline found and 0-7 days before the 8-K: {okd.sum():,} | same day {(F13['lag'] == 0).sum():,}, "
          f"1 day earlier {(F13['lag'] == 1).sum():,}, 2-7 days earlier {F13['lag'].between(2, 7).sum():,} | no usable dateline {(~okd).sum():,}")
    F13["fri_pr"] = np.where(okd, (F13["pr_day"].dt.dayofweek >= 4).astype(int), F13["fri"])
    print("\n8-K Friday (rows) against release Friday (columns):")
    print(pd.crosstab(F13["fri"], F13["fri_pr"]).to_string())
    print(f"Bad news on a Friday: by 8-K {int((F13['fri'] & F13['bad']).sum())}, by release date {int((F13['fri_pr'] & F13['bad']).sum())}")
    G13 = F13.dropna(subset=["log_dollar_vol"]).copy()
    rows = []
    for samp, d0 in [("all", G13), ("release within a day of the 8-K", G13[~G13["lag"].between(2, 7)])]:
        for y, lab in [("car_1_1", "immediate reaction, days -1..+1"), ("car_2_20", "drift, days +2..+20"), ("log_abn_volume", "log abnormal volume")]:
            if y not in d0 or d0[y].notna().sum() < 50:
                continue
            z = d0.dropna(subset=[y]).copy()
            if y != "log_abn_volume":
                z[y] = clip(z[y])
            r = term(fit(f"{y} ~ bad * fri_pr + log_dollar_vol + C(year)", z), "bad:fri_pr")
            rows.append({"sample": samp, "outcome": lab, "bad x Friday (release date)": r["b"], "lo": r["lo"], "hi": r["hi"], "p": r["p"], "n": len(z)})
            if samp == "all" and y == "car_1_1":
                RESULTS.append({"check": "Friday test with the release date: bad news x Friday, days -1..+1", "estimate": f"{r['b'] * 100:+.1f} pts",
                                "interval": f"[{r['lo'] * 100:+.1f}, {r['hi'] * 100:+.1f}]", "p": r["p"], "family": "nulls",
                                "reading": "if similar to Step 5: dating by the 8-K does not hide a Friday effect"})
    FT = pd.DataFrame(rows)
    print("\n=== Bad news x Friday, Friday dated by the press release (clustered by firm) ===")
    print(FT.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    FT.to_csv(os.path.join(NB10_DIR, "friday_release_date.csv"), index=False)
    try:
        r = term(fit("fri_pr ~ bad + log_dollar_vol + C(year)", G13, "logit"), "bad")
        print(f"\nBad news released on Fridays (logit, release date): odds ratio {or_txt(r['b'], r['se'])}, p = {r['p']:.2f}")
    except Exception as e:
        print(f"  Friday choice: did not fit ({e})")
else:
    print("Skipped: needs nb5/texts.parquet and Step 5's Friday sample")


# %% [markdown]
# ## Step 15: Sections 6 and 7 with reliable disclosure matches only
# The hand-coded match sample shows that matches through a drug name shared by six or more of the firm's trials are often wrong. This step drops those matches, reclassifies each stop by its first remaining disclosure (a stop with none left counts as never disclosed), and reruns the main comparisons. A second variant leaves out the stops whose classification rested on such a match.


# %% RELIABLE
HITS = load("stop_disclosure_hits.parquet", "Notebook 3b")
SDF2 = load("stop_disclosure.parquet", "Notebook 3b")
if HITS is not None and SDF2 is not None:
    HITS = HITS[as_bool(HITS["stop_hit"]).to_numpy()].copy()
    HITS["name_trials"] = pd.to_numeric(col(HITS, "name_trials", 1), errors="coerce").fillna(1)
    HITS["days"] = pd.to_numeric(HITS["days"], errors="coerce")
    REL = HITS[HITS["name_trials"] <= 5]
    S15 = SDF2[as_bool(SDF2["in_sample"]).to_numpy()].copy()
    S15["registry_date"] = as_day(S15["registry_date"])
    S15["cik"] = pd.to_numeric(S15["cik"], errors="coerce")
    def first_of(h):
        if not len(h):
            return pd.DataFrame(columns=["days_first", "first_channel"])
        g = h.groupby(["stop_id", "channel"])["days"].min().unstack()
        return pd.DataFrame({"days_first": g.min(axis=1), "first_channel": g.idxmin(axis=1)})
    FA, FR = first_of(HITS[HITS["stop_id"].isin(S15["stop_id"])]), first_of(REL[REL["stop_id"].isin(S15["stop_id"])])
    S15 = S15.join(FR.add_suffix("_rel"), on="stop_id").join(FA.add_suffix("_all"), on="stop_id")
    S15["shared_based"] = S15["days_first_all"].notna() & (S15["days_first_rel"].isna() | (S15["days_first_rel"] > S15["days_first_all"]))
    def outc(fc):
        fc = fc.fillna("none").astype(str)
        return np.select([fc.eq("8-K"), fc.str.contains(r"10-Q|10-K|20-F|40-F"), fc.eq("none")], ["8-K first", "10-Q/10-K first", "never"],
                         "other filing first")
    S15["out_all"], S15["out_rel"] = outc(S15["first_channel_all"]), outc(S15["first_channel_rel"])
    print(f"In-sample stops: {len(S15):,} | never disclosed, all matches: {(S15['out_all'] == 'never').mean():.0%} | "
          f"reliable matches only: {(S15['out_rel'] == 'never').mean():.0%}")
    print(f"Stops whose first disclosure rests on a shared drug name: {int(S15['shared_based'].sum())} "
          f"({S15['shared_based'].mean():.0%}); of these, left with no disclosure at all: "
          f"{int((S15['shared_based'] & S15['days_first_rel'].isna()).sum())}")
    print("\nFirst channel, all matches (rows) against reliable only (columns):")
    print(pd.crosstab(S15["out_all"], S15["out_rel"]).to_string())

    # Section 7: per-stop exits
    if XS is not None and "stop_id" in XS:
        X7 = XS.merge(S15[["stop_id", "out_rel", "shared_based"]], on="stop_id", how="inner")
        X7["cik"] = pd.to_numeric(X7["cik"], errors="coerce")
        X7["w"] = 1 / X7.groupby("cik")["cik"].transform("size")
        ctr = [c for c in ["small", "material", "late"] if c in X7 and X7[c].nunique() > 1]
        rhs = "grp" + "".join(f" + {c}" for c in ctr) + " + year"
        print("\n=== Section 7, never disclosed minus 8-K first (per stop; linear probability, clustered by firm; Firth odds ratio) ===")
        for lab, gcol, keep in [("as in the paper", "outcome", np.ones(len(X7), bool)), ("reliable matches only", "out_rel", np.ones(len(X7), bool)),
                                ("leaving out stops classified by a shared name", "outcome", ~X7["shared_based"].to_numpy())]:
            z0 = X7[keep & X7[gcol].isin(["8-K first", "never"]).to_numpy()].copy()
            z0["grp"] = (z0[gcol] == "never").astype(int)
            for y in ["notice_365", "fail_365"]:
                z = z0.dropna(subset=[y] + ctr + ["year"]).copy()
                r = term(fit(f"{y} ~ {rhs}", z), "grp")
                rw = term(fit(f"{y} ~ {rhs}", z, weights="w"), "grp")
                try:
                    fr = firth_logit(f"{y} ~ {rhs}", z)
                    orr = or_txt(fr.loc["grp", "b"], fr.loc["grp", "se"])
                except Exception:
                    orr = "n/a"
                print(f"  {lab:46s} {y:11s} n = {len(z):5,} | {pp_txt(r)} | each firm once {rw['b'] * 100:+.1f} pts (p = {rw['p']:.2f}) | Firth OR {orr}")
                if lab.startswith("reliable") and y == "notice_365":
                    RESULTS.append({"check": "Notice within a year, never minus 8-K first, reliable matches only (LPM)", "estimate": f"{r['b'] * 100:+.1f} pts",
                                    "interval": f"[{r['lo'] * 100:+.1f}, {r['hi'] * 100:+.1f}]", "p": r["p"], "family": "",
                                    "reading": "same sign and size as the paper: false matches do not drive Section 7"})

    # Section 6: registry-date reaction by group, events rebuilt from reliable matches
    if SE is not None and "nct_ids" in SE:
        tim = S15.set_index("nct_id")
        before_all = (tim["days_first_all"] < 0)
        before_rel = (tim["days_first_rel"] < 0)
        def grp(ids, b):
            v = [bool(b.get(n, False)) for n in str(ids).split(";") if n]
            return "filing first" if v and all(v) else ("registry first or only" if not any(v) else "mixed batch")
        SE["group_rel"] = [grp(i, before_rel.to_dict()) for i in SE["nct_ids"]]
        print("\n=== Section 6, registry posting date, days 0..+1 ===")
        for g in ["registry first or only", "filing first"]:
            a, b = mean_ci(SE[SE["group"] == g], "car01"), mean_ci(SE[SE["group_rel"] == g], "car01")
            print(f"  {g:24s} as in the paper: {fmt_ci(a)}\n  {'':24s} reliable only:   {fmt_ci(b)}")
else:
    print("Skipped: needs stop_disclosure_hits.parquet and stop_disclosure.parquet (Notebook 3b)")


# %% [markdown]
# ## Step 16: Robustness checks
# (a) which months the registry data cover; (b) the implied probability measured from the price before the run-up; (c) how long before the 8-K the press release came out, and the run-up without any early release; (d) the notice test with 8-K filer status; (e) sponsor size counted at the stop date instead of today; (f) whether a late registry date could hide a reaction; (g) the saved figure files, zipped for download. Run Settings, Step 0 and Steps 1 to 6 and 13 first.


# %% REVIEW
import glob
CTG_DIR = os.path.join(PROJECT_DIR, "ctgov")
def boot_line(d, stat):
    g, lo, hi, p = firm_boot(d, stat)
    return g, lo, hi, p, f"{g:+.1%} [{lo:+.1%}, {hi:+.1%}], p = {p:.3f}"

# ---- (a) coverage of the registry data ----
snaps = sorted(glob.glob(os.path.join(CTG_DIR, "aact_monthly", "aact_*.parquet")))
if snaps:
    mo = [os.path.basename(p)[5:11] for p in snaps]
    print(f"(a) AACT monthly snapshots: {len(snaps)} files, first {mo[0][:4]}-{mo[0][4:]}, last {mo[-1][:4]}-{mo[-1][4:]}")
else:
    print("(a) No AACT snapshot files in", os.path.join(CTG_DIR, "aact_monthly"))
SE_ALL = load("stop_events.parquet", "Notebook 4")
if SE_ALL is not None:
    rd = as_day(SE_ALL["registry_date"])
    print(f"    Stop events (firm-days): {len(SE_ALL):,}, registry dates {rd.min():%Y-%m-%d} to {rd.max():%Y-%m-%d}")
    print("    by year: " + " | ".join(f"{y} {n}" for y, n in rd.dt.year.value_counts().sort_index().items()))

# ---- (b) implied probability from the price before the run-up (days -10..+1) ----
if RO is not None:
    B = TA[TA["has_prices"] & TA["outcome"].isin(["success", "failure"])].copy()
    B = B.dropna(subset=["car_1_1", "car_m10_m2"])
    allp = TA[TA["has_prices"]].copy()
    allp["w10"] = allp["car_m10_m2"] + allp["car_1_1"]
    def qc(s, ref):
        lo, hi = ref.quantile([WINSOR, 1 - WINSOR])
        return s.clip(lo, hi)
    B["y_1"] = qc(B["car_1_1"], allp["car_1_1"].dropna())
    B["y_10"] = qc(B["car_m10_m2"] + B["car_1_1"], allp["w10"].dropna())
    def gap_on(y):
        def f(d):
            s, fl = d.loc[d["outcome"] == "success", y].mean(), d.loc[d["outcome"] == "failure", y].mean()
            if not s > fl:
                return np.nan
            return (d["outcome"] == "success").mean() - (-fl / (s - fl))
        return f
    print("\n(b) Implied probability of success, same readouts, two starting prices (clipped 1%; 95% interval resampling firms)")
    for y, lab in [("y_1", "price at day -2 (paper)"), ("y_10", "price at day -11, before the run-up")]:
        s, fl = B.loc[B["outcome"] == "success", y].mean(), B.loc[B["outcome"] == "failure", y].mean()
        p_ = -fl / (s - fl)
        g, lo, hi, p, txt = boot_line(B[["cik", "outcome", y]], gap_on(y))
        print(f"    {lab:38s} U {s:+.1%}, D {fl:+.1%}, implied {p_:.0%}, realized {(B['outcome'] == 'success').mean():.0%}, gap {txt} | n = {len(B):,}")
        if y == "y_10":
            RESULTS.append({"check": "Implied-vs-realized gap, measured from day -11", "estimate": f"{g:+.1%}", "interval": f"[{lo:+.1%}, {hi:+.1%}]",
                            "p": p, "family": "", "reading": "close to the paper's gap: the run-up does not distort the odds test"})

# ---- (c) release dates relative to the 8-K ----
if "F13" in globals():
    L = F13["lag"]
    cat = pd.cut(L, [-np.inf, -1, 0, 1, 7, 30, np.inf], labels=["after the 8-K (date error)", "same day", "1 day before",
                                                              "2-7 days before", "8-30 days before", "more than 30 days before"])
    print(f"\n(c) Press-release dateline against the 8-K, topline readouts (n = {len(F13):,})")
    print("    " + " | ".join(f"{k} {v}" for k, v in cat.value_counts(sort=False).items()) + f" | no dateline {int(L.isna().sum())}")
    far = F13[L > 7].sort_values("lag", ascending=False)
    if len(far):
        print("    Largest gaps (check a few by hand: an old date in the text, or a real early release?):")
        print("    " + far[["accession", "lag"]].head(10).to_string(index=False).replace("\n", "\n    "))
    if "RU" in globals() and RU is not None:
        R16 = RU.merge(F13[["accession", "lag"]], on="accession", how="left")
        sf = lambda z: z.loc[z["outcome"] == "success", "y"].mean() - z.loc[z["outcome"] == "failure", "y"].mean()
        print("    Run-up, successes minus failures, days -10..-2 minus placebo (clipped):")
        for lab, keep in [("all", pd.Series(True, index=R16.index)), ("release no more than 1 day before the 8-K", ~(R16["lag"] > 1)),
                          ("release dated on the 8-K day or the day before", R16["lag"].between(0, 1))]:
            d = R16[keep & R16["outcome"].isin(["success", "failure"])][["cik", "outcome", "xrun_10"]].dropna().rename(columns={"xrun_10": "y"})
            d["y"] = clip(d["y"])
            g, lo, hi, p, txt = boot_line(d, sf)
            print(f"      {lab:48s} {txt} | n = {len(d):,}")
            if lab.startswith("release no more"):
                RESULTS.append({"check": "Run-up without any release more than a day before the 8-K", "estimate": f"{g:+.2%}",
                                "interval": f"[{lo:+.2%}, {hi:+.2%}]", "p": p, "family": "",
                                "reading": "close to 3.8: early releases of any length do not create the run-up"})
else:
    print("\n(c) Skipped: run Step 13 first")

# ---- (d) and (e): per-stop exits with 8-K filer status and sponsor size at the stop date ----
LEADS = None
X16 = load("nb8/exit_outcomes_stops.parquet", "Notebook 8")
if X16 is not None:
    X16["cik"] = pd.to_numeric(X16["cik"], errors="coerce")
    if "files_8k" not in X16:
        CC = load("nb7/channel_choice.parquet", "Notebook 7")
        if CC is not None and "stop_id" in X16 and "files_8k" in CC:
            X16 = X16.merge(CC[["stop_id", "files_8k"]].drop_duplicates("stop_id"), on="stop_id", how="left")
    # sponsor size at the stop date: lead trials first posted on or before that day
    cur_p = sorted(glob.glob(os.path.join(CTG_DIR, "industry_trials_current*.parquet")), key=os.path.getmtime)
    tf_p = os.path.join(CTG_DIR, "trial_firm.parquet")
    LEADS = None
    if cur_p and os.path.exists(tf_p):
        cur = pd.read_parquet(cur_p[-1], columns=["nct_id", "first_posted"])
        tf = pd.read_parquet(tf_p)
        ld = tf[tf["role"] == "lead"][["nct_id", "cik"]].drop_duplicates().merge(cur, on="nct_id", how="left")
        ld["fp"] = as_day(ld["first_posted"])
        LEADS = {int(c): np.sort(g["fp"].dropna().to_numpy()) for c, g in ld.groupby("cik")}
        def n_at(c, d):
            a = LEADS.get(int(c)) if pd.notna(c) else None
            return int(np.searchsorted(a, np.datetime64(d), side="right")) if a is not None and pd.notna(d) else np.nan
        X16["lead_at_stop"] = [n_at(c, d) for c, d in zip(X16["cik"], as_day(X16["registry_date"]))] if "registry_date" in X16 else np.nan
        X16["small_at"] = pd.to_numeric(X16["lead_at_stop"], errors="coerce").between(1, SMALL_LEAD).astype(int)
        print(f"\n(e) Small sponsor (1-{SMALL_LEAD} lead trials) counted today (rows) against at the stop date (columns), per stop:")
        print("    " + pd.crosstab(X16["small"], X16["small_at"]).to_string().replace("\n", "\n    "))
    else:
        print("\n(e) Skipped: needs ctgov/industry_trials_current*.parquet and ctgov/trial_firm.parquet (Notebook 2)")

    Z = X16[X16["outcome"].isin(["8-K first", "never"])].copy()
    Z["grp"] = (Z["outcome"] == "never").astype(int)
    specs = [("paper's controls", "small", Z)]
    if "files_8k" in Z:
        print("\n(d) Never-disclosed share by 8-K filer status, per stop: " + " | ".join(
            f"{'files 8-Ks' if k == 1 else 'no 8-Ks'} {v:.0%}" for k, v in X16.assign(nv=X16["outcome"].eq("never")).groupby("files_8k")["nv"].mean().items()))
        specs += [("plus a control for filing 8-Ks", "small + files_8k", Z), ("8-K filers only", "small", Z[Z["files_8k"] == 1])]
    if "small_at" in Z:
        specs += [("size counted at the stop date", "small_at", Z)]
    print("\n    Never disclosed against 8-K first, per stop (controls: size, safety/efficacy reason, late phase, year)")
    for y in ["notice_365", "fail_365"]:
        if y not in Z:
            continue
        for lab, sz, d in specs:
            rhs = "grp + " + sz + " + material + late + year"
            z = d.dropna(subset=[y, "material", "late", "year"] + [v.strip() for v in sz.split("+")]).copy()
            z[y] = z[y].astype(int)
            try:
                r = term(fit(f"{y} ~ {rhs}", z), "grp")
                fr = firth_logit(f"{y} ~ {rhs}", z)
                b, se_ = fr.loc["grp", "b"], fr.loc["grp", "se"]
                print(f"    {y:11s} {lab:32s} LPM {pp_txt(r)} | Firth OR {or_txt(b, se_)} | n = {len(z):,}, events = {int(z[y].sum())}")
                if y == "notice_365" and lab == "8-K filers only":
                    RESULTS.append({"check": "Notice within a year, never minus 8-K first, 8-K filers only (LPM)", "estimate": f"{r['b'] * 100:+.1f} pts",
                                    "interval": f"[{r['lo'] * 100:+.1f}, {r['hi'] * 100:+.1f}]", "p": r["p"], "family": "",
                                    "reading": "still negative: missing 6-K notices of foreign filers do not create the gap"})
            except Exception as e:
                print(f"    {y:11s} {lab:32s} did not fit ({type(e).__name__})")

# ---- (e, continued) registry reaction for small sponsors counted at the stop date ----
if "SE" in globals() and SE is not None and LEADS is not None:
    SE["lead_at_stop"] = [n_at(c, d) for c, d in zip(SE["cik"], as_day(SE["registry_date"]))]
    sm_at = pd.to_numeric(SE["lead_at_stop"], errors="coerce").between(1, SMALL_LEAD).to_numpy()
    rf_ = SE["group"].eq("registry first or only").to_numpy()
    mat_ = as_bool(col(SE, "material_reason", False)).to_numpy()
    quiet_ = ~as_bool(col(SE, "same_day_8k", False)).to_numpy() & ~as_bool(col(SE, "near_readout", False)).to_numpy()
    print("    Registry-first posting, days 0..+1, small sponsor at the stop date:")
    for lab, m_ in [("no other news that day", rf_ & sm_at & quiet_), ("safety or efficacy, no other news", rf_ & sm_at & quiet_ & mat_)]:
        print(f"      {lab:36s} {fmt_ci(mean_ci(SE[m_], 'car01'))}")

# ---- (f) could a late registry date hide a reaction? the month before the posting date ----
if "SE" in globals() and SE is not None and CAL is not None:
    t2 = col(SE, "ticker", np.nan)
    if t2.isna().all() and RO is not None:
        t2 = SE["cik"].map(RO.dropna(subset=["ticker"]).drop_duplicates("cik").set_index("cik")["ticker"])
    SE["tick16"] = t2
    spos = {c: np.sort(CAL.searchsorted(g.to_numpy())) for c, g in SE.groupby("cik")["day0"]}
    rows = []
    for r in SE.itertuples():
        a = px_arrays(r.tick16) if isinstance(r.tick16, str) else None
        if a is None:
            continue
        ar, live = a
        i0 = int(CAL.searchsorted(r.day0))
        if i0 < 30 or i0 >= len(CAL) - 2:
            continue
        rec = {"event_id": r.Index, "pre": wsum(ar, i0, -21, -1), "win": wsum(ar, i0, -21, 1)}
        cand = np.arange(max(i0 - 500, 30), min(i0 + 500, len(CAL) - 2))
        ev = spos.get(r.cik, np.array([]))
        if len(ev):
            cand = cand[~(np.abs(ev[None, :] - cand[:, None]) <= 60).any(axis=1)]
        cand = cand[live[cand]]
        if len(cand) >= 2:
            pk = rng.choice(cand, size=min(N_PLACEBO, len(cand)), replace=False)
            rec["pre_p"] = np.nanmean([wsum(ar, int(j), -21, -1) for j in pk])
            rec["win_p"] = np.nanmean([wsum(ar, int(j), -21, 1) for j in pk])
        rows.append(rec)
    W16 = SE.join(pd.DataFrame(rows).set_index("event_id"), how="inner")
    W16["x_pre"], W16["x_win"] = W16["pre"] - col(W16, "pre_p", np.nan), W16["win"] - col(W16, "win_p", np.nan)
    W16["x_pre"], W16["x_win"] = clip(W16["x_pre"]), clip(W16["x_win"])
    rf_ = W16["group"].eq("registry first or only")
    mat_ = as_bool(col(W16, "material_reason", False))
    print("\n(f) Registry posting date, the month before it (days -21..-1) and through day +1, minus placebo dates (clipped):")
    for lab, m_ in [("registry first or only", rf_), ("  safety or efficacy reason", rf_ & mat_), ("filing first (already disclosed)", W16["group"].eq("filing first"))]:
        a_, b_ = mean_ci(W16[m_.to_numpy()], "x_pre"), mean_ci(W16[m_.to_numpy()], "x_win")
        print(f"    {lab:34s} days -21..-1: {fmt_ci(a_)}")
        print(f"    {'':34s} days -21..+1: {fmt_ci(b_)}")
        if lab == "registry first or only":
            RESULTS.append({"check": "Registry-first stops, days -21..+1 around the posting date, minus placebo", "estimate": f"{b_['mean']:+.2%}",
                            "interval": f"[{b_['lo']:+.2%}, {b_['hi']:+.2%}]", "p": b_["p"], "family": "nulls",
                            "reading": "compare with filing-first: similar means the usual pre-stop slide, much lower means a hidden reaction"})

# ---- (g) the figures the notebooks saved, newest version of each, zipped ----
import zipfile
FIG_DIR = os.path.join(FEAT_DIR, "figures")
want = ["fig6_readout_car", "fig10_llm_move", "fig11_implied_pos", "fig13_drift", "fig15_exits", "fig16_insiders", "fig17_friday", "fig9_oos_r2"]
pngs = glob.glob(os.path.join(FIG_DIR, "*.png"))
pick = {}
for w in want:
    c_ = [p for p in pngs if os.path.basename(p).startswith(w)]
    if c_:
        pick[w] = max(c_, key=os.path.getmtime)
print(f"\n(g) Figures in {FIG_DIR}:")
for w in want:
    p = pick.get(w)
    print(f"    {w:20s} " + (f"{os.path.basename(p)} (saved {pd.Timestamp(os.path.getmtime(p), unit='s'):%Y-%m-%d %H:%M})" if p else "not found"))
if pick:
    zp = os.path.join(NB10_DIR, "paper_figures.zip")
    with zipfile.ZipFile(zp, "w") as zf:
        for w, p in pick.items():
            zf.write(p, os.path.basename(p))
    print("    Zipped to", zp)
    try:
        from google.colab import files
        files.download(zp)
    except Exception:
        pass


# %% [markdown]
# ## Step 17: The paper's figures in one style
# Redraws every data figure in the paper with one palette, one font and the legends above the plot, at 300 dpi, and zips them for download. Needs Settings, Step 0, Step 1 and Step 5 in this session; it reads the other notebooks' saved files from Drive.


# %% PAPERFIGS
import glob, zipfile
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PAPER_FIG = os.path.join(FEAT_DIR, "figures", "paper")
os.makedirs(PAPER_FIG, exist_ok=True)
NAVY, BRICK, NAVY_L, BRICK_L = "#1f4e79", "#b5452f", "#9fb8d3", "#e4ab9d"
G1, G2, G3, INK = "#3b3b3b", "#8a8a8a", "#c4c4c4", "#262626"
CHAN = {"8-K first": G1, "10-Q/10-K first": G2, "other filing first": G3, "never": NAVY}
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": G1, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True, "xtick.color": G1, "ytick.color": G1,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.6, "legend.frameon": False, "legend.fontsize": 8, "figure.facecolor": "white",
    "savefig.facecolor": "white", "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left"})
MADE = []

def legend_top(ax, ncol, handles=None, labels=None):
    kw = dict(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=ncol, frameon=False, borderaxespad=0,
              handlelength=1.8, columnspacing=1.4, handletextpad=0.5)
    if handles is not None:
        ax.legend(handles, labels, **kw)
    else:
        ax.legend(**kw)

def save(fig, name):
    p = os.path.join(PAPER_FIG, name)
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    MADE.append(p)
    print("  saved", name)

def window(ticker, day0, a, b):
    x = px_arrays(ticker) if isinstance(ticker, str) else None
    if x is None or pd.isna(day0):
        return None
    ar, live = x
    i0 = int(CAL.searchsorted(day0))
    if i0 + a < 0 or i0 + b >= len(CAL):
        return None
    v = ar[i0 + a:i0 + b + 1]
    return v if np.isfinite(v).any() else None

def cum_path(rows):
    m = pd.DataFrame(rows).apply(clip)
    return m.mean(axis=0).fillna(0).cumsum().to_numpy() * 100

def read_csv(rel):
    p = os.path.join(FEAT_DIR, rel)
    return pd.read_csv(p) if os.path.exists(p) else None

# ---- readout paths by tone ----
if RO is not None and CAL is not None:
    D1 = np.arange(-10, 21)
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    for oc, lab, c in [("success", "Positive", NAVY), ("failure", "Negative", BRICK), ("mixed", "Mixed", G2), ("unclear", "Unclear", G3)]:
        rows = [w for w in (window(r.ticker, r.day0, -10, 20) for r in TA[TA["has_prices"] & (TA["outcome"] == oc)].itertuples()) if w is not None]
        if rows:
            ax.plot(D1, cum_path(rows), color=c, label=f"{lab} (n = {len(rows):,})")
    ax.axhline(0, color=G1, linewidth=0.6)
    ax.axvline(0, color=G2, linewidth=0.7, linestyle=(0, (3, 2)))
    ax.set_xlim(-10, 20)
    ax.set_xlabel("Trading days from the readout 8-K")
    ax.set_ylabel("Cumulative abnormal return (%)")
    legend_top(ax, 4)
    save(fig, "fig_readout_paths.png")

# ---- implied against realized probability, by subsample ----
IP = read_csv("nb6/implied_pos.csv")
if IP is not None:
    s = IP[(IP["window"] == "days -1..+1") & ~as_bool(IP["mixed_as_failure"].astype(str).str.lower().eq("true"))].reset_index(drop=True)
    NAMES = {"all": "All topline readouts", "both labels agree": "Tone and language model agree", "clean events": "Clean events",
             "era 2010-2015": "2010 to 2015", "era 2016-2020": "2016 to 2020", "era 2021-2025": "2021 to 2025", "2018 and later": "2018 and later",
             "phase 3": "Phase 3", "phase 2": "Phase 2", "phase 1": "Phase 1", "phase unknown": "Phase not given",
             "small firms (dollar volume)": "Small firms", "mid firms (dollar volume)": "Mid-size firms", "large firms (dollar volume)": "Large firms"}
    fig, ax = plt.subplots(figsize=(6.5, 0.24 * len(s) + 0.9))
    y = np.arange(len(s))[::-1]
    h1 = ax.errorbar(s["implied"] * 100, y, xerr=[(s["implied"] - s["implied_lo"]) * 100, (s["implied_hi"] - s["implied"]) * 100],
                     fmt="o", color=NAVY, ecolor=NAVY_L, elinewidth=2.2, markersize=5, capsize=0)
    h2, = ax.plot(s["realized"] * 100, y, "D", color=BRICK, markersize=4.5)
    ax.set_yticks(y)
    ax.set_yticklabels([NAMES.get(g, g) for g in s["group"]])
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Probability of success (%)")
    legend_top(ax, 2, [h1, h2], ["Implied by prices, with 95% interval", "Realized share of successes"])
    save(fig, "fig_implied.png")

# ---- first filing that discloses a stop, by stated reason ----
CC = load("nb7/channel_choice.parquet", "Notebook 7")
if CC is not None and {"reason", "outcome"} <= set(CC.columns):
    ORDER = ["8-K first", "10-Q/10-K first", "other filing first", "never"]
    LBL = {"8-K first": "8-K first", "10-Q/10-K first": "10-Q or 10-K first", "other filing first": "Other filing first", "never": "Never disclosed"}
    tab = pd.crosstab(CC["reason"].fillna("not given"), CC["outcome"], normalize="index").reindex(columns=ORDER, fill_value=0) * 100
    nn = CC["reason"].fillna("not given").value_counts()
    tab = tab.sort_values("never", ascending=False)
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(tab) + 0.8))
    left = np.zeros(len(tab))
    for k in ORDER:
        v = tab[k].to_numpy()
        ax.barh(np.arange(len(tab)), v, left=left, height=0.62, color=CHAN[k], edgecolor="white", linewidth=0.8, label=LBL[k])
        for i, (l_, w_) in enumerate(zip(left, v)):
            if w_ >= 8:
                ax.text(l_ + w_ / 2, i, f"{w_:.0f}%", ha="center", va="center", fontsize=7.5,
                        color="white" if k in ("8-K first", "never") else INK)
        left += v
    ax.set_yticks(np.arange(len(tab)))
    ax.set_yticklabels([f"{r.capitalize()} (n = {nn[r]:,})" for r in tab.index])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of stops (%)")
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    legend_top(ax, 4)
    save(fig, "fig_reasons.png")

# ---- drift after stops, events against placebo dates ----
S7, D7 = load("nb7/drift_registry.parquet", "Notebook 7"), load("nb7/drift_disclosure.parquet", "Notebook 7")
if S7 is not None and D7 is not None and CAL is not None:
    for X in (S7, D7):
        X["cik"] = pd.to_numeric(X["cik"], errors="coerce")
        X["day0"] = as_day(X["day0"]) if "day0" in X else as_day(X["registry_date"] if "registry_date" in X else X["first_date"])
    stop_days = pd.concat([S7[["cik", "day0"]], D7[["cik", "day0"]]])
    SP = {c: np.sort(CAL.searchsorted(g.to_numpy())) for c, g in stop_days.groupby("cik")["day0"]}
    RP = {c: np.sort(CAL.searchsorted(g.to_numpy())) for c, g in RO.groupby("cik")["day0"]} if RO is not None else {}
    D2 = np.arange(-20, 121)
    def placebo_rows(t, c, day0):
        x = px_arrays(t) if isinstance(t, str) else None
        if x is None:
            return []
        ar, live = x
        i0 = int(CAL.searchsorted(day0))
        cand = np.arange(max(i0 - 500, 20), min(i0 + 500, len(CAL) - 121))
        cand = cand[live[cand] & (np.abs(cand - i0) > 120)]
        for P_, w_ in ((SP.get(c), 120), (RP.get(c), 30)):
            if P_ is not None and len(P_) and len(cand):
                cand = cand[~(np.abs(P_[None, :] - cand[:, None]) <= w_).any(axis=1)]
        if len(cand) < 2:
            return []
        out = []
        for j in rng.choice(cand, size=min(N_PLACEBO, len(cand)), replace=False):
            v = ar[int(j) - 20:int(j) + 121]
            if np.isfinite(v).any():
                out.append(v)
        return out
    groups = [("Registry posting, never disclosed", S7[col(S7, "sgroup", "") == "registry first, never disclosed"], NAVY),
              ("First disclosed in an 8-K", D7[col(D7, "chan", "") == "8-K"], G1),
              ("First disclosed in a 10-Q or 10-K", D7[col(D7, "chan", "") == "10-Q/10-K"], G2)]
    fig, ax = plt.subplots(figsize=(6.5, 3.1))
    hs, ls = [], []
    for lab, X, c in groups:
        ev = [w for w in (window(r.ticker, r.day0, -20, 120) for r in X.itertuples()) if w is not None]
        pl = [v for r in X.itertuples() for v in placebo_rows(r.ticker, r.cik, r.day0)]
        if ev:
            h, = ax.plot(D2, cum_path(ev), color=c)
            hs.append(h); ls.append(f"{lab} (n = {len(ev):,})")
        if pl:
            ax.plot(D2, cum_path(pl), color=c, linewidth=0.9, linestyle=(0, (3, 2)), alpha=0.9)
    hs.append(Line2D([], [], color=G2, linewidth=0.9, linestyle=(0, (3, 2)))); ls.append("Same stocks on placebo dates (dashed)")
    ax.axhline(0, color=G1, linewidth=0.6)
    ax.axvline(0, color=G2, linewidth=0.7, linestyle=(0, (3, 2)))
    ax.set_xlim(-20, 120)
    ax.set_xlabel("Trading days from the event")
    ax.set_ylabel("Cumulative abnormal return (%)")
    legend_top(ax, 2, hs, ls)
    save(fig, "fig_drift.png")

# ---- outcomes within a year, and failure exits over three years ----
XS7 = load("nb8/exit_outcomes_stops.parquet", "Notebook 8")
GRP = [("8-K first", "8-K first"), ("10-Q/10-K first", "10-Q or 10-K first"), ("never", "Never disclosed")]
WRAP = {"8-K first": "8-K\nfirst", "10-Q or 10-K first": "10-Q or\n10-K first", "Never disclosed": "Never\ndisclosed"}
if XS7 is not None:
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
    for ax, (y, t_) in zip(axes, [("fail_365", "Failure exit within a year"), ("notice_365", "Listing-rule notice within a year")]):
        vals, ns = [], []
        for g, _ in GRP:
            z = XS7.loc[XS7["outcome"] == g, y].dropna()
            vals.append(z.mean() * 100); ns.append(len(z))
        b = ax.bar(range(3), vals, width=0.62, color=[CHAN[g] for g, _ in GRP])
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + max(vals) * 0.03, f"{v:.0f}%", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{WRAP[l]}\n(n = {n:,})" for (_, l), n in zip(GRP, ns)], fontsize=7.5)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_title(t_)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Share of stops (%)")
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig_exits.png")
    if {"exit_date", "failure_firm", "registry_date"} <= set(XS7.columns):
        def km(t, event, horizon=1095):
            t, event = np.asarray(t, float), np.asarray(event, bool)
            grid = np.arange(0, horizon + 1)
            s_, out, at_risk = 1.0, [], len(t)
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
                    s_ *= 1 - deaths / at_risk
                out.append(1 - s_)
            return grid, np.array(out)
        P2 = XS7.copy()
        P2["exit_date"], P2["registry_date"] = as_day(P2["exit_date"]), as_day(P2["registry_date"])
        end = P2["exit_date"].where(P2["exit_date"].notna(), pd.Timestamp.today().normalize())
        P2["t"] = (end - P2["registry_date"]).dt.days.clip(lower=0)
        P2["ev"] = P2["exit_date"].notna() & as_bool(P2["failure_firm"])
        fig, ax = plt.subplots(figsize=(6.5, 2.8))
        for g, l in GRP:
            x = P2[P2["outcome"] == g]
            if len(x) >= 10:
                grid, ci = km(x["t"], x["ev"])
                ax.step(grid / 365.25, ci * 100, where="post", color=CHAN[g], label=f"{l} (n = {len(x):,})")
        ax.set_xlim(0, 3)
        ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
        ax.set_xlabel("Years since the registry posted the stop")
        ax.set_ylabel("Stops followed by a failure exit (%)")
        legend_top(ax, 3)
        save(fig, "fig_km.png")

# ---- insider sales around readouts ----
E9, T9 = load("nb9/readout_insiders.parquet", "Notebook 9"), load("nb9/insider_trades.parquet", "Notebook 9")
if E9 is not None and T9 is not None:
    E9["event_date"], T9["trans_date"] = as_day(E9["event_date"]), as_day(T9["trans_date"])
    M9 = E9[["accession", "cik", "event_date"]].merge(T9[(T9["code"] == "S") & as_bool(T9["od"])][["cik", "trans_date"]], on="cik")
    M9["d"] = (M9["trans_date"] - M9["event_date"]).dt.days
    D3 = np.arange(-180, 91)
    sl = M9[M9["d"].between(D3[0], D3[-1])][["accession", "d"]].drop_duplicates().merge(E9[["accession", "outcome"]], on="accession")
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    for oc, lab, c in [("success", "Successful readouts", NAVY), ("failure", "Failed readouts", BRICK)]:
        n = int((E9["outcome"] == oc).sum())
        if n:
            daily = sl[sl["outcome"] == oc].groupby("d")["accession"].nunique().reindex(D3, fill_value=0) / n * 100
            ax.plot(D3, daily.rolling(15, center=True, min_periods=5).mean(), color=c, label=f"{lab} (n = {n:,})")
    ax.axvline(0, color=G2, linewidth=0.7, linestyle=(0, (3, 2)))
    ax.set_xlim(-180, 90)
    ax.set_xlabel("Calendar days from the topline readout")
    ax.set_ylabel("Readouts with an officer or\ndirector sale that day (%)")
    legend_top(ax, 2)
    save(fig, "fig_insiders.png")

# ---- Friday against Monday to Thursday ----
if "F" in globals() and F is not None and len(F):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))
    for ax, (y, t_) in zip(axes, [("car_1_1", "Days -1 to +1"), ("car_2_20", "Days +2 to +20")]):
        vals, errs, labs = [], [], []
        for b_, bl in [(1, "Bad"), (0, "Good")]:
            for f_, fl in [(0, "Mon-Thu"), (1, "Friday")]:
                x = F[(F["bad"] == b_) & (F["fri"] == f_)][y].dropna()
                if len(x) > 20:
                    x = x.clip(*x.quantile([0.01, 0.99]))
                vals.append(x.mean() * 100)
                errs.append(1.96 * x.std(ddof=1) / np.sqrt(len(x)) * 100 if len(x) > 1 else np.nan)
                labs.append(f"{bl}\n{fl}\n(n = {len(x)})")
        ax.bar(range(4), vals, width=0.66, color=[BRICK, BRICK_L, NAVY, NAVY_L], yerr=errs,
               error_kw=dict(ecolor=G1, elinewidth=0.8, capsize=2.5))
        ax.axhline(0, color=G1, linewidth=0.6)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labs, fontsize=7.5)
        ax.tick_params(axis="x", length=0)
        ax.set_title(t_)
    axes[0].set_ylabel("Mean abnormal return (%)")
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig_friday.png")

# ---- move size by the language model's expected move ----
tf = sorted(glob.glob(os.path.join(FEAT_DIR, "nb5", "readout_text_features*.parquet")), key=os.path.getmtime)
if tf:
    d5 = pd.read_parquet(tf[-1])
    if {"exp_move", "abs_car", "event_date"} <= set(d5.columns):
        d5["event_date"] = as_day(d5["event_date"])
        CUT = pd.Timestamp("2023-12-31")
        BINS, LABS = [-0.1, 5, 10, 20, 40, 100], ["0-5%", "5-10%", "10-20%", "20-40%", "40%+"]
        fig, ax = plt.subplots(figsize=(6.5, 2.7))
        for lab, rows, c in [("Up to the model's training cutoff", d5[d5["event_date"] <= CUT], G2),
                             ("After the cutoff", d5[d5["event_date"] > CUT], NAVY)]:
            r2 = rows.dropna(subset=["exp_move", "abs_car"])
            g = r2.groupby(pd.cut(r2["exp_move"], BINS, labels=LABS), observed=True)["abs_car"].agg(["mean", "size"])
            g = g[g["size"] >= 5]
            if len(g):
                ax.plot([LABS.index(i) for i in g.index], g["mean"] * 100, color=c, marker="o", markersize=5,
                        label=f"{lab} (n = {int(g['size'].sum()):,})")
        ax.set_xticks(range(len(LABS)))
        ax.set_xticklabels(LABS)
        ax.set_xlim(-0.3, len(LABS) - 0.7)
        ax.set_xlabel("Expected size of the move, as read by the language model")
        ax.set_ylabel("Mean absolute return,\ndays -1 to +1 (%)")
        legend_top(ax, 2)
        save(fig, "fig_llm_bins.png")

if MADE:
    zp = os.path.join(NB10_DIR, "paper_figures_v2.zip")
    with zipfile.ZipFile(zp, "w") as zf:
        for p in MADE:
            zf.write(p, os.path.basename(p))
    print(f"\n{len(MADE)} figures zipped to {zp}")
    try:
        from google.colab import files
        files.download(zp)
    except Exception:
        pass


# %% [markdown]
# ## Step 18: Robustness of the run-up, and exits with acquisitions as a competing risk
# Checks on the central result. Same sample as the headline run-up (positive and negative topline readouts with prices, without releases issued 2 to 7 days before the 8-K when Step 13 has run).
# (a) beta-adjusted returns, with the beta to XBI from days -250 to -30, net of placebo dates, and a market model with alpha and beta; (b) intervals that allow for market-wide shocks, clustered by firm and calendar week or month, and a bootstrap that resamples calendar months; (c) five placebo seeds and 20 placebo dates; (d) an earlier window, days -30 to -11, where an outcome gap should not appear; (e) splits by phase, size and volatility; (f) readouts where the rule tone and the language model agree; (g) readouts split by registry warnings before them, with the match rate; (h) the 30-day event-time figure; (i) cumulative incidence of failure exits with acquisitions as a competing event (Aalen-Johansen).
# Run Settings, Step 0, Steps 1 and 3 (and 13 if you can) first.


# %% ROBUST
import glob, zipfile
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PAPER_FIG = os.path.join(FEAT_DIR, "figures", "paper")
os.makedirs(PAPER_FIG, exist_ok=True)
NAVY, BRICK, NAVY_L, BRICK_L = "#1f4e79", "#b5452f", "#9fb8d3", "#e4ab9d"
G1, G2, G3, INK = "#3b3b3b", "#8a8a8a", "#c4c4c4", "#262626"
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": G1, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True, "xtick.color": G1, "ytick.color": G1,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.6, "legend.frameon": False, "legend.fontsize": 8, "figure.facecolor": "white",
    "savefig.facecolor": "white"})
R18_FIGS, R18 = [], []

def legend_top18(ax, ncol, handles=None, labels=None):
    kw = dict(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=ncol, frameon=False, borderaxespad=0,
              handlelength=1.8, columnspacing=1.4, handletextpad=0.5)
    (ax.legend(handles, labels, **kw) if handles is not None else ax.legend(**kw))

def save18(fig, name):
    p = os.path.join(PAPER_FIG, name)
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    R18_FIGS.append(p)
    print("  saved", name)

def gap(z):
    return z.loc[z["outcome"] == "success", "y"].mean() - z.loc[z["outcome"] == "failure", "y"].mean()

def gap_row(label, d, v, extra=""):
    z = d[["cik", "outcome", v]].dropna().rename(columns={v: "y"})
    if z["outcome"].nunique() < 2 or min((z["outcome"] == "success").sum(), (z["outcome"] == "failure").sum()) < 8:
        print(f"  {label:58s} too few readouts (n = {len(z)})")
        return None
    z["y"] = clip(z["y"])
    g, lo, hi, p = firm_boot(z, gap)
    ns, nf = int((z["outcome"] == "success").sum()), int((z["outcome"] == "failure").sum())
    print(f"  {label:58s} {g:+.2%} [{lo:+.2%}, {hi:+.2%}] p = {p:.3f} | successes {ns}, failures {nf}{extra}")
    R18.append({"check": label, "gap": g, "lo": lo, "hi": hi, "p": p, "successes": ns, "failures": nf})
    return g, lo, hi, p

if RU is None or CAL is None:
    print("Skipped: run Steps 1 and 3 first")
else:
    # ---- the headline sample ----
    B18 = RU[RU["outcome"].isin(["success", "failure"])].copy()
    if "F13" in globals() and F13 is not None and "lag" in F13:
        B18 = B18.merge(F13[["accession", "lag"]], on="accession", how="left")
        B18 = B18[~B18["lag"].between(2, 7)].copy()
        print(f"Headline sample: {len(B18):,} positive and negative topline readouts, without releases 2-7 days before the 8-K")
    else:
        print(f"Headline sample: {len(B18):,} positive and negative topline readouts (Step 13 not run, so early releases are kept)")
    rpos = {c: np.sort(CAL.searchsorted(g.to_numpy())) for c, g in RO.groupby("cik")["day0"]}
    bc = bclose.reindex(CAL).to_numpy()
    _BS = {}
    def bspan(t, live):
        if t not in _BS:
            L = np.flatnonzero(live)
            out = np.full(len(CAL), np.nan)
            out[L[1:]] = bc[L[1:]] / bc[L[:-1]] - 1
            _BS[t] = out
        return _BS[t]
    gens = {s: np.random.default_rng(SEED + s) for s in range(1, 6)}
    g20 = np.random.default_rng(SEED + 20)
    OFF = np.arange(-30, -1)                                              # days -30..-2 for the figure
    recs, paths = [], []
    for r in B18.itertuples():
        a = px_arrays(r.ticker)
        if a is None:
            continue
        ar, live = a
        i0 = int(CAL.searchsorted(r.day0))
        if i0 < 260 or i0 >= len(CAL) - 2:
            continue
        rm = bspan(r.ticker, live)
        est = slice(i0 - 250, i0 - 29)
        y_, x_ = ar[est] + rm[est], rm[est]
        ok = np.isfinite(y_) & np.isfinite(x_)
        beta = alpha = np.nan
        if ok.sum() >= 100:
            X = np.column_stack([np.ones(ok.sum()), x_[ok]])
            alpha, beta = np.linalg.lstsq(X, y_[ok], rcond=None)[0]
        badj = ar + (1 - beta) * rm if np.isfinite(beta) else None
        mm = ar + (1 - beta) * rm - alpha if np.isfinite(beta) else None
        cand = np.arange(max(i0 - 500, 30), min(i0 + 500, len(CAL) - 30))
        ev = rpos.get(r.cik, np.array([]))
        if len(ev):
            cand = cand[~(np.abs(ev[None, :] - cand[:, None]) <= 30).any(axis=1)]
        cand = cand[live[cand]]
        if len(cand) < 2:
            continue
        rec = {"accession": r.accession, "beta": beta,
               "run_10": wsum(ar, i0, -10, -2), "run_e30": wsum(ar, i0, -30, -11), "run_e20": wsum(ar, i0, -20, -11)}
        if badj is not None:
            rec["brun_10"] = wsum(badj, i0, -10, -2)
            rec["mrun_10"] = wsum(mm, i0, -10, -2)
        pk1 = None
        for s, gs in gens.items():
            pk = gs.choice(cand, size=min(N_PLACEBO, len(cand)), replace=False)
            rec[f"xrun_s{s}"] = rec["run_10"] - np.nanmean([wsum(ar, int(j), -10, -2) for j in pk])
            if s == 1:
                pk1 = pk
                rec["xrun_e30"] = rec["run_e30"] - np.nanmean([wsum(ar, int(j), -30, -11) for j in pk])
                rec["xrun_e20"] = rec["run_e20"] - np.nanmean([wsum(ar, int(j), -20, -11) for j in pk])
                if badj is not None:
                    rec["xbrun_10"] = rec["brun_10"] - np.nanmean([wsum(badj, int(j), -10, -2) for j in pk])
        pk = g20.choice(cand, size=min(20, len(cand)), replace=False)
        rec["xrun_n20"] = rec["run_10"] - np.nanmean([wsum(ar, int(j), -10, -2) for j in pk])
        recs.append(rec)
        ev_d = ar[i0 + OFF]
        pl_d = np.nanmean(np.vstack([ar[int(j) + OFF] for j in pk1]), axis=0)
        paths.append({"accession": r.accession, **{int(k): v for k, v in zip(OFF, ev_d - pl_d)}})
    D18 = B18.drop(columns=[c for c in B18.columns if c in ("run_10",)]).merge(pd.DataFrame(recs), on="accession", how="inner")
    print(f"Readouts with an estimation window and placebo dates: {len(D18):,} | median beta to XBI {D18['beta'].median():.2f}")

    print("\n(a) Return benchmark, successes minus failures, days -10..-2 (clipped; 95% interval resampling firms)")
    gap_row("market-adjusted, net of placebo dates (Step 3)", D18, "xrun_10")
    gap_row("beta-adjusted, net of placebo dates", D18, "xbrun_10")
    gap_row("market model, alpha and beta from days -250..-30", D18, "mrun_10")

    print("\n(b) Inference that allows for market-wide shocks (market-adjusted, net of placebo dates)")
    z = D18[["cik", "day0", "outcome", "xrun_10"]].dropna().copy()
    z["y"] = clip(z["xrun_10"])
    z["succ"] = (z["outcome"] == "success").astype(float)
    z["week"] = pd.to_datetime(z["day0"]).dt.to_period("W").astype(str)
    z["month"] = pd.to_datetime(z["day0"]).dt.to_period("M").astype(str)
    for lab, gcols in [("firm", ["cik"]), ("firm and calendar week", ["cik", "week"]), ("firm and calendar month", ["cik", "month"])]:
        gg = np.column_stack([pd.factorize(z[c].astype(str))[0] for c in gcols]) if len(gcols) > 1 else pd.factorize(z["cik"].astype(str))[0]
        m = smf.ols("y ~ succ", z).fit(cov_type="cluster", cov_kwds={"groups": gg})
        b, se = float(m.params["succ"]), float(m.bse["succ"])
        print(f"  clustered by {lab:28s} {b:+.2%} [{b - 1.96 * se:+.2%}, {b + 1.96 * se:+.2%}] p = {float(m.pvalues['succ']):.3f}")
        R18.append({"check": f"Run-up gap, clustered by {lab}", "gap": b, "lo": b - 1.96 * se, "hi": b + 1.96 * se,
                    "p": float(m.pvalues["succ"]), "successes": int(z["succ"].sum()), "failures": int((1 - z["succ"]).sum())})
    months = z["month"].unique()
    idx = {mo: np.flatnonzero(z["month"].to_numpy() == mo) for mo in months}
    bs = []
    for _ in range(N_BOOT):
        ii = np.concatenate([idx[mo] for mo in rng.choice(months, len(months))])
        bs.append(gap(z.iloc[ii]))
    bs = np.asarray([v for v in bs if np.isfinite(v)])
    g0 = gap(z)
    pb = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    print(f"  bootstrap resampling calendar months     {g0:+.2%} [{np.percentile(bs, 2.5):+.2%}, {np.percentile(bs, 97.5):+.2%}] p = {pb:.3f}"
          f" | {len(months)} months")
    R18.append({"check": "Run-up gap, bootstrap resampling calendar months", "gap": g0, "lo": np.percentile(bs, 2.5),
                "hi": np.percentile(bs, 97.5), "p": pb, "successes": int(z["succ"].sum()), "failures": int((1 - z["succ"]).sum())})

    print("\n(c) Placebo draws")
    for s in range(1, 6):
        gap_row(f"seed {SEED + s}, {N_PLACEBO} placebo dates", D18, f"xrun_s{s}")
    gap_row("20 placebo dates", D18, "xrun_n20")

    print("\n(d) An earlier window, where no outcome gap should appear")
    gap_row("days -30..-11, net of placebo dates", D18, "xrun_e30")
    gap_row("days -20..-11, net of placebo dates", D18, "xrun_e20")

    print("\n(e) Splits (market-adjusted, net of placebo dates)")
    ph = pd.to_numeric(col(D18, "phase", np.nan), errors="coerce")
    sz = pd.to_numeric(col(D18, "log_dollar_vol", np.nan), errors="coerce")
    vo = pd.to_numeric(col(D18, "pre_vol", np.nan), errors="coerce")
    splits = [("phase 3", ph >= 3), ("phase 1 or 2", ph < 3), ("phase unknown", ph.isna()),
              ("smaller firms (dollar volume below median)", sz < sz.median()), ("larger firms (dollar volume above median)", sz >= sz.median()),
              ("calmer stocks (volatility below median)", vo < vo.median()), ("more volatile stocks (volatility above median)", vo >= vo.median())]
    for lab, mk in splits:
        gap_row(lab, D18[mk.fillna(False).to_numpy()], "xrun_10")
    for lab, hi_ in [("size", sz >= sz.median()), ("volatility", vo >= vo.median()), ("phase 3", ph >= 3)]:
        zz = D18.assign(hi=hi_.astype(float), succ=(D18["outcome"] == "success").astype(float))[["cik", "xrun_10", "hi", "succ"]].dropna()
        zz["y"] = clip(zz["xrun_10"])
        try:
            m = fit("y ~ succ * hi", zz)
            t_ = term(m, "succ:hi")
            print(f"  difference in the gap by {lab:10s} {t_['b']:+.2%} [{t_['lo']:+.2%}, {t_['hi']:+.2%}] p = {t_['p']:.2f}")
        except Exception as e:
            print(f"  interaction by {lab} did not fit: {e}")

    print("\n(f) Readouts where the rule tone and the language model agree")
    tf = sorted(glob.glob(os.path.join(FEAT_DIR, "nb5", "readout_text_features*.parquet")), key=os.path.getmtime)
    lf = [f for f in sorted(glob.glob(os.path.join(FEAT_DIR, "nb5", "llm_*_p3*.parquet")), key=os.path.getmtime)
          if "dryrun" not in f and "unmasked" not in f]
    PM = None
    for f in tf[::-1] + lf[::-1]:
        try:
            x = pd.read_parquet(f)
        except Exception:
            continue
        if {"accession", "p_met"} <= set(x.columns):
            PM = x[["accession", "p_met"]].drop_duplicates("accession")
            print(f"  P(met) from {os.path.basename(f)}")
            break
    if PM is None:
        print("  No language-model P(met) found in features/nb5: skipped")
    else:
        DF = D18.merge(PM, on="accession", how="left")
        agree = ((DF["outcome"] == "success") & (DF["p_met"] >= 0.5)) | ((DF["outcome"] == "failure") & (DF["p_met"] < 0.5))
        print(f"  readouts with P(met): {DF['p_met'].notna().sum():,} of {len(DF):,} | agree with the tone: {int(agree.sum()):,}")
        gap_row("run-up gap, tone and language model agree", DF[agree], "xrun_10")
        T4 = TA[TA["has_prices"] & TA["outcome"].isin(["success", "failure"]) & TA["car_1_1"].notna()].copy()
        lo_, hi_ = TA.loc[TA["has_prices"], "car_1_1"].quantile([WINSOR, 1 - WINSOR])
        T4["y"] = T4["car_1_1"].clip(lo_, hi_)
        T4 = T4.merge(PM, on="accession", how="left")
        def igap(x):
            U, D = x.loc[x["outcome"] == "success", "y"].mean(), x.loc[x["outcome"] == "failure", "y"].mean()
            return (x["outcome"] == "success").mean() - (-D / (U - D)) if U > D else np.nan
        for lab, x in [("all positive and negative (Table 4, first row)", T4),
                       ("tone and language model agree", T4[((T4["outcome"] == "success") & (T4["p_met"] >= 0.5)) |
                                                          ((T4["outcome"] == "failure") & (T4["p_met"] < 0.5))])]:
            g, lo, hi, p = firm_boot(x[["cik", "outcome", "y"]].reset_index(drop=True), igap)
            U, D = x.loc[x["outcome"] == "success", "y"].mean(), x.loc[x["outcome"] == "failure", "y"].mean()
            print(f"  implied-realized gap, {lab:46s} realized {(x['outcome'] == 'success').mean():.0%}, implied {-D / (U - D):.0%}, "
                  f"gap {g * 100:+.0f} points [{lo * 100:+.0f}, {hi * 100:+.0f}] | n = {len(x):,}")
            R18.append({"check": f"Implied-realized gap, {lab}", "gap": g, "lo": lo, "hi": hi, "p": p,
                        "successes": int((x["outcome"] == "success").sum()), "failures": int((x["outcome"] == "failure").sum())})

    print("\n(g) Registry warnings before the readout (Notebook 6 links readouts to their own trials)")
    FL = load("nb6/readout_flags.parquet", "Notebook 6")
    nct = col(D18, "nct_ids", "").fillna("").astype(str).str.contains(r"NCT\d{8}")
    print(f"  readouts citing at least one NCT number: {int(nct.sum()):,} of {len(D18):,} ({nct.mean():.0%})")
    if FL is not None and "any_flag" in FL:
        DG = D18.merge(FL[["accession", "any_flag"]].drop_duplicates("accession"), on="accession", how="left")
        linked = DG["any_flag"].notna()
        print(f"  linked to a registered industry trial, a year or more into the registry archive: {int(linked.sum()):,} "
              f"({linked.mean():.0%} of the headline sample) | with a warning in the year before: {int((DG['any_flag'] == 1).sum()):,}")
        gap_row("linked readouts", DG[linked], "xrun_10")
        gap_row("linked, no registry warning in the year before", DG[DG["any_flag"] == 0], "xrun_10")
        gap_row("linked, a registry warning in the year before", DG[DG["any_flag"] == 1], "xrun_10")
        gap_row("not linked to a registry record", DG[~linked], "xrun_10")
        zz = DG[linked].assign(succ=lambda d: (d["outcome"] == "success").astype(float))[["cik", "xrun_10", "any_flag", "succ"]].dropna()
        zz["y"] = clip(zz["xrun_10"])
        if ((zz["succ"] == 0) & (zz["any_flag"] == 1)).sum() < 8:
            print("  too few failures after a registry warning to test the difference in the gap")
        else:
          try:
            t_ = term(fit("y ~ succ * any_flag", zz), "succ:any_flag")
            print(f"  difference in the gap, warning minus none: {t_['b']:+.2%} [{t_['lo']:+.2%}, {t_['hi']:+.2%}] p = {t_['p']:.2f}")
          except Exception as e:
            print(f"  interaction did not fit: {e}")

    print("\n(h) 30-day event-time figure")
    PA = pd.DataFrame(paths).merge(D18[["accession", "cik", "outcome"]], on="accession")
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    for oc, c, cl, lab in [("success", NAVY, NAVY_L, "Successes"), ("failure", BRICK, BRICK_L, "Failures")]:
        x = PA[PA["outcome"] == oc].reset_index(drop=True)
        M = x[list(OFF)].apply(clip)
        path = M.mean(axis=0).fillna(0).cumsum().to_numpy() * 100
        firms = x["cik"].unique()
        fidx = {f_: np.flatnonzero(x["cik"].to_numpy() == f_) for f_ in firms}
        bsp = []
        for _ in range(200):
            ii = np.concatenate([fidx[f_] for f_ in rng.choice(firms, len(firms))])
            bsp.append(M.iloc[ii].mean(axis=0).fillna(0).cumsum().to_numpy() * 100)
        bsp = np.vstack(bsp)
        ax.fill_between(OFF, np.percentile(bsp, 2.5, axis=0), np.percentile(bsp, 97.5, axis=0), color=cl, alpha=0.45, linewidth=0)
        ax.plot(OFF, path, color=c, label=f"{lab} (n = {len(x):,})")
        print(f"  {lab}: cumulative abnormal return net of placebo dates, day -30 to day -11 {path[OFF == -11][0]:+.2f}%, "
              f"to day -2 {path[-1]:+.2f}%")
    ax.axhline(0, color=G2, linewidth=0.6)
    ax.axvline(-10, color=G2, linewidth=0.6, linestyle=(0, (3, 2)))
    ax.set_xlim(-30, -2)
    ax.set_xticks([-30, -25, -20, -15, -10, -5, -2])
    ax.set_xlabel("Trading days before the readout (day 0 is the first session that can react to the 8-K)")
    ax.set_ylabel("Cumulative abnormal return,\nnet of placebo dates (%)")
    legend_top18(ax, 2)
    save18(fig, "fig_runup30.png")

    pd.DataFrame(R18).to_csv(os.path.join(NB10_DIR, "robust_runup.csv"), index=False)

# ---- (i) exits with acquisitions as a competing risk ----
print("\n(i) Failure exits with acquisitions as a competing event (Aalen-Johansen), per stop")
XS18 = load("nb8/exit_outcomes_stops.parquet", "Notebook 8")
if XS18 is not None and {"exit_date", "failure_firm", "registry_date", "outcome"} <= set(XS18.columns):
    X = XS18.copy()
    X["exit_date"], X["registry_date"] = as_day(X["exit_date"]), as_day(X["registry_date"])
    end = X["exit_date"].where(X["exit_date"].notna(), pd.Timestamp.today().normalize())
    X["t"] = (end - X["registry_date"]).dt.days.clip(lower=0)
    fail = X["exit_date"].notna() & as_bool(X["failure_firm"]).to_numpy()
    X["st"] = np.where(fail, 1, np.where(X["exit_date"].notna(), 2, 0))
    if "exit_type" in X:
        print("  competing exits by type: " + ", ".join(f"{k} {v}" for k, v in X.loc[X["st"] == 2, "exit_type"].value_counts().items()))
    def aj(t, st, grid):
        t, st = np.asarray(t, float), np.asarray(st, int)
        times = np.unique(t[st > 0])
        S, F1, F2, out1, out2, k = 1.0, 0.0, 0.0, [], [], 0
        cif = {}
        for u in times:
            n = (t >= u).sum()
            d1, d2 = ((t == u) & (st == 1)).sum(), ((t == u) & (st == 2)).sum()
            if n:
                F1 += S * d1 / n
                F2 += S * d2 / n
                S *= 1 - (d1 + d2) / n
            cif[u] = (F1, F2)
        keys = np.array(list(cif.keys()))
        for g_ in grid:
            j = np.searchsorted(keys, g_, side="right") - 1
            f = cif[keys[j]] if j >= 0 else (0.0, 0.0)
            out1.append(f[0]); out2.append(f[1])
        return np.array(out1), np.array(out2)
    def km1(t, st, g_):
        t, ev = np.asarray(t, float), np.asarray(st) == 1
        s_ = 1.0
        for u in np.unique(t[ev & (t <= g_)]):
            n = (t >= u).sum()
            s_ *= 1 - ((t == u) & ev).sum() / n
        return 1 - s_
    grid = np.arange(0, 1096)
    GR = [("8-K first", "8-K first", G1), ("10-Q/10-K first", "10-Q or 10-K first", G2), ("never", "Never disclosed", NAVY)]
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)
    rows = []
    for g_, lab, c in GR:
        x = X[X["outcome"] == g_]
        if len(x) < 10:
            continue
        f1, f2 = aj(x["t"], x["st"], grid)
        axs[0].step(grid / 365.25, f1 * 100, where="post", color=c, label=f"{lab} (n = {len(x):,})")
        axs[1].step(grid / 365.25, f2 * 100, where="post", color=c)
        for yr in (1, 2, 3):
            rows.append({"group": lab, "years": yr, "failure, competing risks": f1[365 * yr], "acquisition or other exit": f2[365 * yr],
                         "failure, Kaplan-Meier": km1(x["t"], x["st"], 365 * yr), "n": len(x)})
    CI = pd.DataFrame(rows)
    print(CI.to_string(index=False, float_format=lambda v: f"{v:.1%}" if v < 1 else f"{v:.0f}"))
    def dgap(d, yr):
        a = d[d["outcome"] == "never"]; b_ = d[d["outcome"] == "8-K first"]
        if len(a) < 5 or len(b_) < 5:
            return np.nan
        return aj(a["t"], a["st"], [365 * yr])[0][0] - aj(b_["t"], b_["st"], [365 * yr])[0][0]
    for yr in (1, 3):
        g, lo, hi, p = firm_boot(X[["cik", "outcome", "t", "st"]], lambda d: dgap(d, yr), n=min(N_BOOT, 300))
        print(f"  never disclosed minus 8-K first, failure within {yr} year{'s' if yr > 1 else ''}: {g * 100:+.1f} points "
              f"[{lo * 100:+.1f}, {hi * 100:+.1f}], p = {p:.2f}")
        R18.append({"check": f"Failure exit within {yr} years, never minus 8-K first, competing risks", "gap": g, "lo": lo, "hi": hi, "p": p})
    for ax, ttl in zip(axs, ["Failure exit", "Acquisition or other exit"]):
        ax.set_xlim(0, 3)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlabel("Years since the registry posted the stop")
        ax.text(0.02, 0.97, ttl, transform=ax.transAxes, va="top", ha="left", fontsize=8.5, fontweight="bold", color=INK)
    axs[0].set_ylabel("Stops followed by the exit (%)")
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.subplots_adjust(wspace=0.12, top=0.86)
    fig.legend(h_, l_, loc="lower left", bbox_to_anchor=(axs[0].get_position().x0, 0.88), ncol=3, frameon=False,
               borderaxespad=0, handlelength=1.8, columnspacing=1.4, handletextpad=0.5)
    save18(fig, "fig_cif.png")
    CI.to_csv(os.path.join(NB10_DIR, "robust_cif.csv"), index=False)
    pd.DataFrame(R18).to_csv(os.path.join(NB10_DIR, "robust_runup.csv"), index=False)
else:
    print("  needs exit_outcomes_stops.parquet with exit_date and failure_firm (Notebook 8): skipped")

zp = os.path.join(NB10_DIR, "robust_figures.zip")
with zipfile.ZipFile(zp, "w") as zf:
    for p in R18_FIGS:
        zf.write(p, os.path.basename(p))
print(f"\nSaved robust_runup.csv, robust_cif.csv and {len(R18_FIGS)} figures in {os.path.basename(zp)} (in {NB10_DIR})")
try:
    from google.colab import files
    files.download(zp)
except Exception:
    pass


# %% [markdown]
# ## Step 19: Every run-up number with 20 placebo dates per readout
# Step 18 showed that the run-up net of placebo dates moves with the random draw of 5 placebo dates (2.7 to 4.0 points across draws). This step redraws 20 placebo dates per readout with a fixed seed, so the benchmark is stable, and recomputes every run-up number the paper reports. It also shows the spread across five draws of 20, the Holm correction with the new p-value, the 30-day figure, and Figure 2 with the tone-and-language-model row.
# Run Settings, Step 0, Steps 1 and 3 first (Steps 4, 6 and 8 too if you want the Holm correction).


# %% RUNUP20
import glob, zipfile
import matplotlib as mpl
import matplotlib.pyplot as plt

N_PL20 = 20
PAPER_FIG = os.path.join(FEAT_DIR, "figures", "paper")
os.makedirs(PAPER_FIG, exist_ok=True)
NAVY, BRICK, NAVY_L, BRICK_L = "#1f4e79", "#b5452f", "#9fb8d3", "#e4ab9d"
G1, G2, G3, INK = "#3b3b3b", "#8a8a8a", "#c4c4c4", "#262626"
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": G1, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True, "xtick.color": G1, "ytick.color": G1,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.6, "legend.frameon": False, "legend.fontsize": 8, "figure.facecolor": "white",
    "savefig.facecolor": "white"})
R19, R19_FIGS = [], []

def legend_top19(ax, ncol, handles=None, labels=None):
    kw = dict(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=ncol, frameon=False, borderaxespad=0,
              handlelength=1.8, columnspacing=1.4, handletextpad=0.5)
    (ax.legend(handles, labels, **kw) if handles is not None else ax.legend(**kw))

def save19(fig, name):
    p = os.path.join(PAPER_FIG, name)
    fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    R19_FIGS.append(p)
    print("  saved", name)

def gap19(z):
    return z.loc[z["outcome"] == "success", "y"].mean() - z.loc[z["outcome"] == "failure", "y"].mean()

def med19(z):
    return z.loc[z["outcome"] == "success", "y"].median() - z.loc[z["outcome"] == "failure", "y"].median()

def row19(label, d, v, stat=gap19):
    z = d[["cik", "outcome", v]].dropna().rename(columns={v: "y"})
    ns, nf = int((z["outcome"] == "success").sum()), int((z["outcome"] == "failure").sum())
    if min(ns, nf) < 8:
        print(f"  {label:66s} too few readouts (successes {ns}, failures {nf})")
        return None
    z["y"] = clip(z["y"])
    g, lo, hi, p = firm_boot(z, stat)
    print(f"  {label:66s} {g:+.2%} [{lo:+.2%}, {hi:+.2%}] p = {p:.3f} | successes {ns}, failures {nf}")
    R19.append({"check": label, "gap": g, "lo": lo, "hi": hi, "p": p, "successes": ns, "failures": nf})
    return g, lo, hi, p

if RO is None or CAL is None:
    print("Skipped: run Steps 0 and 1 first")
else:
    # ---- readouts, release dates and meeting mentions ----
    B = TA[TA["has_prices"] & TA["outcome"].isin(["success", "failure"])].copy()
    TXr = load("nb5/texts.parquet", "Notebook 5")
    if TXr is not None:
        TXr = TXr.drop_duplicates("accession", keep="last")[["accession", "text_raw"]]
        MON = (r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|"
               r"Nov(?:ember)?|Dec(?:ember)?)")
        DL = re.compile(MON + r"\.?\s+(\d{1,2}),?\s+((?:19|20)\d{2})")
        def pr_date(t):
            m = DL.search(str(t)[:600])
            return pd.to_datetime(f"{m.group(1)[:3]} {int(m.group(2))} {m.group(3)}", format="%b %d %Y", errors="coerce") if m else pd.NaT
        CONF_KW = re.compile(r"\b(?:abstracts?|posters?|late[- ]breaking|oral presentation|podium presentation|annual meeting|"
                             r"scientific sessions?|congress|symposium|(?:medical|scientific|international) conference)\b", re.I)
        CONF_AT = re.compile(r"\b[Pp]resent(?:ed|s|ing|ation|ations)?\s+(?:at|during)\s+(?:the\s+)?(?:\d{4}\s+)?[A-Z][A-Za-z]")
        B = B.merge(TXr, on="accession", how="left")
        acc_day = as_day(B["acceptance_et"]) if "acceptance_et" in B else B["event_date"]
        B["lag"] = (acc_day - as_day(B["text_raw"].map(pr_date))).dt.days
        B["meeting"] = B["text_raw"].map(lambda t: bool(CONF_KW.search(t) or CONF_AT.search(t)) if isinstance(t, str) else np.nan).astype(object)
    else:
        B["lag"], B["meeting"] = np.nan, np.nan
    B["early"] = B["lag"].between(2, 7)
    print(f"Positive and negative topline readouts with prices: {len(B):,} | release issued 2-7 days before the 8-K: {int(B['early'].sum())} "
          f"| dated on the 8-K day or the day before: {int(B['lag'].between(0, 1).sum())}")

    # ---- returns and 20 placebo dates per readout, five draws ----
    rpos = {c: np.sort(CAL.searchsorted(g.to_numpy())) for c, g in RO.groupby("cik")["day0"]}
    bc = bclose.reindex(CAL).to_numpy()
    _BS19 = {}
    def bspan19(t, live):
        if t not in _BS19:
            L = np.flatnonzero(live)
            out = np.full(len(CAL), np.nan)
            out[L[1:]] = bc[L[1:]] / bc[L[:-1]] - 1
            _BS19[t] = out
        return _BS19[t]
    gens = [np.random.default_rng(SEED + 100 + s) for s in range(5)]
    OFF = np.arange(-30, -1)
    W = {"10": (-10, -2), "5": (-5, -2), "e30": (-30, -11), "e20": (-20, -11)}
    recs, paths = [], []
    for r in B.itertuples():
        a = px_arrays(r.ticker)
        if a is None:
            continue
        ar, live = a
        i0 = int(CAL.searchsorted(r.day0))
        if i0 < 260 or i0 >= len(CAL) - 2:
            continue
        cand = np.arange(max(i0 - 500, 30), min(i0 + 500, len(CAL) - 30))
        ev = rpos.get(r.cik, np.array([]))
        if len(ev):
            cand = cand[~(np.abs(ev[None, :] - cand[:, None]) <= 30).any(axis=1)]
        cand = cand[live[cand]]
        if len(cand) < 2:
            continue
        rm = bspan19(r.ticker, live)
        est = slice(i0 - 250, i0 - 29)
        y_, x_ = ar[est] + rm[est], rm[est]
        ok = np.isfinite(y_) & np.isfinite(x_)
        beta = alpha = np.nan
        if ok.sum() >= 100:
            alpha, beta = np.linalg.lstsq(np.column_stack([np.ones(ok.sum()), x_[ok]]), y_[ok], rcond=None)[0]
        rec = {"accession": r.accession, **{f"run_{k}": wsum(ar, i0, *w) for k, w in W.items()}}
        for s, gs in enumerate(gens):
            pk = gs.choice(cand, size=min(N_PL20, len(cand)), replace=False)
            rec[f"x10_s{s}"] = rec["run_10"] - np.nanmean([wsum(ar, int(j), -10, -2) for j in pk])
            if s == 0:
                for k, w in W.items():
                    rec[f"x{k}"] = rec[f"run_{k}"] - np.nanmean([wsum(ar, int(j), *w) for j in pk])
                if np.isfinite(beta):
                    badj = ar + (1 - beta) * rm
                    rec["xb10"] = wsum(badj, i0, -10, -2) - np.nanmean([wsum(badj, int(j), -10, -2) for j in pk])
                    rec["mm10"] = wsum(badj - alpha, i0, -10, -2)
                pl = np.nanmean(np.vstack([ar[int(j) + OFF] for j in pk]), axis=0)
                paths.append({"accession": r.accession, **{int(k): v for k, v in zip(OFF, ar[i0 + OFF] - pl)}})
        recs.append(rec)
    D = B.merge(pd.DataFrame(recs), on="accession", how="inner")
    H = D[~D["early"]]
    print(f"With prices and placebo dates: {len(D):,} | headline sample (without releases 2-7 days early): {len(H):,}")

    print(f"\nRun-up, successes minus failures, net of {N_PL20} placebo dates (clipped 1%; 95% interval resampling firms)")
    head = row19("Headline: days -10..-2, without releases 2-7 days before the 8-K", H, "x10")
    row19("Same, days -5..-2", H, "x5")
    row19("Same, medians", H, "x10", med19)
    row19("Same, all readouts", D, "x10")
    if D["meeting"].notna().any():
        row19("Same, all readouts, releases that mention no medical meeting", D[D["meeting"] == False], "x10")
    row19("Same, releases dated on the 8-K day or the day before", D[D["lag"].between(0, 1)], "x10")
    row19("Same, without placebo adjustment (raw), headline sample", H, "run_10")
    row19("Same, beta-adjusted (beta to XBI from days -250..-30)", H, "xb10")
    row19("Same, market model (alpha and beta from days -250..-30), no placebo", H, "mm10")
    row19("Earlier window, days -30..-11, headline sample", H, "xe30")
    row19("Earlier window, days -20..-11, headline sample", H, "xe20")

    print("\nThe headline gap across five draws of 20 placebo dates")
    for s in range(5):
        row19(f"draw {s + 1} (seed {SEED + 100 + s})", H, f"x10_s{s}")

    print("\nInference that allows for market-wide shocks (headline sample)")
    z = H[["cik", "day0", "outcome", "x10"]].dropna().copy()
    z["y"], z["succ"] = clip(z["x10"]), (z["outcome"] == "success").astype(float)
    z["week"] = pd.to_datetime(z["day0"]).dt.to_period("W").astype(str)
    z["month"] = pd.to_datetime(z["day0"]).dt.to_period("M").astype(str)
    for lab, gc in [("firm", ["cik"]), ("firm and calendar week", ["cik", "week"]), ("firm and calendar month", ["cik", "month"])]:
        gg = np.column_stack([pd.factorize(z[c].astype(str))[0] for c in gc]) if len(gc) > 1 else pd.factorize(z["cik"].astype(str))[0]
        m = smf.ols("y ~ succ", z).fit(cov_type="cluster", cov_kwds={"groups": gg})
        b, se = float(m.params["succ"]), float(m.bse["succ"])
        print(f"  clustered by {lab:28s} {b:+.2%} [{b - 1.96 * se:+.2%}, {b + 1.96 * se:+.2%}] p = {float(m.pvalues['succ']):.3f}")
        R19.append({"check": f"Headline, clustered by {lab}", "gap": b, "lo": b - 1.96 * se, "hi": b + 1.96 * se, "p": float(m.pvalues["succ"])})
    months = z["month"].unique()
    idx = {mo: np.flatnonzero(z["month"].to_numpy() == mo) for mo in months}
    bs = np.asarray([gap19(z.iloc[np.concatenate([idx[mo] for mo in rng.choice(months, len(months))])]) for _ in range(N_BOOT)])
    bs = bs[np.isfinite(bs)]
    pb = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    print(f"  bootstrap resampling calendar months     {gap19(z):+.2%} [{np.percentile(bs, 2.5):+.2%}, {np.percentile(bs, 97.5):+.2%}] p = {pb:.3f}")
    R19.append({"check": "Headline, bootstrap resampling calendar months", "gap": gap19(z), "lo": np.percentile(bs, 2.5),
                "hi": np.percentile(bs, 97.5), "p": pb})

    print("\nSplits (headline sample)")
    ph = pd.to_numeric(col(H, "phase", np.nan), errors="coerce")
    sz = pd.to_numeric(col(H, "log_dollar_vol", np.nan), errors="coerce")
    vo = pd.to_numeric(col(H, "pre_vol", np.nan), errors="coerce")
    for lab, mk in [("phase 3", ph >= 3), ("phase 1 or 2", ph < 3), ("phase not given", ph.isna()),
                    ("dollar volume below median", sz < sz.median()), ("dollar volume above median", sz >= sz.median()),
                    ("volatility below median", vo < vo.median()), ("volatility above median", vo >= vo.median())]:
        row19(lab, H[mk.fillna(False).to_numpy()], "x10")
    for lab, hi_ in [("size", sz >= sz.median()), ("volatility", vo >= vo.median()), ("phase 3", ph >= 3)]:
        zz = H.assign(hi=hi_.astype(float), succ=(H["outcome"] == "success").astype(float))[["cik", "x10", "hi", "succ"]].dropna()
        zz["y"] = clip(zz["x10"])
        try:
            t_ = term(fit("y ~ succ * hi", zz), "succ:hi")
            print(f"  difference in the gap by {lab:10s} {t_['b']:+.2%} [{t_['lo']:+.2%}, {t_['hi']:+.2%}] p = {t_['p']:.2f}")
        except Exception as e:
            print(f"  interaction by {lab} did not fit: {e}")

    PM = None
    for f in sorted(glob.glob(os.path.join(FEAT_DIR, "nb5", "readout_text_features*.parquet")), key=os.path.getmtime)[::-1]:
        x = pd.read_parquet(f)
        if {"accession", "p_met"} <= set(x.columns):
            PM = x[["accession", "p_met"]].drop_duplicates("accession")
            break
    if PM is not None:
        HL = H.merge(PM, on="accession", how="left")
        agree = ((HL["outcome"] == "success") & (HL["p_met"] >= 0.5)) | ((HL["outcome"] == "failure") & (HL["p_met"] < 0.5))
        print("\nReadouts where the tone and the language model agree")
        row19("Headline, tone and language model agree", HL[agree], "x10")

    FL = load("nb6/readout_flags.parquet", "Notebook 6")
    if FL is not None and "any_flag" in FL:
        HG = H.merge(FL[["accession", "any_flag"]].drop_duplicates("accession"), on="accession", how="left")
        nct = col(HG, "nct_ids", "").fillna("").astype(str).str.contains(r"NCT\d{8}")
        print(f"\nRegistry records (headline sample): cite an NCT number {int(nct.sum()):,} ({nct.mean():.0%}) | linked to a registered "
              f"industry trial a year into the archive {int(HG['any_flag'].notna().sum()):,} ({HG['any_flag'].notna().mean():.0%}) | "
              f"with a warning in the year before {int((HG['any_flag'] == 1).sum()):,}")
        row19("Linked readouts", HG[HG["any_flag"].notna()], "x10")
        row19("Linked, no registry warning in the year before", HG[HG["any_flag"] == 0], "x10")
        row19("Linked, a registry warning in the year before", HG[HG["any_flag"] == 1], "x10")
        row19("Not linked to a registry record", HG[HG["any_flag"].isna()], "x10")
        zz = HG[HG["any_flag"].notna()].assign(succ=lambda d: (d["outcome"] == "success").astype(float))[["cik", "x10", "any_flag", "succ"]].dropna()
        zz["y"] = clip(zz["x10"])
        if ((zz["succ"] == 0) & (zz["any_flag"] == 1)).sum() >= 8:
            t_ = term(fit("y ~ succ * any_flag", zz), "succ:any_flag")
            print(f"  difference in the gap, warning minus none: {t_['b']:+.2%} [{t_['lo']:+.2%}, {t_['hi']:+.2%}] p = {t_['p']:.2f}")

    # ---- Holm across the four positive findings, with the new run-up p ----
    if head is not None:
        others = [x for x in RESULTS if x.get("family") == "positive findings" and not str(x.get("check", "")).startswith("Run-up")
                  and pd.notna(x.get("p"))]
        if len(others) == 3:
            ps = sorted([("run-up", head[3])] + [(x["check"], float(x["p"])) for x in others], key=lambda t: t[1])
            run, adj = 0.0, {}
            for i, (k, p_) in enumerate(ps):
                run = max(run, min(1.0, (4 - i) * p_))
                adj[k] = run
            print("\nHolm across the four positive findings: " + " | ".join(f"{k[:40]} {v:.3f}" for k, v in adj.items()))
        else:
            print(f"\nHolm: found {len(others)} of the other three positive findings in this session (run Steps 4, 6 and 8 for it)")

    # ---- 30-day figure with 20 placebo dates ----
    PA = pd.DataFrame(paths).merge(D[["accession", "cik", "outcome", "early"]], on="accession")
    PA = PA[~PA["early"]]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    for oc, c, cl, lab in [("success", NAVY, NAVY_L, "Successes"), ("failure", BRICK, BRICK_L, "Failures")]:
        x = PA[PA["outcome"] == oc].reset_index(drop=True)
        M = x[list(OFF)].apply(clip)
        path = M.mean(axis=0).fillna(0).cumsum().to_numpy() * 100
        firms = x["cik"].unique()
        fidx = {f_: np.flatnonzero(x["cik"].to_numpy() == f_) for f_ in firms}
        bsp = np.vstack([M.iloc[np.concatenate([fidx[f_] for f_ in rng.choice(firms, len(firms))])].mean(axis=0).fillna(0).cumsum().to_numpy() * 100
                         for _ in range(200)])
        ax.fill_between(OFF, np.percentile(bsp, 2.5, axis=0), np.percentile(bsp, 97.5, axis=0), color=cl, alpha=0.45, linewidth=0)
        ax.plot(OFF, path, color=c, label=f"{lab} (n = {len(x):,})")
        print(f"  {lab}: net of placebo dates, by day -21 {path[OFF == -21][0]:+.2f}%, by day -11 {path[OFF == -11][0]:+.2f}%, by day -2 {path[-1]:+.2f}%")
    ax.axhline(0, color=G2, linewidth=0.6)
    ax.axvline(-10, color=G2, linewidth=0.6, linestyle=(0, (3, 2)))
    ax.set_xlim(-30, -2)
    ax.set_xticks([-30, -25, -20, -15, -10, -5, -2])
    ax.set_xlabel("Trading days before the readout (day 0 is the first session that can react to the 8-K)")
    ax.set_ylabel("Cumulative abnormal return,\nnet of placebo dates (%)")
    legend_top19(ax, 2)
    save19(fig, "fig_runup30.png")

    # ---- Figure 2 with the tone-and-language-model row ----
    IP = os.path.join(FEAT_DIR, "nb6", "implied_pos.csv")
    if os.path.exists(IP) and PM is not None:
        s = pd.read_csv(IP)
        s = s[(s["window"] == "days -1..+1") & ~s["mixed_as_failure"].astype(str).str.lower().eq("true")].reset_index(drop=True)
        T4 = TA[TA["has_prices"] & TA["outcome"].isin(["success", "failure"]) & TA["car_1_1"].notna()].copy()
        lo_, hi_ = TA.loc[TA["has_prices"], "car_1_1"].quantile([WINSOR, 1 - WINSOR])
        T4["y"] = T4["car_1_1"].clip(lo_, hi_)
        T4 = T4.merge(PM, on="accession", how="left")
        A = T4[((T4["outcome"] == "success") & (T4["p_met"] >= 0.5)) | ((T4["outcome"] == "failure") & (T4["p_met"] < 0.5))]
        def imp(x):
            U, D_ = x.loc[x["outcome"] == "success", "y"].mean(), x.loc[x["outcome"] == "failure", "y"].mean()
            return -D_ / (U - D_) if U > D_ else np.nan
        g, lo, hi, _ = firm_boot(A[["cik", "outcome", "y"]].reset_index(drop=True), imp)
        rl = (A["outcome"] == "success").mean()
        print(f"\nTone and language model agree: implied {g:.0%} [{lo:.0%}, {hi:.0%}], realized {rl:.0%}, n = {len(A):,}")
        s.loc[s["group"] == "both labels agree", ["implied", "implied_lo", "implied_hi", "realized"]] = [g, lo, hi, rl]
        NAMES = {"all": "All topline readouts", "both labels agree": "Tone and language model agree", "clean events": "Clean events",
                 "era 2010-2015": "2010 to 2015", "era 2016-2020": "2016 to 2020", "era 2021-2025": "2021 to 2025", "2018 and later": "2018 and later",
                 "phase 3": "Phase 3", "phase 2": "Phase 2", "phase 1": "Phase 1", "phase unknown": "Phase not given",
                 "small firms (dollar volume)": "Small firms", "mid firms (dollar volume)": "Mid-size firms", "large firms (dollar volume)": "Large firms"}
        fig, ax = plt.subplots(figsize=(6.5, 0.24 * len(s) + 0.9))
        y = np.arange(len(s))[::-1]
        h1 = ax.errorbar(s["implied"] * 100, y, xerr=[(s["implied"] - s["implied_lo"]) * 100, (s["implied_hi"] - s["implied"]) * 100],
                         fmt="o", color=NAVY, ecolor=NAVY_L, elinewidth=2.2, markersize=5, capsize=0)
        h2, = ax.plot(s["realized"] * 100, y, "D", color=BRICK, markersize=4.5)
        ax.set_yticks(y)
        ax.set_yticklabels([NAMES.get(g_, g_) for g_ in s["group"]])
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Probability of success (%)")
        legend_top19(ax, 2, [h1, h2], ["Implied by prices, with 95% interval", "Realized share of successes"])
        save19(fig, "fig_implied.png")

    pd.DataFrame(R19).to_csv(os.path.join(NB10_DIR, "runup_20_placebos.csv"), index=False)
    zp = os.path.join(NB10_DIR, "runup20_figures.zip")
    with zipfile.ZipFile(zp, "w") as zf:
        for p in R19_FIGS:
            zf.write(p, os.path.basename(p))
    print(f"\nSaved runup_20_placebos.csv and {os.path.basename(zp)} in {NB10_DIR}")
    try:
        from google.colab import files
        files.download(zp)
    except Exception:
        pass


# %% [markdown]
# ## Step 19c: The 30-day gap, the run-up relative to the announcement, and Holm
# Prints the six-week gap and the ratio of the run-up gap to the announcement gap. The share reported in the paper, the run-up gap divided by the sum of the run-up and announcement gaps (Appendix B), comes from Step 21 (d). Run right after Step 19.


# %% GAP30
# ===== Step 19c: the 30-day gap, the share of uncertainty resolved, and Holm (run right after Step 19) =====
H30 = H.assign(x30=H["xe30"] + H["x10"])
D30 = D.assign(x30=D["xe30"] + D["x10"])
print("Run-up, successes minus failures, net of 20 placebo dates (clipped 1%; 95% interval resampling firms)")
r30 = row19("Days -30..-2, headline sample", H30, "x30")
row19("Days -30..-2, medians", H30, "x30", med19)
row19("Days -30..-2, all readouts", D30, "x30")
if PM is not None:
    A30 = H30.merge(PM, on="accession", how="left")
    ag = ((A30["outcome"] == "success") & (A30["p_met"] >= 0.5)) | ((A30["outcome"] == "failure") & (A30["p_met"] < 0.5))
    row19("Days -30..-2, tone and language model agree", A30[ag], "x30")

# ---- run-up gap relative to the announcement gap (the paper's share is in Step 21 (d)) ----
lo_, hi_ = TA.loc[TA["has_prices"], "car_1_1"].quantile([WINSOR, 1 - WINSOR])
SH = H30[["cik", "outcome", "x10", "x30", "car_1_1"]].dropna().copy()
SH["ann"] = SH["car_1_1"].clip(lo_, hi_)
SH["y10"], SH["y30"] = clip(SH["x10"]), clip(SH["x30"])
def share_of(v):
    def f(x):
        s_, f_ = x["outcome"] == "success", x["outcome"] == "failure"
        return (x.loc[s_, v].mean() - x.loc[f_, v].mean()) / (x.loc[s_, "ann"].mean() - x.loc[f_, "ann"].mean())
    return f
gap_ann = SH.loc[SH["outcome"] == "success", "ann"].mean() - SH.loc[SH["outcome"] == "failure", "ann"].mean()
print(f"\nAnnouncement gap, days -1..+1 (clipped): {gap_ann:+.1%} | n = {len(SH):,}")
for v, lab in [("y10", "days -10..-2"), ("y30", "days -30..-2")]:
    g, lo, hi, p = firm_boot(SH, share_of(v))
    print(f"Run-up gap / announcement gap, {lab}: {g:.1%} [{lo:.1%}, {hi:.1%}], p = {p:.3f}")

# ---- Holm across the four positive findings, from the saved summary ----
SMp = os.path.join(NB10_DIR, "summary.csv")
if os.path.exists(SMp):
    SMs = pd.read_csv(SMp)
    oth = SMs[(SMs.get("family") == "positive findings") & ~SMs["check"].astype(str).str.startswith("Run-up")].dropna(subset=["p"])
    print("\nOther positive findings: " + " | ".join(f"{c[:45]} p = {p:.4f}" for c, p in zip(oth["check"], oth["p"])))
    def holm_first(p0, others):
        ps = sorted([("x", p0)] + [(c, float(p)) for c, p in others], key=lambda t: t[1])
        run = 0.0
        for i, (k, p_) in enumerate(ps):
            run = max(run, min(1.0, (len(ps) - i) * p_))
            if k == "x":
                return run
    for lab, rr in [("days -10..-2", head), ("days -30..-2", r30)]:
        if rr is not None and len(oth) == 3:
            print(f"Holm-adjusted p for the run-up, {lab}: {holm_first(rr[3], zip(oth['check'], oth['p'])):.3f}")
else:
    print("\nNo summary.csv in NB10_DIR yet: run Steps 1 to 9 once for the Holm correction")


# %% [markdown]
# ## Step 20: Two standard checks on the run-up, fixed before seeing their results
# (1) The outcome read from the sign of the market's own reaction, so tone-label errors drop out and mixed and unclear readouts count, with a falsification test on placebo dates. (2) Run-ups scaled by each stock's volatility. Run right after Step 19.


# %% CHECKS20
# ===== Step 20: Two standard checks on the run-up, fixed before seeing their results =====
# (1) the outcome read from the market's own reaction, so tone-label errors drop out and mixed and unclear readouts count
# (2) run-ups scaled by each stock's own volatility (standardized abnormal returns), so a few very volatile stocks carry less weight
# Run right after Step 19 (it reuses D, H, TXr, pr_date and row19). Report both, whatever they show.
if "D" not in globals() or "H" not in globals():
    print("Run Step 19 first")
else:
    have = set(D["accession"])
    M = TA[TA["has_prices"] & TA["outcome"].isin(["mixed", "unclear"]) & ~TA["accession"].isin(have)].copy()
    if TXr is not None:
        M = M.merge(TXr, on="accession", how="left")
        acc_day = as_day(M["acceptance_et"]) if "acceptance_et" in M else M["event_date"]
        M["lag"] = (acc_day - as_day(M["text_raw"].map(pr_date))).dt.days
    else:
        M["lag"] = np.nan
    M["early"] = M["lag"].between(2, 7)
    g2 = np.random.default_rng(SEED + 200)
    rec2 = []
    for r in M.itertuples():
        a = px_arrays(r.ticker)
        if a is None:
            continue
        ar, live = a
        i0 = int(CAL.searchsorted(r.day0))
        if i0 < 260 or i0 >= len(CAL) - 2:
            continue
        cand = np.arange(max(i0 - 500, 30), min(i0 + 500, len(CAL) - 30))
        ev = rpos.get(r.cik, np.array([]))
        if len(ev):
            cand = cand[~(np.abs(ev[None, :] - cand[:, None]) <= 30).any(axis=1)]
        cand = cand[live[cand]]
        if len(cand) < 2:
            continue
        pk = g2.choice(cand, size=min(N_PL20, len(cand)), replace=False)
        rec2.append({"accession": r.accession,
                     "x10": wsum(ar, i0, -10, -2) - np.nanmean([wsum(ar, int(j), -10, -2) for j in pk]),
                     "xe30": wsum(ar, i0, -30, -11) - np.nanmean([wsum(ar, int(j), -30, -11) for j in pk])})
    M = M.merge(pd.DataFrame(rec2), on="accession", how="inner")
    cols = ["accession", "cik", "ticker", "day0", "outcome", "x10", "xe30", "car_1_1", "pre_vol", "early"]
    ALL = pd.concat([D[cols], M[cols]], ignore_index=True)
    ALL = ALL[~ALL["early"].astype(bool) & ALL["car_1_1"].notna()].copy()
    ALL["x30"] = ALL["xe30"] + ALL["x10"]
    ALL["label"] = ALL["outcome"]
    ALL["outcome"] = np.where(ALL["car_1_1"] > 0, "success", "failure")
    print(f"Topline readouts with prices and placebo dates, without releases 2-7 days early: {len(ALL):,}")
    print("Reaction over days -1..+1 by tone label (share negative): " + " | ".join(
        f"{lab} {(ALL.loc[ALL['label'] == lab, 'outcome'] == 'failure').mean():.0%} of {(ALL['label'] == lab).sum():,}"
        for lab in ["success", "failure", "mixed", "unclear"]))

    print("\n(1) Outcome read from the sign of the reaction over days -1..+1 (all topline readouts)")
    row19("Days -10..-2, outcome from the reaction", ALL, "x10")
    row19("Days -30..-2, outcome from the reaction", ALL, "x30")
    agree = ALL["label"] == ALL["outcome"]
    row19("Days -10..-2, tone label and reaction agree", ALL[agree], "x10")
    row19("Days -30..-2, tone label and reaction agree", ALL[agree], "x30")
    # falsification: the same sorting on placebo dates, where no readout happens, should give no gap
    g3 = np.random.default_rng(SEED + 300)
    pr_rows = []
    for r in ALL.itertuples():
        a = px_arrays(r.ticker)
        if a is None:
            continue
        ar, live = a
        i0 = int(CAL.searchsorted(r.day0))
        cand = np.arange(max(i0 - 500, 40), min(i0 + 500, len(CAL) - 30))
        ev = rpos.get(r.cik, np.array([]))
        if len(ev):
            cand = cand[~(np.abs(ev[None, :] - cand[:, None]) <= 30).any(axis=1)]
        cand = cand[live[cand]]
        for j in g3.choice(cand, size=min(5, len(cand)), replace=False) if len(cand) >= 2 else []:
            post = wsum(ar, int(j), -1, 1)
            if np.isfinite(post) and post != 0:
                pr_rows.append({"cik": r.cik, "outcome": "success" if post > 0 else "failure",
                                "p10": wsum(ar, int(j), -10, -2), "p30": wsum(ar, int(j), -30, -2)})
    PL = pd.DataFrame(pr_rows)
    print("  Falsification on placebo dates, sorted the same way by their own days -1..+1 return (should be near zero):")
    row19("    placebo dates, days -10..-2", PL, "p10")
    row19("    placebo dates, days -30..-2", PL, "p30")

    print("\n(2) Run-ups scaled by each stock's daily volatility (days -250..-30); gap in standard deviations")
    def zrow(label, d, v, k):
        z = d[["cik", "outcome", v, "pre_vol"]].dropna()
        z = z[z["pre_vol"] > 0]
        z = z.assign(y=clip(z[v] / (z["pre_vol"] * np.sqrt(k))))[["cik", "outcome", "y"]]
        ns, nf = int((z["outcome"] == "success").sum()), int((z["outcome"] == "failure").sum())
        g, lo, hi, p = firm_boot(z, gap19)
        typ = float(d["pre_vol"].median() * np.sqrt(k))
        print(f"  {label:58s} {g:+.3f} SD [{lo:+.3f}, {hi:+.3f}] p = {p:.3f} | about {g * typ:+.2%} at the median stock "
              f"| successes {ns}, failures {nf}")
        R19.append({"check": label + " (standardized, SD)", "gap": g, "lo": lo, "hi": hi, "p": p, "successes": ns, "failures": nf})
    Hs = H.assign(x30=H["xe30"] + H["x10"])
    zrow("Days -10..-2, tone labels (headline sample)", Hs, "x10", 9)
    zrow("Days -30..-2, tone labels (headline sample)", Hs, "x30", 29)
    zrow("Days -10..-2, outcome from the reaction", ALL, "x10", 9)
    zrow("Days -30..-2, outcome from the reaction", ALL, "x30", 29)
    pd.DataFrame(R19).to_csv(os.path.join(NB10_DIR, "runup_20_placebos.csv"), index=False)


# %% [markdown]
# ## Step 21: The six-week run-up under every robustness check, a selection-adjusted test, the corrected share and sample counts
# Run right after Step 19 in the same session. It redraws Step 19's placebo dates with the same seeds, applies the two-week checks to days -30..-2, tests the six-week gap against the best of all windows from day a to day -2 (a = -30..-3) under two nulls, computes the run-up gap as a share of the run-up plus announcement gap, and prints why readouts drop out of each sample.


# %% ROBUST30
# ===== Step 21: the six-week run-up under every robustness check, a selection-adjusted test, the corrected share, sample counts =====
# Run right after Step 19 in the same session. It redraws Step 19's placebo dates with the same seeds, in the same order.
if any(k not in globals() for k in ["B", "D", "H", "rpos", "bspan19", "row19", "gap19"]):
    print("Run Step 19 first, in this session")
else:
    R21 = []
    B_PERM, B_PSEUDO = 2000, 1000
    A_ = np.arange(-30, -2)                      # window starts for the selection-adjusted test; every window ends at day -2

    def w30(x_, i):                              # days -30..-2, built from the same two pieces as Step 19c
        return wsum(x_, i, -30, -11) + wsum(x_, i, -10, -2)

    def wins(x_, i):                             # sums over days a..-2 for every start a in A_, with wsum's rule for missing days
        seg = x_[i - 30:i - 1]
        f = np.isfinite(seg)
        s = np.cumsum(np.where(f, seg, 0.0)[::-1])[::-1][:len(A_)]
        k = np.cumsum(f[::-1])[::-1][:len(A_)]
        out = s.astype(float)
        out[k < np.maximum(1, (-1 - A_) // 2)] = np.nan
        return out

    def row21(label, d, v, q=WINSOR, trim=None, stat=gap19):
        z = d[["cik", "outcome", v]].dropna().rename(columns={v: "y"})
        if trim:
            lo_q = z.groupby("outcome")["y"].transform(lambda s: s.quantile(trim))
            hi_q = z.groupby("outcome")["y"].transform(lambda s: s.quantile(1 - trim))
            z = z[(z["y"] >= lo_q) & (z["y"] <= hi_q)]
        else:
            z["y"] = clip(z["y"], q)
        ns, nf = int((z["outcome"] == "success").sum()), int((z["outcome"] == "failure").sum())
        g, lo, hi, p = firm_boot(z, stat)
        print(f"  {label:66s} {g:+.2%} [{lo:+.2%}, {hi:+.2%}] p = {p:.3f} | successes {ns}, failures {nf}")
        R21.append({"check": label, "gap": g, "lo": lo, "hi": hi, "p": p, "successes": ns, "failures": nf})

    # ---- redraw Step 19's placebo dates and compute the six-week variants ----
    g21 = [np.random.default_rng(SEED + 100 + s) for s in range(5)]
    why, recs, OBW, PLW = [], [], [], []
    for r in B.itertuples():
        a = px_arrays(r.ticker)
        if a is None:
            why.append((r.accession, "no usable price series"))
            continue
        ar, live = a
        i0 = int(CAL.searchsorted(r.day0))
        if i0 < 260:
            why.append((r.accession, "in the first 260 trading days of the price calendar"))
            continue
        if i0 >= len(CAL) - 2:
            why.append((r.accession, "in the last two days of the price data"))
            continue
        cand = np.arange(max(i0 - 500, 30), min(i0 + 500, len(CAL) - 30))
        ev = rpos.get(r.cik, np.array([]))
        if len(ev):
            cand = cand[~(np.abs(ev[None, :] - cand[:, None]) <= 30).any(axis=1)]
        cand = cand[live[cand]]
        if len(cand) < 2:
            why.append((r.accession, "fewer than two eligible placebo dates"))
            continue
        picks = [gs.choice(cand, size=min(N_PL20, len(cand)), replace=False) for gs in g21]
        rm = bspan19(r.ticker, live)
        est = slice(max(i0 - 280, 0), i0 - 30)                      # days -280..-31, before the six-week window
        y_, x_ = ar[est] + rm[est], rm[est]
        ok = np.isfinite(y_) & np.isfinite(x_)
        rec = {"accession": r.accession, "raw30": w30(ar, i0)}
        for s, pk in enumerate(picks):
            rec[f"x30_s{s}"] = ((wsum(ar, i0, -30, -11) - np.nanmean([wsum(ar, int(j), -30, -11) for j in pk]))
                                + (wsum(ar, i0, -10, -2) - np.nanmean([wsum(ar, int(j), -10, -2) for j in pk])))
        if ok.sum() >= 100:
            alpha, beta = np.linalg.lstsq(np.column_stack([np.ones(ok.sum()), x_[ok]]), y_[ok], rcond=None)[0]
            badj = ar + (1 - beta) * rm
            rec["xb30"] = w30(badj, i0) - np.nanmean([w30(badj, int(j)) for j in picks[0]])
            rec["mm30"] = w30(badj - alpha, i0)
        recs.append(rec)
        OBW.append((r.accession, wins(ar, i0)))
        pw = np.full((N_PL20, len(A_)), np.nan)
        pw[:len(picks[0])] = np.vstack([wins(ar, int(j)) for j in picks[0]])
        PLW.append(pw)
    D21 = B.merge(pd.DataFrame(recs), on="accession", how="inner")
    H21 = D21[~D21["early"]].copy()

    # ---- (a) sample counts ----
    def so(d):
        return f"successes {int((d['outcome'] == 'success').sum())}, failures {int((d['outcome'] == 'failure').sum())}"
    print("(a) Sample counts")
    print(f"  Positive and negative topline readouts with prices: {len(B):,} ({so(B)}) | release 2-7 days before the 8-K: {int(B['early'].sum())}")
    WY = pd.DataFrame(why, columns=["accession", "reason"]).merge(B[["accession", "outcome", "early"]], on="accession")
    for rsn, g_ in WY.groupby("reason"):
        print(f"  dropped, {rsn}: {len(g_)} ({so(g_)}; early releases {int(g_['early'].sum())})")
    print(f"  With prices and placebo dates: {len(D21):,} ({so(D21)}) | early releases among them: {int(D21['early'].sum())} ({so(D21[D21['early']])})")
    print(f"  Headline sample without them: {len(H21):,} ({so(H21)}) | Step 19 had {len(D):,} and {len(H):,}")
    RI = load("nb9/readout_insiders.parquet", "Notebook 9")
    if RI is not None and "log_dollar_vol" in RI:
        x = RI[RI["outcome"].isin(["success", "failure"])]
        m_ = x["log_dollar_vol"].isna()
        print(f"  Insider tests (Notebook 9): {len(x):,} ({so(x)}) | without the size control (log dollar volume): {int(m_.sum())} ({so(x[m_])})")
        for p_, lab in [(0, "before"), (1, "from")]:
            print(f"    {lab} {POST_START}, with the size control: {so(x[~m_ & (x['post'] == p_)])}")

    # ---- (b) the six-week run-up under the robustness checks ----
    print(f"\n(b) Run-up, days -30..-2, successes minus failures (95% interval resampling firms; headline sample unless noted)")
    row21("Headline (Step 19c, draw 1 of 20 placebo dates, clipped 1%)", H21, "x30_s0")
    row21("Without placebo adjustment (raw)", H21, "raw30")
    row21("Beta-adjusted (beta to XBI from days -280..-31), net of placebo dates", H21, "xb30")
    row21("Market model (alpha and beta from days -280..-31), no placebo", H21, "mm30")
    for s in range(5):
        row21(f"Placebo draw {s + 1} of 20 dates (seed {SEED + 100 + s})", H21, f"x30_s{s}")
    row21("No clipping", H21, "x30_s0", q=0)
    row21("Clipped 2.5%", H21, "x30_s0", q=0.025)
    row21("Clipped 5%", H21, "x30_s0", q=0.05)
    row21("Trimmed 5% within each outcome", H21, "x30_s0", trim=0.05)
    if D21["meeting"].notna().any():
        row21("All readouts, releases that mention no medical meeting", D21[D21["meeting"] == False], "x30_s0")
    row21("Releases dated on the 8-K day or the day before", D21[D21["lag"].between(0, 1)], "x30_s0")
    z = H21[["cik", "day0", "outcome", "x30_s0"]].dropna().copy()
    z["y"], z["succ"] = clip(z["x30_s0"]), (z["outcome"] == "success").astype(float)
    z["week"] = pd.to_datetime(z["day0"]).dt.to_period("W").astype(str)
    z["month"] = pd.to_datetime(z["day0"]).dt.to_period("M").astype(str)
    for lab, gc in [("firm", ["cik"]), ("firm and calendar week", ["cik", "week"]), ("firm and calendar month", ["cik", "month"])]:
        gg = np.column_stack([pd.factorize(z[c].astype(str))[0] for c in gc]) if len(gc) > 1 else pd.factorize(z["cik"].astype(str))[0]
        m = smf.ols("y ~ succ", z).fit(cov_type="cluster", cov_kwds={"groups": gg})
        b, se = float(m.params["succ"]), float(m.bse["succ"])
        print(f"  {'Clustered by ' + lab:66s} {b:+.2%} [{b - 1.96 * se:+.2%}, {b + 1.96 * se:+.2%}] p = {float(m.pvalues['succ']):.3f}")
        R21.append({"check": f"Clustered by {lab}", "gap": b, "lo": b - 1.96 * se, "hi": b + 1.96 * se, "p": float(m.pvalues["succ"])})
    months = z["month"].unique()
    idx = {mo: np.flatnonzero(z["month"].to_numpy() == mo) for mo in months}
    bs = np.asarray([gap19(z.iloc[np.concatenate([idx[mo] for mo in rng.choice(months, len(months))])]) for _ in range(N_BOOT)])
    bs = bs[np.isfinite(bs)]
    pb = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    print(f"  {'Bootstrap resampling calendar months':66s} {gap19(z):+.2%} [{np.percentile(bs, 2.5):+.2%}, {np.percentile(bs, 97.5):+.2%}] p = {pb:.3f}")
    R21.append({"check": "Bootstrap resampling calendar months", "gap": gap19(z), "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5), "p": pb})

    # ---- (c) selection-adjusted test: how unusual is the six-week gap among all windows from day a to day -2, a = -30..-3? ----
    BU = B.drop_duplicates("accession").set_index("accession")
    keep = ~BU.loc[[k for k, _ in OBW], "early"].to_numpy().astype(bool)
    acc_ = np.array([k for k, _ in OBW])[keep]
    OB = np.vstack([w for _, w in OBW])[keep]
    PLA = np.stack(PLW)[keep]                                   # readouts x 20 placebo dates x windows
    X_ = OB - np.nanmean(PLA, axis=1)                           # each window net of the readout's 20 placebo dates (draw 1)
    meta = BU.loc[acc_]
    dsu = (meta["outcome"].to_numpy() == "success").astype(float)
    cid = pd.factorize(meta["cik"].astype(str))[0]
    Cm = np.zeros((len(cid), cid.max() + 1))
    Cm[np.arange(len(cid)), cid] = 1.0

    def clipc(Y):
        lo, hi = np.nanquantile(Y, WINSOR, axis=0), np.nanquantile(Y, 1 - WINSOR, axis=0)
        return np.where(np.isfinite(Y), np.clip(Y, lo, hi), np.nan)

    def tcols(Yc, d):                                           # success-minus-failure gap per column, t clustered by firm (as statsmodels)
        W = np.isfinite(Yc).astype(float)
        Y0, Ds = np.nan_to_num(Yc), d[:, None] * W
        Df = W - Ds
        n1, n0 = Ds.sum(0), Df.sum(0)
        m1, m0 = (Ds * Y0).sum(0) / n1, (Df * Y0).sum(0) / n0
        E = W * (Y0 - np.where(d[:, None] == 1, m1, m0))
        S = Cm.T @ (Ds * E) / n1 - Cm.T @ (Df * E) / n0
        G = (Cm.T @ W > 0).sum(0)
        n = n1 + n0
        se = np.sqrt(G / (G - 1) * (n - 1) / (n - 2) * (S ** 2).sum(0))
        return (m1 - m0) / se, m1 - m0

    t_obs, g_obs = tcols(clipc(X_), dsu)
    i30 = int(np.flatnonzero(A_ == -30)[0])
    print(f"\n(c) Windows from day a to day -2, headline sample (n = {len(dsu):,}), gap and t clustered by firm:")
    print("  " + " | ".join(f"a = {a:+d}: {g_obs[k]:+.1%} (t {t_obs[k]:.2f})" for k, a in enumerate(A_) if a in (-30, -25, -20, -15, -11, -10, -5, -3)))
    print(f"  largest t at a = {A_[np.nanargmax(t_obs)]:+d} (t {np.nanmax(t_obs):.2f}); six-week t = {t_obs[i30]:.2f}")
    rp = np.random.default_rng(SEED + 2100)
    Xc = clipc(X_)
    nul_perm = np.vstack([tcols(Xc, rp.permutation(dsu))[0] for _ in range(B_PERM)])
    npick = np.isfinite(PLA).any(axis=2).sum(axis=1)
    tot, cnt = np.nansum(PLA, axis=1), np.isfinite(PLA).sum(axis=1)
    nul_ps = []
    for _ in range(B_PSEUDO):
        k = np.floor(rp.random(len(npick)) * npick).astype(int)
        v = PLA[np.arange(len(k)), k]
        Y = v - (tot - np.nan_to_num(v)) / (cnt - np.isfinite(v))   # a placebo date as the event, net of the other placebo dates
        nul_ps.append(tcols(clipc(Y), dsu)[0])
    nul_ps = np.vstack(nul_ps)
    for lab, NL in [(f"label permutation ({B_PERM:,} draws)", nul_perm), (f"placebo dates as pseudo-events ({B_PSEUDO:,} draws)", nul_ps)]:
        mx, mxa = np.nanmax(NL, axis=1), np.nanmax(np.abs(NL), axis=1)
        p_fix = (NL[:, i30] >= t_obs[i30]).mean()
        p_sel = (mx >= t_obs[i30]).mean()
        p_sel2 = (mxa >= abs(t_obs[i30])).mean()
        print(f"  Null from {lab}: six-week window fixed, one-sided p = {p_fix:.3f} | best of the {len(A_)} windows, one-sided p = {p_sel:.3f}, "
              f"two-sided p = {p_sel2:.3f} | 95th percentile of the largest t = {np.percentile(mx, 95):.2f}")
        R21.append({"check": f"Selection-adjusted, {lab}", "gap": g_obs[i30], "p": p_sel, "p_two_sided": p_sel2, "p_fixed": p_fix})

    # ---- (d) the run-up gap as a share of the run-up plus the announcement gap ----
    lo_, hi_ = TA.loc[TA["has_prices"], "car_1_1"].quantile([WINSOR, 1 - WINSOR])
    SH = H.assign(x30=H["xe30"] + H["x10"])[["cik", "outcome", "x10", "x30", "car_1_1"]].dropna().copy()
    SH["ann"] = SH["car_1_1"].clip(lo_, hi_)
    SH["y10"], SH["y30"] = clip(SH["x10"]), clip(SH["x30"])
    def share2(v):
        def f(x):
            s_, f_ = x["outcome"] == "success", x["outcome"] == "failure"
            gp = x.loc[s_, v].mean() - x.loc[f_, v].mean()
            return gp / (gp + x.loc[s_, "ann"].mean() - x.loc[f_, "ann"].mean())
        return f
    print(f"\n(d) Run-up gap / (run-up gap + announcement gap over days -1..+1, clipped), n = {len(SH):,}")
    for v, lab in [("y30", "days -30..-2"), ("y10", "days -10..-2")]:
        g, lo, hi, p = firm_boot(SH, share2(v))
        print(f"  {lab}: {g:.1%} [{lo:.1%}, {hi:.1%}], p = {p:.3f}")
        R21.append({"check": f"Share of the separation before the filing, {lab}", "gap": g, "lo": lo, "hi": hi, "p": p})
    pd.DataFrame(R21).to_csv(os.path.join(NB10_DIR, "runup30_robust.csv"), index=False)
    print("\nSaved runup30_robust.csv")


# %% [markdown]
# ## Summary, including Steps 10-13


# %% SUMMARY2
SM = pd.DataFrame(RESULTS)
if len(SM):
    SM["Holm p (within family)"] = np.nan
    for fam, g in SM[SM["family"] == "positive findings"].dropna(subset=["p"]).groupby("family"):
        order = g["p"].sort_values().index
        m_ = len(order)
        adj, run = [], 0.0
        for i, ix in enumerate(order):
            run = max(run, min(1.0, (m_ - i) * g.loc[ix, "p"]))
            adj.append((ix, run))
        for ix, v in adj:
            SM.loc[ix, "Holm p (within family)"] = v
    SM.to_csv(os.path.join(NB10_DIR, "summary.csv"), index=False)
    with pd.option_context("display.max_colwidth", 70, "display.width", 300):
        print(SM.assign(p=SM["p"].round(3), **{"Holm p (within family)": SM["Holm p (within family)"].round(3)})[
            ["check", "estimate", "interval", "p", "Holm p (within family)", "reading"]].to_string(index=False))
print("\nSaved in", NB10_DIR, ":", sorted(os.listdir(NB10_DIR)))

"""Notebook 9: Who knows first?

Pipeline stage of the who-knows-first replication code. Cells are separated by '# %%' markers, so the file
runs top to bottom as a script and opens as a notebook in Jupyter, VS Code or (after tools/make_notebooks.py) Colab.
See README.md for inputs, outputs and the run order.
"""


# %% [markdown]
# # Notebook 9: Who knows first?
#
# **Part A. Insiders before trial readouts.** Officers and directors file Form 4 within two business days of trading their company's stock. The SEC's free Insider Transactions Data Sets have every Form 3, 4 and 5 since 2006. For each topline readout we count open-market sales (code S) and purchases (code P) by officers and directors in the 60 days before the readout (also 30 and 120), and compare readouts that turned out to be failures with successes. More selling before failures than before successes means insiders traded on what they knew.
#
# **The reform.** Many insider sales go through Rule 10b5-1 plans, set up in advance as a defense against insider-trading claims. Before 2023 an insider could adopt a plan and start selling almost at once. The amendments effective February 27, 2023 require officers and directors to wait at least 90 days after adopting or changing a plan. If insiders used plans to sell ahead of bad readouts, the failure-minus-success gap in selling should shrink after the reform (readouts from `POST_START` on). Plan sales are flagged from the Form 4's 10b5-1 checkbox (filed from April 2023; `AFF10B5ONE`) or a footnote mentioning 10b5-1 (common before 2023).
#
# Comparisons control for the firm's usual selling (the log number of open-market sales by officers and directors in the two years before, leaving out the last 120 days), size and year, with errors clustered by firm. Firms that file no Form 4s at all (foreign private issuers are exempt) are left out.
#
# **Part B. The Friday test.** DellaVigna and Pollet (2009) show that news released on Friday gets a smaller immediate reaction and more drift afterward, because investors pay less attention, and that firms put worse news on Fridays. Here: topline readouts and stop 8-Ks filed on a Friday (or the weekend) vs Monday to Thursday, for bad news (negative or mixed) and good news. Immediate reaction: days -1..+1; delayed: days +2..+20; and abnormal volume.
#
# **Needs:** Notebook 4 v2 (`readout_events.parquet`, `stop_disclosure_events.parquet`). Set `USER_AGENT`. Downloads one zip per quarter from the SEC (about 15 MB each, 2008 onward, about 75 files), keeps only the sample's firms, and saves the filtered rows per quarter to Drive, so a rerun downloads nothing. About 15 to 30 minutes the first time. No GPU.
#
# **Output** (in `MyDrive/biopharma_tiv/features/nb9/`): `insider_trades.parquet`, `readout_insiders.parquet`, figures `fig16_insiders.png`, `fig17_friday.png`.


# %% CONFIG
# ==== Settings: edit USER_AGENT ====
import os
USER_AGENT = os.environ.get("SEC_USER_AGENT", "Your Name your.email@example.com")   # same as Notebooks 1, 3b and 8
PROJECT_DIR = os.environ.get("WKF_PROJECT_DIR", "/content/drive/MyDrive/biopharma_tiv")
MAX_RPS = 5
FIRST_Q = "2008q1"                 # baseline selling needs two years before the first readouts
LAST_Q = None                      # None = up to the latest quarter the SEC has posted
ZIP_DIR = "/content/insider_zips"  # raw zips (Colab disk, not Drive); deleted after filtering unless KEEP_ZIPS
KEEP_ZIPS = False
WINDOWS = [(-30, -1), (-60, -1), (-120, -1)]       # calendar days before the readout
AFTER = (1, 60)                                    # after the readout (contrast)
MAIN = (-60, -1)
BASE = (-850, -121)                                # the firm's usual selling: two years, leaving out the last 120 days
REFORM_DATE = "2023-02-27"
POST_START = "2023-06-01"                          # readouts after the 90-day cooling-off could bind


# %% [markdown]
# ## Step 0: Setup


# %% SETUP
import os, re, io, glob, time, random, zipfile, threading, warnings
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import statsmodels.formula.api as smf

if PROJECT_DIR.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
assert "@" in USER_AGENT and "Your Name" not in USER_AGENT, "Set USER_AGENT to your real name and email first (first code cell)."
FEAT_DIR = os.path.join(PROJECT_DIR, "features")
NB9_DIR = os.path.join(FEAT_DIR, "nb9")
QDIR = os.path.join(NB9_DIR, "insider_quarters")
FIG_DIR = os.path.join(FEAT_DIR, "figures")
for d in (NB9_DIR, QDIR, FIG_DIR, ZIP_DIR):
    os.makedirs(d, exist_ok=True)

class FatalSECBlock(Exception):
    pass

class SECClient:
    """Polite SEC client: declared User-Agent, rate limit, retries with backoff (same as Notebooks 1, 3b, 8)."""
    def __init__(self, user_agent, max_rps=5):
        self.min_gap, self._next, self._lock = 1.0 / max_rps, 0.0, threading.Lock()
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.n_requests = 0

    def get(self, url, timeout=300, max_tries=6):
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

def sec_date(s):
    """SEC data sets write dates as DD-MON-YYYY; fall back to a general parse."""
    s = pd.Series(s, dtype=object)
    d = pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")
    bad = d.isna() & s.notna()
    if bad.any():
        d[bad] = pd.to_datetime(s[bad], errors="coerce", format="mixed")
    return d.dt.normalize().astype("datetime64[ns]")

def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)

def cluster_groups(firm, date=None):
    f = pd.factorize(pd.Series(firm).astype(str))[0]
    if len(set(f)) < 10:
        return None
    if date is None:
        return f
    d = pd.factorize(pd.Series(date).astype(str))[0]
    return np.column_stack([f, d]) if len(set(d)) >= 10 else f

def mean_t(df, c, date_col="day0"):
    d = df[[c, "cik", date_col]].dropna() if c in df else pd.DataFrame()
    n = len(d)
    if n <= 2:
        return f"n = {n}"
    y = d[c].astype(float).to_numpy()
    t = np.nan
    if y.std() > 0:
        g = cluster_groups(d["cik"], d[date_col])
        kw = {"cov_type": "cluster", "cov_kwds": {"groups": g}} if g is not None else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = float(np.asarray(sm.OLS(y, np.ones(n)).fit(**kw).tvalues)[0])
    return f"{y.mean():+.2%} (t = {t:.1f}), n = {n:,}"

def fit(formula, df, kind="ols"):
    z = df.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = smf.logit if kind == "logit" else smf.ols
        m = f(formula, z).fit(**({"disp": 0} if kind == "logit" else {}), cov_type="cluster",
                              cov_kwds={"groups": pd.factorize(z["cik"])[0]})
    return m

def show(m, keep, title, odds=False):
    k = [i for i in m.params.index if any(i == x or i.startswith(x + "[") for x in keep) or i in keep]
    t = pd.DataFrame({"coef": m.params[k], "se": m.bse[k], "p": m.pvalues[k]})
    if odds:
        t.insert(0, "odds ratio", np.exp(m.params[k]))
    print(f"-- {title} (n = {int(m.nobs):,})\n" + t.round(4).to_string())
print("Saving to:", NB9_DIR)


# %% [markdown]
# ## Step 1: Readouts and the firms to keep
# Topline readouts with prices from Notebook 4 (success = positive, failure = negative, mixed kept separately). The insider data are filtered to these firms (and the stop firms, for later use).


# %% LOAD
ro = pd.read_parquet(os.path.join(FEAT_DIR, "readout_events.parquet"))
ro["event_date"] = as_day(ro["event_date"])
ro["cik"] = pd.to_numeric(ro["cik"], errors="coerce")
ro = ro[as_bool(col(ro, "has_prices", True)) & ro["cik"].notna()].copy()
ro["cik"] = ro["cik"].astype(int)
ro["outcome"] = ro["readout_tone"].map({"positive": "success", "negative": "failure", "mixed": "mixed"}).fillna("unclear")
ro["year"] = ro["event_date"].dt.year
dse_path = os.path.join(FEAT_DIR, "stop_disclosure_events.parquet")
dse = pd.read_parquet(dse_path) if os.path.exists(dse_path) else pd.DataFrame(columns=["cik"])
dse["cik"] = pd.to_numeric(dse["cik"], errors="coerce")
KEEP_CIKS = set(ro["cik"]) | set(dse["cik"].dropna().astype(int))
top = ro[ro["readout_kind"] == "topline_or_endpoint"].copy()
print(f"Readouts with prices: {len(ro):,} | topline: {len(top):,} ({top['outcome'].value_counts().to_dict()}) | firms kept: {len(KEEP_CIKS):,}")


# %% [markdown]
# ## Step 2: Insider transactions from the SEC's quarterly data sets
# For each quarter: download the zip (two possible SEC paths), keep Forms 4 and 4/A of the sample's firms, and save the non-derivative transactions with the reporting owner's role and a 10b5-1 flag. Quarters already saved are skipped.


# %% INSIDERS
BASES = ["https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/",
         "https://www.sec.gov/files/datastandardsinnovation/data/insider-transactions-data-sets/"]
PLAN_RX = re.compile(r"10b\s*5\s*-?\s*1", re.I)

def quarter_list(first, last):
    y, q = int(first[:4]), int(first[-1])
    if last is None:
        t = pd.Timestamp.today()
        ly, lq = t.year, (t.month - 1) // 3 + 1
    else:
        ly, lq = int(last[:4]), int(last[-1])
    out = []
    while (y, q) <= (ly, lq):
        out.append(f"{y}q{q}")
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out

def get_zip(qname):
    p = os.path.join(ZIP_DIR, f"{qname}_form345.zip")
    if os.path.exists(p) and zipfile.is_zipfile(p):
        return p
    for b in BASES:
        r = SEC.get(b + f"{qname}_form345.zip")
        if r.status_code == 200 and r.content[:2] == b"PK":
            with open(p, "wb") as f:
                f.write(r.content)
            return p
    return None

def read_table(zf, name, usecols):
    member = next((n for n in zf.namelist() if os.path.basename(n).upper().startswith(name)
                   and n.lower().endswith((".tsv", ".txt"))), None)
    if member is None:
        return pd.DataFrame(columns=usecols)
    with zf.open(member) as fh:
        df = pd.read_csv(fh, sep="\t", dtype=str, quoting=3, encoding_errors="replace", on_bad_lines="skip",
                         usecols=lambda c: c in usecols)
    for c in usecols:
        if c not in df:
            df[c] = None
    return df[usecols]

COLS = ["accession", "cik", "filing_date", "doc", "trans_date", "code", "ad", "shares", "price", "value",
        "owner", "officer", "director", "ten_pct", "title", "plan"]

def process(qname):
    out_p = os.path.join(QDIR, f"{qname}.parquet")
    if os.path.exists(out_p):
        return pd.read_parquet(out_p), "saved"
    zp = get_zip(qname)
    if zp is None:
        return None, "not posted"
    with zipfile.ZipFile(zp) as zf:
        sub = read_table(zf, "SUBMISSION", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "REMARKS", "AFF10B5ONE"])
        sub["cik"] = pd.to_numeric(sub["ISSUERCIK"], errors="coerce")
        sub = sub[sub["cik"].isin(KEEP_CIKS) & sub["DOCUMENT_TYPE"].isin(["4", "4/A"])].copy()
        acc = set(sub["ACCESSION_NUMBER"])
        own = read_table(zf, "REPORTINGOWNER", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"])
        own = own[own["ACCESSION_NUMBER"].isin(acc)]
        tr = read_table(zf, "NONDERIV_TRANS", ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                                               "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"])
        tr = tr[tr["ACCESSION_NUMBER"].isin(acc)]
        fn = read_table(zf, "FOOTNOTES", ["ACCESSION_NUMBER", "FOOTNOTE_TXT"])
        fn = fn[fn["ACCESSION_NUMBER"].isin(acc)]
    if not KEEP_ZIPS:
        os.remove(zp)
    rel = own.assign(r=own["RPTOWNER_RELATIONSHIP"].fillna("").str.upper()).groupby("ACCESSION_NUMBER").agg(
        owner=("RPTOWNERCIK", "first"), officer=("r", lambda s: s.str.contains("OFFICER").any()),
        director=("r", lambda s: s.str.contains("DIRECTOR").any()), ten_pct=("r", lambda s: s.str.contains("TENPERCENT").any()),
        title=("RPTOWNER_TITLE", "first"))
    plan_fn = set(fn.loc[fn["FOOTNOTE_TXT"].fillna("").str.contains(PLAN_RX), "ACCESSION_NUMBER"])
    s = sub.set_index("ACCESSION_NUMBER")
    flag = s["AFF10B5ONE"].fillna("").astype(str).str.strip().str.lower().isin(["1", "true", "y", "yes", "x"])
    plan = flag | s["REMARKS"].fillna("").str.contains(PLAN_RX) | s.index.isin(plan_fn)
    t = tr.rename(columns={"ACCESSION_NUMBER": "accession"})
    t = t.join(s[["cik", "FILING_DATE", "DOCUMENT_TYPE"]], on="accession").join(rel, on="accession")
    t["plan"] = t["accession"].map(plan).fillna(False).astype(bool)
    out = pd.DataFrame({"accession": t["accession"], "cik": t["cik"].astype("Int64"), "filing_date": sec_date(t["FILING_DATE"]),
                        "doc": t["DOCUMENT_TYPE"], "trans_date": sec_date(t["TRANS_DATE"]), "code": t["TRANS_CODE"].fillna("").str.strip(),
                        "ad": t["TRANS_ACQUIRED_DISP_CD"].fillna("").str.strip(),
                        "shares": pd.to_numeric(t["TRANS_SHARES"], errors="coerce"),
                        "price": pd.to_numeric(t["TRANS_PRICEPERSHARE"], errors="coerce"), "owner": t["owner"],
                        "officer": t["officer"].fillna(False).astype(bool), "director": t["director"].fillna(False).astype(bool),
                        "ten_pct": t["ten_pct"].fillna(False).astype(bool), "title": t["title"], "plan": t["plan"]}, columns=COLS[:9] + COLS[10:])
    out.insert(9, "value", out["shares"] * out["price"])
    out.to_parquet(out_p, index=False)
    return out, "downloaded"

qs = quarter_list(FIRST_Q, LAST_Q)
parts, status, t0 = [], {}, time.time()
for k, q in enumerate(qs, 1):
    df, st = process(q)
    status[st] = status.get(st, 0) + 1
    if df is not None:
        parts.append(df)
    if k % 8 == 0 or k == len(qs):
        print(f"  {k}/{len(qs)} quarters | {status} | {SEC.n_requests} requests | {time.time() - t0:.0f}s")
T = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)
T = T.dropna(subset=["cik", "trans_date"]).copy()
T["cik"] = T["cik"].astype(int)
# an amendment repeats the original: keep a transaction once per owner, day, code, shares and price
T = T.sort_values(["doc"]).drop_duplicates(["cik", "owner", "trans_date", "code", "shares", "price"], keep="first")
T["od"] = T["officer"] | T["director"]
T.to_parquet(os.path.join(NB9_DIR, "insider_trades.parquet"), index=False)
print(f"Transactions kept: {len(T):,} at {T['cik'].nunique():,} firms | dates {T['trans_date'].min():%Y-%m} to {T['trans_date'].max():%Y-%m}")
print("Codes:", T["code"].value_counts().head(8).to_dict())
os_ = T[(T["code"] == "S") & T["od"]]
print(f"Open-market sales by officers/directors: {len(os_):,} | flagged as 10b5-1 plan sales: {os_['plan'].mean():.0%} "
      f"(before 2023: {os_.loc[os_['trans_date'] < '2023-01-01', 'plan'].mean():.0%}, from 2023: "
      f"{os_.loc[os_['trans_date'] >= '2023-01-01', 'plan'].mean():.0%})")


# %% [markdown]
# ## Step 3: Insider trading before readouts
# Per readout and window: did an officer or director sell (any, through a plan, not through a plan), how much, and did any buy. Firms with no Form 4 in the data are left out. Then: failure vs success, and failure vs success before vs after the reform.


# %% READOUTS
has4 = set(T["cik"])
E = top[top["cik"].isin(has4) & top["outcome"].isin(["success", "failure", "mixed"])][
    ["accession", "cik", "event_date", "day0", "outcome", "year", "log_dollar_vol", "car_1_1", "car_m10_m2"]].copy()
E["fail"] = (E["outcome"] == "failure").astype(int)
E["post"] = (E["event_date"] >= pd.Timestamp(POST_START)).astype(int)
print(f"Topline readouts at firms with Form 4 data: {len(E):,} of {len(top[top['outcome'].isin(['success', 'failure', 'mixed'])]):,} "
      f"| after {POST_START}: {E['post'].sum():,}")

TS = T[T["code"].isin(["S", "P"])][["cik", "trans_date", "code", "value", "owner", "od", "ten_pct", "plan"]]
M = E[["accession", "cik", "event_date"]].merge(TS, on="cik")
M["d"] = (M["trans_date"] - M["event_date"]).dt.days

def window_stats(a, b, tag):
    x = M[(M["d"] >= a) & (M["d"] <= b)]
    sell, buy = x[(x["code"] == "S") & x["od"]], x[(x["code"] == "P") & x["od"]]
    g = pd.DataFrame(index=E["accession"])
    g[f"sell_{tag}"] = g.index.isin(sell["accession"]).astype(int)
    g[f"sell_plan_{tag}"] = g.index.isin(sell.loc[sell["plan"], "accession"]).astype(int)
    g[f"sell_noplan_{tag}"] = g.index.isin(sell.loc[~sell["plan"], "accession"]).astype(int)
    g[f"buy_{tag}"] = g.index.isin(buy["accession"]).astype(int)
    g[f"sellers_{tag}"] = sell.groupby("accession")["owner"].nunique().reindex(g.index).fillna(0)
    g[f"nsell_{tag}"] = sell.groupby("accession").size().reindex(g.index).fillna(0)
    g[f"sell_usd_{tag}"] = sell.groupby("accession")["value"].sum().reindex(g.index).fillna(0)
    g[f"ten_sell_{tag}"] = g.index.isin(x.loc[(x["code"] == "S") & x["ten_pct"] & ~x["od"], "accession"]).astype(int)
    return g.reset_index()

def tag(a, b):
    return f"{a}_{b}".replace("-", "m")

for a, b in WINDOWS + [AFTER, BASE]:
    E = E.merge(window_stats(a, b, tag(a, b)), on="accession", how="left")
MT, BT = tag(*MAIN), tag(*BASE)
E["base_sell"] = np.log1p(E[f"nsell_{BT}"])
E["log_sell_usd"] = np.log1p(E[f"sell_usd_{MT}"])
E.to_parquet(os.path.join(NB9_DIR, "readout_insiders.parquet"), index=False)

print(f"\n=== Share of topline readouts with an officer/director open-market sale or purchase (95% interval) ===")
for (a, b) in WINDOWS + [AFTER]:
    tg = tag(a, b)
    print(f"-- days {a:+d}..{b:+d}")
    for oc in ["success", "failure", "mixed"]:
        x = E[E["outcome"] == oc]
        line = []
        for v in ["sell", "sell_noplan", "sell_plan", "buy"]:
            k, n = int(x[f"{v}_{tg}"].sum()), len(x)
            lo, hi = wilson(k, n)
            line.append(f"{v} {k / n if n else np.nan:5.0%} ({lo:.0%}-{hi:.0%})")
        print(f"   {oc:8s} n = {len(x):5,} | " + " | ".join(line))

print(f"\n=== Dollars sold by officers/directors in days {MAIN[0]:+d}..{MAIN[1]:+d} ===")
for oc in ["success", "failure", "mixed"]:
    x = E[E["outcome"] == oc]
    print(f"  {oc:8s} total ${x[f'sell_usd_{MT}'].sum() / 1e6:,.1f}m | mean per readout ${x[f'sell_usd_{MT}'].mean() / 1e3:,.0f}k | "
          f"readouts with a sale: {x[f'sell_{MT}'].mean():.0%}")
fx = E[E["outcome"] == "failure"]
print(f"  Losses avoided by pre-failure sellers, at the readout move: ${(fx[f'sell_usd_{MT}'] * (-fx['car_1_1']).clip(lower=0)).sum() / 1e6:,.1f}m")

print("\n=== Do insiders sell more before failures? (failure vs success; controls: usual selling, size, year; clustered by firm) ===")
S2 = E[E["outcome"].isin(["success", "failure"])].dropna(subset=["log_dollar_vol"]).copy()
res = {}
for y, kind in [(f"sell_{MT}", "logit"), (f"sell_noplan_{MT}", "logit"), (f"sell_plan_{MT}", "logit"), (f"buy_{MT}", "logit"),
                ("log_sell_usd", "ols")]:
    if S2[y].sum() < 10 or S2[y].nunique() < 2:
        print(f"  {y}: too few")
        continue
    try:
        m = fit(f"{y} ~ fail + base_sell + log_dollar_vol + C(year)", S2, kind)
        res[y] = m
        show(m, ["fail", "base_sell"], f"{y} ({kind})", odds=(kind == "logit"))
    except Exception as e:
        print(f"  {y}: did not fit ({type(e).__name__}: {str(e)[:100]})")

print(f"\n=== The 10b5-1 reform: failure-minus-success gap before vs after {POST_START} ===")
for y in [f"sell_{MT}", f"sell_plan_{MT}", f"sell_noplan_{MT}"]:
    tab = S2.groupby(["post", "outcome"])[y].agg(["mean", "size"]).unstack("outcome")
    for p_, lab in [(0, "before"), (1, "after")]:
        if p_ in tab.index:
            r = tab.loc[p_]
            print(f"  {y:18s} {lab:6s}: failure {r[('mean', 'failure')]:.0%} (n = {int(r[('size', 'failure')])}) | success "
                  f"{r[('mean', 'success')]:.0%} (n = {int(r[('size', 'success')])}) | gap {r[('mean', 'failure')] - r[('mean', 'success')]:+.0%}")
res_did = {}
for y in [f"sell_{MT}", f"sell_plan_{MT}", f"sell_noplan_{MT}", "log_sell_usd"]:
    if S2.loc[S2["post"] == 1, y].sum() < 5:
        print(f"  {y}: too few after the reform for the interaction")
        continue
    try:
        m = smf.ols(f"{y} ~ fail * post + base_sell + log_dollar_vol + C(year)", S2).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(S2["cik"])[0]})
        res_did[y] = m
        show(m, ["fail", "fail:post", "base_sell"], f"{y}: linear probability, year fixed effects")
    except Exception as e:
        print(f"  {y}: did not fit ({type(e).__name__}: {str(e)[:100]})")
print("  (fail = the gap before the reform; fail:post = how much it changed after; a negative fail:post means the reform closed the gap)")

print("\n=== Did pre-readout insider buying go with the price run-up (days -10..-2)? ===")
for v, lab in [(1, "insider bought"), (0, "no insider buying")]:
    x = E[(E["outcome"] == "success") & (E[f"buy_{MT}"] == v)]
    print(f"  successes, {lab:18s}: run-up {mean_t(x.assign(day0=x['event_date']), 'car_m10_m2', 'day0')}")


# %% [markdown]
# ## Step 4: The Friday test
# Filing day from the 8-K's acceptance time (Eastern). Friday = filed on Friday or the weekend. Bad news = negative or mixed topline readouts; also stop 8-Ks (Notebook 4's first-disclosure events through an 8-K). If investors pay less attention on Fridays, Friday bad news should get a smaller immediate reaction, more drift, and less abnormal volume.


# %% FRIDAY
F = top[top["outcome"].isin(["success", "failure", "mixed"])].copy()
fday = as_day(F["acceptance_et"]) if "acceptance_et" in F else F["event_date"]
F["fri"] = (fday.dt.dayofweek >= 4).astype(int)
F["bad"] = F["outcome"].isin(["failure", "mixed"]).astype(int)
F["neg"] = (F["outcome"] == "failure").astype(int)
F["log_abn_volume"] = np.log(F["abn_volume"].where(F["abn_volume"] > 0)) if "abn_volume" in F else np.nan
print(f"Topline readouts: {len(F):,} | filed Friday or weekend: {F['fri'].sum():,} "
      f"(bad news {F.loc[F['bad'] == 1, 'fri'].mean():.0%}, good news {F.loc[F['bad'] == 0, 'fri'].mean():.0%})")

print("\n=== Mean reaction by news and filing day (t clustered by firm and date) ===")
for b_, lab in [(1, "bad news (negative or mixed)"), (0, "good news")]:
    for f_, dl in [(1, "Friday/weekend"), (0, "Mon-Thu")]:
        x = F[(F["bad"] == b_) & (F["fri"] == f_)]
        print(f"  {lab:29s} {dl:15s} n = {len(x):4,} | days -1..+1 {mean_t(x, 'car_1_1')} | days +2..+20 {mean_t(x, 'car_2_20')}"
              + (f" | abnormal volume {x['abn_volume'].median():.1f}x (median)" if "abn_volume" in x else ""))
neg = F[F["outcome"] == "failure"]
for f_, dl in [(1, "Friday/weekend"), (0, "Mon-Thu")]:
    x = neg[neg["fri"] == f_]
    print(f"  {'negative only':29s} {dl:15s} n = {len(x):4,} | days -1..+1 {mean_t(x, 'car_1_1')} | days +2..+20 {mean_t(x, 'car_2_20')}")

res_fri = {}
print("\n=== Regressions, topline readouts (clustered by firm; year fixed effects; tails clipped at 1%) ===")
G = F.dropna(subset=["log_dollar_vol"]).copy()
for y in ["car_1_1", "car_2_20", "log_abn_volume"]:
    if y not in G or G[y].notna().sum() < 50:
        continue
    z = G.dropna(subset=[y]).copy()
    if y != "log_abn_volume":
        lo, hi = z[y].quantile([0.01, 0.99])
        z[y] = z[y].clip(lo, hi)
    m = fit(f"{y} ~ bad * fri + log_dollar_vol + C(year)", z)
    res_fri[y] = m
    show(m, ["bad", "fri", "bad:fri"], y)
print("  (bad:fri > 0 for days -1..+1 means a smaller immediate drop for Friday bad news; bad:fri < 0 for days +2..+20 means more drift)")

print("\n=== Do firms choose Fridays for bad news? (logit, clustered by firm) ===")
try:
    m = fit("fri ~ bad + log_dollar_vol + C(year)", G, "logit")
    res_fri["choice"] = m
    show(m, ["bad"], "Friday filing on bad news", odds=True)
except Exception as e:
    print(f"  did not fit ({type(e).__name__}: {str(e)[:100]})")

if len(dse) and "first_date" in dse:
    Q = dse.copy()
    Q["first_date"] = as_day(Q["first_date"])
    Q = Q[col(Q, "channel", "").fillna("").astype(str).str.contains("8-K") & col(Q, "car_1_1", np.nan).notna()].copy()
    Q["fri"] = (Q["first_date"].dt.dayofweek >= 4).astype(int)
    Q["cik"] = pd.to_numeric(Q["cik"], errors="coerce")
    Q["day0"] = col(Q, "day0", Q["first_date"])
    print(f"\n=== Stop 8-Ks (first disclosure), n = {len(Q):,} with prices ===")
    for f_, dl in [(1, "Friday/weekend"), (0, "Mon-Thu")]:
        x = Q[Q["fri"] == f_]
        print(f"  {dl:15s} n = {len(x):4,} | days -1..+1 {mean_t(x, 'car_1_1')} | days +2..+20 {mean_t(x, 'car_2_20')}")


# %% [markdown]
# ## Step 5: Figures
# Check the printed output of Steps 2 to 4 and both figures.


# %% FIGS
import matplotlib.pyplot as plt
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"success": "#2a78d6", "failure": "#eb6834", "mixed": "#eda100"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "legend.frameon": False, "axes.axisbelow": True})
DAYS = np.arange(-180, 91)
sl = M[(M["code"] == "S") & M["od"] & M["d"].between(DAYS[0], DAYS[-1])][["accession", "d"]].drop_duplicates()
sl = sl.merge(E[["accession", "outcome"]], on="accession")
fig, ax = plt.subplots(figsize=(8.5, 4))
for oc in ["success", "failure"]:
    n = (E["outcome"] == oc).sum()
    if not n:
        continue
    daily = sl[sl["outcome"] == oc].groupby("d")["accession"].nunique().reindex(DAYS, fill_value=0) / n * 100
    ax.plot(DAYS, daily.rolling(15, center=True, min_periods=5).mean(), color=COL[oc], linewidth=2.2, label=f"{oc} (n = {n:,})")
ax.axvline(0, color=INK2, linewidth=1)
ax.set_xlabel("Calendar days from the topline readout")
ax.set_ylabel("Readouts with an insider sale that day (%)")
ax.set_title("Officer and director open-market sales around topline readouts (15-day average)", loc="left", fontsize=11)
ax.legend(fontsize=9, loc="upper left")
fig.savefig(os.path.join(FIG_DIR, "fig16_insiders.png"), dpi=200, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
for ax, (y, lab) in zip(axes, [("car_1_1", "Immediate: days -1..+1"), ("car_2_20", "Delayed: days +2..+20")]):
    vals, errs, labels = [], [], []
    for b_, bl in [(1, "bad"), (0, "good")]:
        for f_, fl_ in [(0, "Mon-Thu"), (1, "Fri")]:
            x = F[(F["bad"] == b_) & (F["fri"] == f_)][y].dropna()
            lo, hi = x.quantile([0.01, 0.99]) if len(x) > 20 else (x.min(), x.max())
            x = x.clip(lo, hi)
            vals.append(x.mean() * 100)
            errs.append(1.96 * x.std(ddof=1) / np.sqrt(len(x)) * 100 if len(x) > 1 else np.nan)
            labels.append(f"{bl}\n{fl_}\n(n = {len(x)})")
    colors = ["#eb6834", "#f4a582", "#2a78d6", "#92c5de"]
    ax.bar(range(4), vals, yerr=errs, color=colors, edgecolor=SURF, linewidth=2, ecolor=INK2)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.grid(axis="x", visible=False)
    ax.set_title(lab, loc="left", fontsize=11)
axes[0].set_ylabel("Mean abnormal return (%), tails clipped")
fig.suptitle("Friday vs Monday-Thursday topline readouts", x=0.01, ha="left", fontweight="bold", color=INK)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig17_friday.png"), dpi=200, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## What next
# - Check the printed output of Steps 2 to 4 and both figures.
# - Insiders: a positive `fail` coefficient (more selling before failures) is informed trading; a negative `fail:post` means the 2023 reform closed the gap. A zero `fail` with a dip in selling just before readouts (fig16) means blackout policies work.
# - Friday: positive `bad:fri` for days -1..+1 and negative for days +2..+20 is the inattention result; a positive `bad` in the choice logit is strategic timing.
# - Insider results are about averages across firms; the paper should not name individuals or firms as trading on inside information.

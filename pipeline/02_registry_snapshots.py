"""Notebook 2 — Clinical trials: which firm runs them, and how they changed month by month

Pipeline stage of the who-knows-first replication code. Cells are separated by '# %%' markers, so the file
runs top to bottom as a script and opens as a notebook in Jupyter, VS Code or (after tools/make_notebooks.py) Colab.
See README.md for inputs, outputs and the run order.
"""


# %% [markdown]
# # Notebook 2 — Clinical trials: which firm runs them, and how they changed month by month
#
# Builds the registry half of the pilot for the firms found in Notebook 1. It:
#
# 1. Loads the firm list Notebook 1 saved on Drive.
# 2. Downloads the current record of every industry-sponsored interventional trial from ClinicalTrials.gov's official API.
# 3. Matches each trial sponsor to a firm by name (exact, former name, close spelling or your own `ALIASES`) and writes a sample to check by hand.
# 4. Lists AACT's monthly snapshots of ClinicalTrials.gov through the AACT API. AACT is run by CTTI, a partnership co-founded by Duke and the FDA.
# 5. Goes through those snapshots one month at a time. It keeps each matched trial's status, dates, enrollment, lead sponsor and primary outcomes, and deletes the large snapshot file before the next month.
#
# **How to run**
#
# - Needs Notebook 1's firm list on Drive, and an AACT API key (AACT > Settings > API keys) saved in Colab Secrets as `AACT_API` with Notebook access on.
# - Edit `USER_AGENT` in the first code cell, and add `ALIASES` once you've seen Step 3's output.
# - **First run:** leave `DRY_RUN = True` and choose *Runtime → Run all*. Steps 1–3 run in full; Step 5 processes four months spread over the archive, to check each era of AACT's format. Check the last cell's output.
# - **Full run:** set `DRY_RUN = False` and *Run all*. Steps 2–3 reuse saved files; Step 5 processes every month and skips months already done, so after a disconnect just *Run all* again.
#
# **Terms:** ClinicalTrials.gov data are free to use. Credit ClinicalTrials.gov and AACT/CTTI as sources, give the date the data were processed, and describe any changes you made.
#
# **Output** (in `MyDrive/biopharma_tiv/ctgov/`): current trial records, `sponsor_matches.csv`, `audit_sample.csv`, `trial_firm.parquet`, and `aact_monthly/` with one small file per month.


# %% CONFIG
# ==== Settings: edit USER_AGENT; leave the rest unless you have a reason ====
import os
USER_AGENT = os.environ.get("SEC_USER_AGENT", "Your Name your.email@example.com")
PROJECT_DIR = os.environ.get("WKF_PROJECT_DIR", "/content/drive/MyDrive/biopharma_tiv")
UNIVERSE_FILE = None          # None = the newest universe_*.parquet that Notebook 1 saved

INCLUDE_COLLABORATOR_TRIALS = True    # also link trials where the firm is an industry collaborator
FUZZY_THRESHOLD = 92          # 0-100; how close a spelling must be to count as a match

# Sponsor names on ClinicalTrials.gov that belong to a firm under another name (e.g. a subsidiary).
# Use find_firm("johnson") after Step 1 to look up CIKs; use firms that file 8-Ks.
ALIASES = {
    # "Janssen Research & Development, LLC": 0,          # Johnson & Johnson
    # "Johnson & Johnson Pharmaceutical Research & Development, L.L.C.": 0,
    # "Merck Sharp & Dohme LLC": 0,                       # Merck & Co. (MRK)
    # "Abbott": 0,                                        # Abbott Laboratories
    # "Biogen": 0,                                        # current Biogen (BIIB)
}

AACT_FIRST_MONTH = "2010-01"   # earliest monthly snapshot to use (the archive may start later)
AACT_LAST_MONTH  = "2026-01"   # the 1 January 2026 snapshot shows trials as of the end of 2025

DRY_RUN   = True    # first run: True (Step 5 does four months spread over the archive). Then set to False.
MAX_RPS   = 2.0     # requests per second to ClinicalTrials.gov's API
MIN_FREE_GB = 8     # Step 5 pauses if the Colab disk has less free space than this


# %% [markdown]
# ## Step 0 — Setup


# %% SETUP
import os, re, io, csv, json, time, glob, gzip, shutil, random, hashlib, zipfile, threading, datetime as dt
import requests
import pandas as pd
try:
    from rapidfuzz import process, fuzz
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rapidfuzz"], check=True)
    from rapidfuzz import process, fuzz

assert "@" in USER_AGENT and "Your Name" not in USER_AGENT, "Set USER_AGENT to your real name and email first (first code cell)."
if PROJECT_DIR.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")

EDGAR_DIR = os.path.join(PROJECT_DIR, "edgar")
CTG_DIR   = os.path.join(PROJECT_DIR, "ctgov")
AACT_DIR  = os.path.join(CTG_DIR, "aact_dryrun" if DRY_RUN else "aact_monthly")
LOCAL_TMP = "/content/tmp" if os.path.isdir("/content") else "/tmp/tiv_tmp"
for d in (CTG_DIR, AACT_DIR, LOCAL_TMP):
    os.makedirs(d, exist_ok=True)
print("Saving to:", CTG_DIR)


# %% HTTP
# ---- Polite client: identifies itself, retries, and slows down when the site pushes back ----
class SiteBlocked(Exception):
    pass

class PoliteClient:
    RETRY = {408, 421, 425, 429, 500, 502, 503, 504, 403}

    def __init__(self, user_agent, max_rps=2.0, min_rps=0.2):
        self.ua, self.max_rps, self.min_rps = user_agent, max_rps, min_rps
        self.rps = max_rps
        self._lock = threading.Lock()
        self._next = 0.0
        self._ok_streak = 0
        self._local = threading.local()
        self.n_requests = 0
        self.consec_403 = 0

    def _session(self, reset=False):
        if reset or getattr(self._local, "s", None) is None:
            s = requests.Session()
            s.headers.update({"User-Agent": self.ua, "Accept": "application/json"})
            self._local.s = s
        return self._local.s

    def _wait_turn(self):
        with self._lock:
            t = max(time.monotonic(), self._next)
            self._next = t + 1.0 / self.rps
        delay = t - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def _slow_down(self):
        with self._lock:
            old = self.rps
            self.rps = max(self.min_rps, self.rps / 2)
            self._ok_streak = 0
        if self.rps < old:
            print(f"  site pushed back: slowing to {self.rps:.2f} requests/second")

    def _speed_up(self):
        with self._lock:
            self._ok_streak += 1
            if self._ok_streak >= 200 and self.rps < self.max_rps:
                self.rps = min(self.max_rps, self.rps * 1.25)
                self._ok_streak = 0

    def _fetch(self, url, params, timeout):
        return self._session().get(url, params=params, timeout=timeout)

    def get(self, url, params=None, timeout=90, max_tries=7):
        last = None
        for attempt in range(max_tries):
            self._wait_turn()
            try:
                r = self._fetch(url, params, timeout)
                self.n_requests += 1
            except requests.RequestException as e:
                last = f"{type(e).__name__}: {e}"
                self._session(reset=True)
            else:
                with self._lock:
                    self.consec_403 = self.consec_403 + 1 if r.status_code == 403 else 0
                    blocked = self.consec_403 >= 20
                if blocked:
                    raise SiteBlocked("ClinicalTrials.gov keeps refusing requests (HTTP 403). Stop here and try again later.")
                if r.status_code not in self.RETRY:
                    if r.status_code == 200:
                        self._speed_up()
                    return r                          # 200, 400, 404 ...: the caller decides
                last = f"HTTP {r.status_code}"
                if r.status_code in (429, 503, 403):
                    self._slow_down()
                if r.status_code == 421:
                    self._session(reset=True)
                wait = r.headers.get("Retry-After")
                if wait and str(wait).isdigit():
                    time.sleep(min(int(wait), 300))
            time.sleep(min(120, 2 ** attempt + random.random()))
        raise RuntimeError(f"GET failed after {max_tries} tries: {url} ({last})")

CT = PoliteClient(USER_AGENT, MAX_RPS)
print("Client ready:", USER_AGENT, f"| start at {MAX_RPS} requests/second")


# %% [markdown]
# ## Step 1 — Firm names from Notebook 1
# Loads the firm list and every name each firm has used (current and former), cleaned so that "ACME BIO INC /DE/" and "Acme Bio, Inc." compare as equal.


# %% FIRMS
uni_files = sorted(glob.glob(os.path.join(EDGAR_DIR, "universe_*.parquet")), key=os.path.getmtime)
UNI_PATH = UNIVERSE_FILE or (uni_files[-1] if uni_files else None)
assert UNI_PATH and os.path.exists(UNI_PATH), "No firm list found: run Notebook 1's Steps 1-2 first."
universe = pd.read_parquet(UNI_PATH)
fi_files = sorted(glob.glob(os.path.join(EDGAR_DIR, "filing_index_*.parquet")), key=os.path.getmtime)
ciks_with_8k = set(pd.read_parquet(fi_files[-1], columns=["cik"])["cik"].astype(int)) if fi_files else set()
print(f"Firm list: {os.path.basename(UNI_PATH)} ({len(universe):,} firms; {len(ciks_with_8k):,} with 8-Ks in the filing list)")

SUFFIXES = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "llc", "lp", "llp",
            "plc", "sa", "ag", "nv", "bv", "se", "gmbh", "kk", "spa", "ab", "as", "asa", "oyj", "holdings", "holding",
            "group", "and", "the", "usa", "us"}
CANON = {"pharmaceuticals": "pharma", "pharmaceutical": "pharma", "pharm": "pharma", "therapeutic": "therapeutics",
         "biosciences": "bioscience", "technologies": "technology", "laboratories": "labs", "laboratory": "labs",
         "international": "intl", "biopharmaceuticals": "biopharma", "biopharmaceutical": "biopharma"}

def norm_name(s):
    s = str(s or "").lower()
    s = re.sub(r"/[a-z]{1,4}/", " ", s)            # SEC state tags such as /DE/
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = [CANON.get(t, t) for t in s.split()]
    while toks and toks[-1] in SUFFIXES:
        toks.pop()
    while toks and toks[0] == "the":
        toks.pop(0)
    return " ".join(toks)

rows = []
for r in universe.itertuples():
    rows.append((int(r.cik), r.name, False))
    for fn in str(r.former_names or "").split(";"):
        if fn.strip():
            rows.append((int(r.cik), fn.strip(), True))
firm_names = pd.DataFrame(rows, columns=["cik", "variant", "is_former"])
firm_names["norm"] = firm_names["variant"].map(norm_name)
firm_names = firm_names[firm_names["norm"].str.len() >= 3].drop_duplicates(["cik", "norm"])
norm2ciks = firm_names.groupby("norm")["cik"].agg(lambda s: sorted(set(s))).to_dict()
norm_is_former = firm_names.groupby("norm")["is_former"].all().to_dict()
cik2name = universe.set_index(universe["cik"].astype(int))["name"].to_dict()
print(f"{len(firm_names):,} name variants for matching")

def find_firm(text):
    """Look up firms by part of a current or former name, e.g. find_firm("johnson")."""
    t = text.lower()
    m = universe["name"].str.lower().str.contains(t, regex=False) | \
        universe["former_names"].fillna("").str.lower().str.contains(t, regex=False)
    out = universe.loc[m, ["cik", "name", "tickers"]].assign(files_8k=universe.loc[m, "cik"].astype(int).isin(ciks_with_8k))
    return out if len(out) else "no firm matches"


# %% [markdown]
# ## Step 2 — Current records of industry trials
# Downloads every interventional trial with an industry lead sponsor (and, if chosen, an industry collaborator), 1,000 records per request. Skipped if the file exists.


# %% CURRENT
CURRENT_PATH = os.path.join(CTG_DIR, f"industry_trials_current{'_withcollab' if INCLUDE_COLLABORATOR_TRIALS else ''}.parquet")
API = "https://clinicaltrials.gov/api/v2/studies"
PIECES = ("NCTId,BriefTitle,OrgStudyId,SecondaryId,LeadSponsorName,LeadSponsorClass,CollaboratorName,CollaboratorClass,"
          "StudyType,Phase,OverallStatus,WhyStopped,StartDate,StartDateType,PrimaryCompletionDate,PrimaryCompletionDateType,"
          "CompletionDate,CompletionDateType,StudyFirstSubmitDate,StudyFirstPostDate,LastUpdatePostDate,ResultsFirstPostDate,"
          "EnrollmentCount,EnrollmentType,InterventionType,InterventionName,Condition")
SPONS = ("(AREA[LeadSponsorClass]INDUSTRY OR AREA[CollaboratorClass]INDUSTRY)" if INCLUDE_COLLABORATOR_TRIALS
         else "AREA[LeadSponsorClass]INDUSTRY")
ADV = f"{SPONS} AND AREA[StudyType]INTERVENTIONAL"
ATTEMPTS = [{"fields": PIECES, "filter.advanced": ADV},       # smallest download
            {"fields": "protocolSection", "filter.advanced": ADV},
            {"fields": "protocolSection"}]                   # last resort: everything, filtered here

def _g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d

def flatten_current(s):
    p = s.get("protocolSection", {}) or {}
    collab = _g(p, "sponsorCollaboratorsModule", "collaborators", default=[]) or []
    inter = _g(p, "armsInterventionsModule", "interventions", default=[]) or []
    sec = _g(p, "identificationModule", "secondaryIdInfos", default=[]) or []
    st = lambda *k: _g(p, "statusModule", *k)
    return {
        "nct_id": _g(p, "identificationModule", "nctId"),
        "brief_title": _g(p, "identificationModule", "briefTitle"),
        "org_study_id": _g(p, "identificationModule", "orgStudyIdInfo", "id"),
        "secondary_ids": ";".join(str(x.get("id", "")) for x in sec),
        "lead_sponsor": _g(p, "sponsorCollaboratorsModule", "leadSponsor", "name"),
        "lead_class": _g(p, "sponsorCollaboratorsModule", "leadSponsor", "class"),
        "collaborators": "|".join(str(c.get("name", "")) for c in collab),
        "collaborator_classes": "|".join(str(c.get("class", "")) for c in collab),
        "study_type": _g(p, "designModule", "studyType"),
        "phases": "|".join(_g(p, "designModule", "phases", default=[]) or []),
        "overall_status": st("overallStatus"),
        "why_stopped": st("whyStopped"),
        "start_date": st("startDateStruct", "date"), "start_type": st("startDateStruct", "type"),
        "pcd": st("primaryCompletionDateStruct", "date"), "pcd_type": st("primaryCompletionDateStruct", "type"),
        "completion_date": st("completionDateStruct", "date"), "completion_type": st("completionDateStruct", "type"),
        "first_submitted": st("studyFirstSubmitDate"),
        "first_posted": st("studyFirstPostDateStruct", "date"),
        "last_update_posted": st("lastUpdatePostDateStruct", "date"),
        "results_first_posted": st("resultsFirstPostDateStruct", "date"),
        "enrollment": _g(p, "designModule", "enrollmentInfo", "count"),
        "enrollment_type": _g(p, "designModule", "enrollmentInfo", "type"),
        "intervention_types": ";".join(sorted({str(i.get("type", "")) for i in inter})),
        "interventions": ";".join(str(i.get("name", "")) for i in inter)[:2000],
        "conditions": ";".join(_g(p, "conditionsModule", "conditions", default=[]) or [])[:1000],
    }

def fetch_all(params):
    out, token, page, t0 = [], None, 0, time.time()
    while True:
        q = dict(params, pageSize=1000, format="json")
        if page == 0:
            q["countTotal"] = "true"
        if token:
            q["pageToken"] = token
        r = CT.get(API, params=q)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: {r.text[:200]}"
        js = r.json()
        if page == 0:
            print(f"  {js.get('totalCount', '?')} matching records")
        out.extend(flatten_current(s) for s in js.get("studies", []))
        page += 1
        if page % 20 == 0:
            print(f"  page {page}: {len(out):,} records ({time.time() - t0:.0f}s)")
        new = js.get("nextPageToken")
        if not new or new == token:
            return out, None
        token = new

if os.path.exists(CURRENT_PATH):
    cur = pd.read_parquet(CURRENT_PATH)
    print(f"Loaded {len(cur):,} current trial records ({CURRENT_PATH})")
else:
    for params in ATTEMPTS:
        print("Downloading with", {k: (v[:60] + "..." if len(v) > 60 else v) for k, v in params.items()})
        records, err = fetch_all(params)
        if records is not None:
            break
        print("  that request was refused:", err)
    assert records is not None, "Could not download trial records; see the messages above."
    cur = pd.DataFrame(records).dropna(subset=["nct_id"]).drop_duplicates("nct_id")
    industry = (cur["lead_class"] == "INDUSTRY")
    if INCLUDE_COLLABORATOR_TRIALS:
        industry |= cur["collaborator_classes"].fillna("").str.contains("INDUSTRY")
    cur = cur[(cur["study_type"] == "INTERVENTIONAL") & industry].reset_index(drop=True)
    cur.to_parquet(CURRENT_PATH, index=False)
    print(f"Saved {len(cur):,} industry interventional trials to {CURRENT_PATH}")
print(cur["phases"].replace("", "none").value_counts().head(8).to_string())


# %% [markdown]
# ## Step 3 — Match sponsors to firms
# Each industry sponsor name is matched to a firm in this order: your `ALIASES`, then an exact match on the cleaned current name, then on a former name, then a close spelling (score ≥ `FUZZY_THRESHOLD`). Names that fit several firms are left unmatched and listed.
#
# Writes `sponsor_matches.csv` (every sponsor) and `audit_sample.csv` (200 random matches). Open the sample in Google Sheets and mark each row right or wrong: that is the precision check in the paper's kill tests. Re-runs every time, so new `ALIASES` take effect.


# %% MATCH
lead = cur.loc[cur["lead_class"] == "INDUSTRY", ["nct_id", "lead_sponsor"]].rename(columns={"lead_sponsor": "sponsor"})
lead["role"] = "lead"
links = [lead]
if INCLUDE_COLLABORATOR_TRIALS:
    pairs = [(n, name, cls)
             for n, names, classes in zip(cur["nct_id"], cur["collaborators"].fillna(""), cur["collaborator_classes"].fillna(""))
             for name, cls in zip(names.split("|"), classes.split("|"))]
    c = pd.DataFrame(pairs, columns=["nct_id", "sponsor", "cls"])
    c = c[(c["cls"] == "INDUSTRY") & (c["sponsor"].str.len() > 0)][["nct_id", "sponsor"]]
    c["role"] = "collaborator"
    links.append(c)
links = pd.concat(links, ignore_index=True).dropna(subset=["sponsor"])

sponsors = links.groupby("sponsor").size().rename("n_trials").reset_index()
sponsors["norm"] = sponsors["sponsor"].map(norm_name)
alias_norm = {norm_name(k): int(v) for k, v in ALIASES.items()}
choices = list(norm2ciks.keys())

def match_one(name, norm):
    if norm in alias_norm:
        return alias_norm[norm], "alias", 100.0
    if norm in norm2ciks:
        ciks = norm2ciks[norm]
        if len(ciks) > 1:
            return None, "ambiguous:" + ";".join(map(str, ciks)), 100.0
        return ciks[0], "exact_former" if norm_is_former.get(norm) else "exact", 100.0
    if len(norm) >= 6:
        hit = process.extractOne(norm, choices, scorer=fuzz.token_sort_ratio, score_cutoff=FUZZY_THRESHOLD)
        if hit:
            ciks = norm2ciks[hit[0]]
            if len(ciks) == 1:
                return ciks[0], "fuzzy", float(hit[1])
    return None, "none", 0.0

res = [match_one(s, n) for s, n in zip(sponsors["sponsor"], sponsors["norm"])]
sponsors["cik"] = [r[0] for r in res]
sponsors["method"] = [r[1] for r in res]
sponsors["score"] = [r[2] for r in res]
sponsors["firm_name"] = sponsors["cik"].map(cik2name)
sponsors.sort_values("n_trials", ascending=False).to_csv(os.path.join(CTG_DIR, "sponsor_matches.csv"), index=False)

matched = sponsors[sponsors["cik"].notna()]
audit = matched.sample(min(200, len(matched)), random_state=7)[["sponsor", "firm_name", "method", "score", "n_trials"]]
audit.assign(correct="").to_csv(os.path.join(CTG_DIR, "audit_sample.csv"), index=False)

trial_firm = links.merge(matched[["sponsor", "cik", "method"]], on="sponsor")
trial_firm["cik"] = trial_firm["cik"].astype(int)
trial_firm = trial_firm.drop_duplicates(["nct_id", "cik", "role"])
trial_firm.to_parquet(os.path.join(CTG_DIR, "trial_firm.parquet"), index=False)

print(f"{len(sponsors):,} industry sponsor names; matched {len(matched):,} "
      f"({matched['n_trials'].sum() / sponsors['n_trials'].sum():.0%} of sponsor-trial links)")
print(sponsors["method"].str.split(":").str[0].value_counts().to_string())
print(f"\n{trial_firm['cik'].nunique():,} firms linked to {trial_firm['nct_id'].nunique():,} trials "
      f"({trial_firm.loc[trial_firm['role'] == 'lead', 'nct_id'].nunique():,} as lead sponsor)")
print("\nMost trials among matched firms:\n" + trial_firm[trial_firm["role"] == "lead"]["cik"].map(cik2name)
      .value_counts().head(10).to_string())
print("\nUnmatched sponsors with the most trials (candidates for ALIASES):\n" +
      sponsors[sponsors["cik"].isna()].nlargest(25, "n_trials")[["sponsor", "n_trials", "method"]].to_string(index=False))

f8 = universe[universe["cik"].astype(int).isin(ciks_with_8k)]
print(f"\nFirms that file 8-Ks with at least one matched trial: {f8['cik'].astype(int).isin(trial_firm['cik']).mean():.0%} of {len(f8):,}")


# %% [markdown]
# ## Step 4 — List AACT's monthly snapshots
# Reads your AACT API key from Colab Secrets (`AACT_SECRET`, default `AACT_API`), checks it works, and lists AACT's archive of flat-file snapshots (one per month, the first of the month) between `AACT_FIRST_MONTH` and `AACT_LAST_MONTH`. The key is sent only to the AACT API, never printed or saved.


# %% FIND_AACT
import os
AACT_SECRET = "AACT_API"      # the name of your AACT key in Colab Secrets (key icon on the left; Notebook access on)
AACT_API = "https://aact.ctti-clinicaltrials.org/api/v1"

def aact_key():
    try:
        from google.colab import userdata
    except ImportError:                       # not on Colab: use an environment variable instead
        key = os.environ.get("AACT_API_KEY")
        if not key:
            raise RuntimeError("Set the AACT_API_KEY environment variable to your AACT API key.")
        return key
    try:
        return userdata.get(AACT_SECRET)
    except Exception as e:
        raise RuntimeError(f"Could not read the Colab secret '{AACT_SECRET}' ({type(e).__name__}). Open the key icon on the left, "
                           "check the name matches AACT_SECRET, and turn on Notebook access.") from None

AACT_HEADERS = {"Authorization": "Bearer " + aact_key(), "User-Agent": USER_AGENT}

class AACTError(RuntimeError):
    pass

def aact_get(url, params=None, tries=5, **kw):
    """GET from the AACT API with the key. Waits out rate limits (429) and server errors."""
    url = url if url.startswith("http") else AACT_API + url
    assert url.startswith(AACT_API), "the API key is only ever sent to the AACT API"
    for attempt in range(tries):
        r = requests.get(url, headers=AACT_HEADERS, params=params, timeout=60, **kw)
        if r.status_code == 429 or r.status_code >= 500:
            try:
                wait = float(r.headers.get("Retry-After"))
            except (TypeError, ValueError):
                wait = min(120, 5 * 2 ** attempt)
            time.sleep(wait + 1)
            continue
        if r.status_code >= 400:
            try:
                err = r.json().get("error", {})
            except ValueError:
                err = {}
            msg = f"AACT API {r.status_code} {err.get('code', '')}: {err.get('message', r.text[:200])}"
            if r.status_code == 401:
                msg += f"\nCheck the key saved as '{AACT_SECRET}' in Colab Secrets, or make a new one in AACT > Settings > API keys."
            raise AACTError(msg)
        return r
    raise AACTError(f"AACT API still busy after {tries} tries: {url}")

types_ok = [t.get("type") for t in aact_get("/snapshots").json()["data"]]
print("AACT API key works. Snapshot types:", ", ".join(types_ok))

first = f"{AACT_FIRST_MONTH}-01"
last = pd.Period(AACT_LAST_MONTH, "M").end_time.strftime("%Y-%m-%d")
snaps = pd.DataFrame(aact_get("/snapshots/flat/monthly", {"from": first, "to": last}).json()["data"])
if snaps.empty:
    monthly = pd.DataFrame()
    print(f"AACT's archive has no flat-file snapshots between {first} and {last}.")
else:
    snaps["date"] = pd.to_datetime(snaps["created_at"], utc=True).dt.tz_localize(None)
    snaps["month"] = snaps["date"].dt.to_period("M")
    snaps["url"] = AACT_API + "/snapshots/flat/" + snaps["id"].astype(str) + "/download"
    monthly = snaps.sort_values("date").drop_duplicates("month", keep="first").reset_index(drop=True)
    gb = pd.to_numeric(monthly["size_bytes"], errors="coerce").sum() / 1e9 if "size_bytes" in monthly else float("nan")
    print(f"{len(monthly)} monthly flat-file snapshots from {monthly['month'].min()} to {monthly['month'].max()} "
          f"({gb:,.0f} GB in all; each is downloaded, read and deleted in turn)")
    if monthly["month"].min() > pd.Period(AACT_FIRST_MONTH, "M"):
        print(f"The archive starts at {monthly['month'].min()}; earlier months are not available from AACT.")
    gaps = pd.period_range(monthly["month"].min(), monthly["month"].max(), freq="M").difference(monthly["month"])
    if len(gaps):
        print(f"Months with no snapshot in the archive: {len(gaps)} (e.g. {', '.join(map(str, gaps[:6]))})")


# %% [markdown]
# ## Step 5 — Monthly trial histories from AACT
# For each month it asks the AACT API for a download link, downloads the snapshot to the Colab machine and reads three tables: `studies`, `sponsors` and `design_outcomes`. It keeps the rows for your matched trials, saves `aact_YYYYMM.parquet` to Drive, and deletes the snapshot.
#
# AACT allows 30 downloads an hour per key, so downloads are spaced at least two minutes apart.
#
# AACT's format has changed over the years, so each month is put into today's format before saving:
# - **Older column names:** fields under an older name (e.g. `first_received_date`, `start_month_year`) are filled from it.
# - **Dates:** all dates become `YYYY-MM-DD`; a month-only date becomes the 1st.
# - **Codes:** status, phase and date-type text become today's codes (e.g. `Active, not recruiting` → `ACTIVE_NOT_RECRUITING`, `Anticipated` → `ESTIMATED`).
#
# Columns a month lacks are left empty and listed in the summary. `headers.csv` records every column each month's tables had.


# %% AACT
STUDY_COLS = ["nct_id", "overall_status", "last_known_status", "why_stopped", "phase", "study_type", "enrollment",
              "enrollment_type", "start_date", "start_date_type", "primary_completion_date", "primary_completion_date_type",
              "completion_date", "completion_date_type", "study_first_submitted_date", "last_update_submitted_date",
              "last_update_posted_date", "results_first_submitted_date", "results_first_posted_date", "verification_date"]
SPONSOR_COLS = ["nct_id", "agency_class", "lead_or_collaborator", "name"]
OUTCOME_COLS = ["nct_id", "outcome_type", "measure"]
keep_ids = set(trial_firm["nct_id"])
COLUMN_LOG = os.path.join(AACT_DIR, "column_log.csv")    # per month: columns missing, or filled from an older name
HEADER_LOG = os.path.join(AACT_DIR, "headers.csv")       # per month: every column each table had
# Older AACT snapshots use other names for some fields (ClinicalTrials.gov's pre-2017 wording, and month-year text
# columns such as "June 2014" before AACT added date versions). Used only when the current name is absent or empty.
ALT_COLS = {
    "start_date": ["start_month_year"],
    "completion_date": ["completion_month_year"],
    "primary_completion_date": ["primary_completion_month_year"],
    "verification_date": ["verification_month_year"],
    "study_first_submitted_date": ["first_received_date"],
    "last_update_submitted_date": ["last_changed_date"],
    "results_first_submitted_date": ["first_received_results_date"],
}
DATE_COLS = [c for c in STUDY_COLS if c.endswith("_date")]
CODE_COLS = ["overall_status", "last_known_status", "phase", "enrollment_type",
             "start_date_type", "primary_completion_date_type", "completion_date_type"]

def iso_dates(s):
    """Dates in any AACT format ("2014-06-15", "June 2014", "June 15, 2014") -> "YYYY-MM-DD"; month-only -> the 1st."""
    s = s.astype("string").str.strip()
    out = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    for fmt in ("%B %Y", "%B %d, %Y", "%Y-%m", "mixed"):
        todo = out.isna() & s.notna() & (s != "")
        if not todo.any():
            break
        out[todo] = pd.to_datetime(s[todo], format=fmt, errors="coerce")
    return out.dt.strftime("%Y-%m-%d")

def code(s):
    """Status/phase/date-type text in any AACT era -> today's codes ("Active, not recruiting" -> ACTIVE_NOT_RECRUITING,
    "Phase 1/Phase 2" -> PHASE1/PHASE2, "Anticipated" -> ESTIMATED)."""
    s = s.astype("string").str.strip().str.upper().str.replace(r"[^A-Z0-9/]+", "_", regex=True).str.strip("_")
    s = s.str.replace(r"PHASE_(\d)", r"PHASE\1", regex=True)
    return s.replace({"ANTICIPATED": "ESTIMATED", "UNKNOWN_STATUS": "UNKNOWN", "N/A": "NA", "": pd.NA})

_last_download = [0.0]

def download(url, path, tries=5):
    """Ask the AACT API for a fresh signed link (valid 5 minutes), then stream the file to disk."""
    last = None
    for attempt in range(tries):
        try:
            wait = 121 - (time.time() - _last_download[0])     # AACT allows 30 downloads an hour per key
            if wait > 0:
                time.sleep(wait)
            _last_download[0] = time.time()
            r = aact_get(url, allow_redirects=False)
            link = r.headers.get("Location")
            if r.status_code not in (301, 302, 303, 307, 308) or not link:
                raise RuntimeError(f"expected a redirect to the file, got HTTP {r.status_code}")
            # the signed link carries its own permission, so the API key is not sent to it
            with requests.get(link, stream=True, timeout=120, headers={"User-Agent": USER_AGENT}) as f_resp:
                if f_resp.status_code != 200:
                    raise RuntimeError(f"HTTP {f_resp.status_code} from the file link")
                with open(path + ".part", "wb") as f:
                    for block in f_resp.iter_content(8 << 20):
                        f.write(block)
            os.replace(path + ".part", path)
            if zipfile.is_zipfile(path):
                return path
            raise RuntimeError("the file is not a zip")
        except AACTError:
            raise
        except Exception as e:
            last = e
            time.sleep(min(120, 5 * 2 ** attempt))
    raise RuntimeError(f"download failed: {url} ({last})")

def member(zf, table):
    names = [n for n in zf.namelist() if os.path.basename(n).lower() == f"{table}.txt"]
    return names[0] if names else None

def read_table(zf, table, cols):
    """Returns (rows for keep_ids, log of columns missing or filled from an older name, the table's header)."""
    name = member(zf, table)
    if name is None:
        return pd.DataFrame(columns=cols), [(c, "missing") for c in cols], []
    with zf.open(name) as fh:
        header = fh.readline().decode("utf-8", "replace").strip().split("|")
    alts = {c: [a for a in ALT_COLS.get(c, []) if a in header] for c in cols}
    use = [c for c in header if c in cols or any(c in a for a in alts.values())]
    parts = []
    with zf.open(name) as fh:
        for chunk in pd.read_csv(fh, sep="|", usecols=use, dtype=str, quoting=csv.QUOTE_NONE, chunksize=250_000,
                                 on_bad_lines="skip", encoding="utf-8", encoding_errors="replace"):
            parts.append(chunk[chunk["nct_id"].isin(keep_ids)])
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=use)
    log = []
    for c in cols:
        for a in alts[c]:
            fill = df[c].isna() if c in df else pd.Series(True, index=df.index)
            if (fill & df[a].notna()).any() or c not in df:
                df[c] = df[a].where(fill, df[c]) if c in df else df[a]
                log.append((c, "from " + a))
        if c not in header and not alts[c]:
            log.append((c, "missing"))
    return df.reindex(columns=cols), log, header

def summarise_month(zf, snap):
    st, log_s, h_s = read_table(zf, "studies", STUDY_COLS)
    sp, log_p, h_p = read_table(zf, "sponsors", SPONSOR_COLS)
    oc, log_o, h_o = read_table(zf, "design_outcomes", OUTCOME_COLS)
    st = st.drop_duplicates("nct_id")
    for c in DATE_COLS:
        st[c] = iso_dates(st[c])
    for c in CODE_COLS:
        st[c] = code(st[c])
    lead = sp[sp["lead_or_collaborator"].fillna("").str.lower() == "lead"].drop_duplicates("nct_id")
    st["lead_sponsor"] = st["nct_id"].map(lead.set_index("nct_id")["name"])
    if len(oc):
        oc["outcome_type"] = oc["outcome_type"].fillna("").str.lower()
        prim = oc[oc["outcome_type"] == "primary"]
        st["n_primary_outcomes"] = st["nct_id"].map(prim.groupby("nct_id").size()).fillna(0).astype(int)
        st["n_secondary_outcomes"] = st["nct_id"].map(oc[oc["outcome_type"] == "secondary"].groupby("nct_id").size()).fillna(0).astype(int)
        st["primary_outcomes_hash"] = st["nct_id"].map(prim.groupby("nct_id")["measure"].agg(
            lambda s: hashlib.md5("||".join(sorted(x.strip().lower() for x in s.fillna(""))).encode()).hexdigest()[:12]))
    st.insert(1, "snapshot_date", snap.strftime("%Y-%m-%d"))
    log = [("studies." + c, n) for c, n in log_s] + [("sponsors." + c, n) for c, n in log_p] + \
          [("design_outcomes." + c, n) for c, n in log_o]
    headers = {"studies": h_s, "sponsors": h_p, "design_outcomes": h_o}
    return st, log, headers

todo = monthly.copy() if "monthly" in globals() else pd.DataFrame()
if not todo.empty:
    if DRY_RUN:     # four months spread over the archive, to cover each era of AACT's format
        n = len(todo)
        todo = todo.iloc[sorted({round(k * (n - 1) / 3) for k in range(4)})]
    done = {os.path.basename(p)[5:11] for p in glob.glob(os.path.join(AACT_DIR, "aact_*.parquet"))}
    todo = todo[~todo["month"].map(lambda m: m.strftime("%Y%m")).isin(done)]
    print(f"{len(keep_ids):,} matched trials | {len(done)} months done | {len(todo)} months to process"
          + (" (dry run: four months spread over the archive)" if DRY_RUN else ""))
    t0 = time.time()
    for i, row in enumerate(todo.itertuples(), 1):
        free_gb = shutil.disk_usage(LOCAL_TMP).free / 1e9
        if free_gb < MIN_FREE_GB:
            print(f"Only {free_gb:.1f} GB free on the Colab disk; stopping. Runtime > Restart session, then Run all.")
            break
        tag = row.month.strftime("%Y%m")
        zpath = os.path.join(LOCAL_TMP, f"aact_{tag}.zip")
        try:
            download(row.url, zpath)
            with zipfile.ZipFile(zpath) as zf:
                df, log, headers = summarise_month(zf, row.date)
            pd.DataFrame([{"month": tag, "table": t, "columns": "|".join(h)} for t, h in headers.items()]).to_csv(
                HEADER_LOG, mode="a", header=not os.path.exists(HEADER_LOG), index=False)
            if log:
                pd.DataFrame([{"month": tag, "column": c, "note": n} for c, n in log]).to_csv(
                    COLUMN_LOG, mode="a", header=not os.path.exists(COLUMN_LOG), index=False)
            out = os.path.join(AACT_DIR, f"aact_{tag}.parquet")
            df.to_parquet(out + ".tmp", index=False, compression="zstd")
            os.replace(out + ".tmp", out)
            miss = sum(n == "missing" for _, n in log)
            el = time.time() - t0
            print(f"{tag}: {len(df):,} matched trials found | {i}/{len(todo)} months | "
                  f"about {el / i * (len(todo) - i) / 3600:.1f} h left" + (f" | missing columns: {miss}" if miss else ""))
        except AACTError as e:
            print(f"{tag}: STOPPED - {e}")
            break
        except Exception as e:
            print(f"{tag}: FAILED ({type(e).__name__}: {str(e)[:200]})")
        finally:
            if os.path.exists(zpath):
                os.remove(zpath)
else:
    print("No monthly snapshots to process yet (see Step 4).")


# %% [markdown]
# ## Step 6 — Check the results
# Check this cell's output.


# %% SUMMARY
print(f"Run: {'DRY RUN' if DRY_RUN else 'FULL'}")
print(f"Industry trials: {len(cur):,} | linked to firms: {trial_firm['nct_id'].nunique():,} | firms: {trial_firm['cik'].nunique():,}")
paths = sorted(glob.glob(os.path.join(AACT_DIR, "aact_*.parquet")))
if not paths:
    print("No monthly AACT files yet.")
else:
    panel = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    panel["snap"] = pd.to_datetime(panel["snapshot_date"])
    panel = panel.sort_values(["nct_id", "snap"])
    per_month = panel.groupby("snap").size()
    print(f"\nMonths: {len(per_month)} ({per_month.index.min():%Y-%m} to {per_month.index.max():%Y-%m}) | "
          f"matched trials ever seen: {panel['nct_id'].nunique():,} of {len(keep_ids):,}")
    show = per_month.index if len(per_month) <= 6 else per_month.index[[0, len(per_month) // 4, len(per_month) // 2, 3 * len(per_month) // 4, -1]]
    fields = [c for c in panel.columns if c not in ("nct_id", "snapshot_date", "snap")]
    filled = panel[panel["snap"].isin(show)].groupby("snap")[fields].agg(lambda s: s.notna().mean()).T.astype(float).round(2)
    filled.columns = [f"{d:%Y-%m}" for d in filled.columns]
    print("\nMatched trials per snapshot: " + ", ".join(f"{d:%Y-%m} {per_month[d]:,}" for d in show))
    print("Share of rows with each field filled, by snapshot:\n" + filled.to_string())
    st = panel[panel["snap"].isin(show)].groupby("snap")["overall_status"].value_counts().unstack(0).fillna(0).astype(int)
    st.columns = [f"{d:%Y-%m}" for d in st.columns]
    print("\nStatus codes by snapshot (should be the same codes in every column):\n" + st.to_string())
    upd = pd.to_datetime(panel["last_update_submitted_date"], errors="coerce")
    recent = (panel["snap"] - upd).dt.days.between(0, 31)
    fresh = recent.groupby(panel["snap"]).mean()
    if len(fresh) > 2:
        stale = fresh[fresh < fresh.median() / 3]
        print(f"\nShare of trials updated in the month before each snapshot: median {fresh.median():.1%}"
              + (f"; possibly stale snapshots: {', '.join(f'{d:%Y-%m}' for d in stale.index)}" if len(stale) else "; no stale-looking months"))
    if len(per_month) > 1:
        g = panel.groupby("nct_id")
        chg = g["overall_status"].apply(lambda s: (s.fillna("") != s.fillna("").shift()).iloc[1:].sum())
        print(f"Trials whose status changed between snapshots: {(chg > 0).mean():.0%}")
        d0 = pd.to_datetime(g["primary_completion_date"].first(), errors="coerce")
        d1 = pd.to_datetime(g["primary_completion_date"].last(), errors="coerce")
        slip = ((d1.dt.year - d0.dt.year) * 12 + (d1.dt.month - d0.dt.month)).dropna()   # whole months: old snapshots give month only
        slip = slip[slip != 0]
        if len(slip):
            print(f"Primary completion date moved between first and last snapshot for {len(slip):,} trials: "
                  f"median {slip.median():+.0f} months, later by 6+ months for {(slip >= 6).mean():.0%} of them")
        n_snaps = g.size()
        pick = chg[(chg > 0) & (n_snaps == n_snaps.max())]
        ex = pick.index[0] if len(pick) else panel["nct_id"].iloc[0]
        print("\nExample trial:\n" + panel[panel["nct_id"] == ex][
            ["nct_id", "snapshot_date", "overall_status", "phase", "primary_completion_date", "primary_completion_date_type",
             "enrollment", "last_update_submitted_date", "lead_sponsor"]].head(8).to_string(index=False))
    if os.path.exists(COLUMN_LOG):
        cl = pd.read_csv(COLUMN_LOG, dtype=str)
        print("\nColumns missing, or filled from an older name, in some months:\n"
              + cl.groupby(["column", "note"])["month"].agg(["count", "min", "max"]).to_string())
print(f"\nHand-check sample: {os.path.join(CTG_DIR, 'audit_sample.csv')}")


# %% [markdown]
# ## What next
# - **After the dry run:** check Step 6's output. If the four months look right, set `DRY_RUN = False` and *Run all*.
# - **Aliases:** look through Step 3's list of unmatched sponsors, add subsidiaries of firms that file 8-Ks to `ALIASES`, and *Run all*. Months already processed keep the old trial list, so add aliases before the full run if you can.
# - **Hand check:** fill in the `correct` column of `audit_sample.csv` (yes/no). The share marked yes is the match accuracy the paper reports.

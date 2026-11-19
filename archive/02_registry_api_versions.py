"""Archived: first version of Notebook 2, which read version histories from the ClinicalTrials.gov website.

Replaced by pipeline/02_registry_snapshots.py, which uses AACT's monthly snapshots. Not run by the pipeline.
"""


# %% [markdown]
# # Notebook 2 — ClinicalTrials.gov trials and their version histories
#
# Builds the registry half of the pilot for the firms found in Notebook 1. It:
#
# 1. Loads the firm list Notebook 1 saved on Drive.
# 2. Downloads the current record of every industry-sponsored interventional trial on ClinicalTrials.gov.
# 3. Matches each trial sponsor to a firm by name (exact, former name, close spelling or your own aliases) and writes a sample for you to check by hand.
# 4. Downloads each matched trial's **version history**: every date the record changed, its status on that date and which sections changed.
# 5. For Phase 2 and 3 trials, downloads the versions where the status or design changed, to measure delays in completion dates and changes in enrollment.
#
# **How to run**
#
# - **Needs Notebook 1's Steps 1–2 done** (the firm list on Drive). The 8-K download can still be running.
# - Edit `USER_AGENT` in the first code cell.
# - **First run:** leave `DRY_RUN = True` and choose *Runtime → Run all*. Steps 2–3 run in full (about 10–20 minutes); Steps 4–5 fetch only a small sample. Check the last cell's output.
# - **Full run:** set `DRY_RUN = False` and *Run all*. Steps 2–3 are skipped because their files are saved. Steps 4–5 are long; the time estimate prints as they start.
# - **If Colab disconnects:** reconnect and *Run all*. Finished trials are skipped.
#
# ClinicalTrials.gov publishes no fixed request limit, so the notebook starts at `MAX_RPS` and slows down by itself whenever the site pushes back. The version-history pages are the ones the site's own "Record History" tab uses (the same source as the R package `cthist`).
#
# **Output** (in `MyDrive/biopharma_tiv/ctgov/`): current trial records, the sponsor matches, a hand-check sample, and `full/` (or `dryrun/`) with the history and version files.


# %% CONFIG
# ==== Settings: edit USER_AGENT; leave the rest unless you have a reason ====
import os
USER_AGENT = os.environ.get("SEC_USER_AGENT", "Your Name your.email@example.com")
PROJECT_DIR = os.environ.get("WKF_PROJECT_DIR", "/content/drive/MyDrive/biopharma_tiv")
UNIVERSE_FILE = None          # None = the newest universe_*.parquet that Notebook 1 saved

SAMPLE_START = "2005-01-01"   # skip trials whose record has not changed since before this date
INCLUDE_COLLABORATOR_TRIALS = True    # also link trials where the firm is an industry collaborator
HISTORY_FOR_COLLABORATOR_TRIALS = False  # histories only for trials the firm leads
FUZZY_THRESHOLD = 92          # 0-100; how close a spelling must be to count as a match

# Sponsor names on ClinicalTrials.gov that should map to a firm, e.g. a subsidiary.
# Find the firm's CIK in the universe file. Example (check the CIK before using it):
ALIASES = {
    # "Janssen Research & Development, LLC": 200406,   # Johnson & Johnson
}

GET_VERSION_DETAILS = True
DETAIL_PHASES = {"PHASE2", "PHASE3"}   # trials whose changed versions are downloaded in full
VERSION_BUDGET = None                  # cap on version downloads per run (None = no cap)

DRY_RUN   = True    # first run: True (small sample in Steps 4-5). Then set to False.
DRY_RUN_N = 40      # trials in the dry-run sample

MAX_RPS   = 2.0     # starting requests per second; lowered automatically if the site pushes back
N_WORKERS = 2
CHUNK     = 500


# %% [markdown]
# ## Step 0 — Setup


# %% SETUP
import os, re, json, time, glob, random, threading, datetime as dt
from concurrent.futures import ThreadPoolExecutor
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
OUT_DIR   = os.path.join(CTG_DIR, "dryrun" if DRY_RUN else "full")
HIST_DIR  = os.path.join(OUT_DIR, "history_shards")
VER_DIR   = os.path.join(OUT_DIR, "version_shards")
for d in (CTG_DIR, OUT_DIR, HIST_DIR, VER_DIR):
    os.makedirs(d, exist_ok=True)
print("Saving to:", OUT_DIR)


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
                    raise SiteBlocked("ClinicalTrials.gov keeps refusing requests (HTTP 403) from this machine. "
                                      "Wait an hour, or use Runtime > Disconnect and delete runtime for a new machine, then Run all.")
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


# %% [markdown]
# ## Step 4 — Version histories
# For each matched trial, one request returns the list of record versions: the date of each change, the trial's status on that date and which sections changed. Order: firms that file 8-Ks first, then trials the firm leads, small firms before large ones, Phase 3 before Phase 2 before Phase 1. So if the run stops early, the most useful trials are done.


# %% HISTORY
HDONE = os.path.join(OUT_DIR, "history_done.txt")
HFAIL = os.path.join(OUT_DIR, "history_failures.csv")

def phase_rank(p):
    p = p or ""
    return 0 if "PHASE3" in p else 1 if "PHASE2" in p else 2 if "PHASE1" in p else 3

tf = trial_firm.merge(cur[["nct_id", "phases", "last_update_posted"]], on="nct_id")
tf = tf[pd.to_datetime(tf["last_update_posted"], errors="coerce") >= pd.Timestamp(SAMPLE_START)]
n_lead = tf[tf["role"] == "lead"].groupby("cik").size()
tf = tf.assign(firm_n_lead=tf["cik"].map(n_lead).fillna(0), has_8k=tf["cik"].isin(ciks_with_8k),
               role_rank=(tf["role"] != "lead").astype(int), phase_rank=tf["phases"].map(phase_rank))
want = tf if HISTORY_FOR_COLLABORATOR_TRIALS else tf[tf["role"] == "lead"]
order = (want.sort_values(["has_8k", "role_rank", "firm_n_lead", "phase_rank"], ascending=[False, True, True, True])
             .drop_duplicates("nct_id")["nct_id"].tolist())

def read_lines(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {l.strip() for l in f if l.strip()}

def fail_counts(path, key):
    return pd.read_csv(path)[key].value_counts().to_dict() if os.path.exists(path) else {}

def write_parquet_atomic(df, path):
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)

def next_id(folder, prefix):
    ids = [int(re.search(r"(\d+)", os.path.basename(p)).group(1)) for p in glob.glob(os.path.join(folder, prefix + "_*.parquet"))]
    return max(ids, default=0) + 1

def fetch_history(nct):
    try:
        r = CT.get(f"https://clinicaltrials.gov/api/int/studies/{nct}", params={"history": "true"})
        if r.status_code != 200:
            return nct, None, f"HTTP {r.status_code}"
        changes = ((r.json() or {}).get("history") or {}).get("changes") or []
        if not changes:
            return nct, None, "no history in response"
        rows = []
        for c in changes:
            rows.append({"nct_id": nct, "version": c.get("version"), "date": c.get("date"), "status": c.get("status"),
                         "module_labels": ";".join(c.get("moduleLabels") or []),
                         "other": json.dumps({k: v for k, v in c.items() if k not in ("version", "date", "status", "moduleLabels")})})
        return nct, rows, None
    except SiteBlocked:
        raise
    except Exception as e:
        return nct, None, f"{type(e).__name__}: {e}"[:300]

def run_batches(keys, worker, out_dir, prefix, done_path, fail_path, key_name):
    try:
        _run_batches(keys, worker, out_dir, prefix, done_path, fail_path, key_name)
    except SiteBlocked as e:
        print("STOPPED:", e)

def _run_batches(keys, worker, out_dir, prefix, done_path, fail_path, key_name):
    t0, n_ok, n_fail = time.time(), 0, 0
    with ThreadPoolExecutor(N_WORKERS) as ex:
        for start in range(0, len(keys), CHUNK):
            chunk = keys[start:start + CHUNK]
            rows, done_keys, fails = [], [], []
            for key, rws, err in ex.map(worker, chunk):
                if err:
                    fails.append({key_name: key, "error": err,
                                  "when_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
                else:
                    rows.extend(rws)
                    done_keys.append(key)
            sid = next_id(out_dir, prefix)
            if rows:
                write_parquet_atomic(pd.DataFrame(rows), os.path.join(out_dir, f"{prefix}_{sid:05d}.parquet"))
            with open(done_path, "a") as f:
                f.writelines(k + "\n" for k in done_keys)
            if fails:
                pd.DataFrame(fails).to_csv(fail_path, mode="a", header=not os.path.exists(fail_path), index=False)
            n_ok, n_fail = n_ok + len(done_keys), n_fail + len(fails)
            rate = (n_ok + n_fail) / max(time.time() - t0, 1e-9)
            left = len(keys) - start - len(chunk)
            print(f"{prefix} batch {sid:05d}: {n_ok:,} done, {n_fail:,} failed this run | {rate:.2f}/s "
                  f"(limit now {CT.rps:.2f}) | {left:,} left, about {left / max(rate, 1e-9) / 3600:.1f} h")
    print(f"Finished: {n_ok:,} done and {n_fail:,} failed in {(time.time() - t0) / 60:.1f} minutes.")

done, fails = read_lines(HDONE), fail_counts(HFAIL, "nct_id")
todo = [n for n in order if n not in done and fails.get(n, 0) < 3]
if DRY_RUN:
    todo = random.Random(0).sample(todo, min(DRY_RUN_N, len(todo)))
print(f"{len(order):,} trials need histories; {len(done):,} done; {len(todo):,} to fetch now"
      + (" (dry run sample)" if DRY_RUN else "") + f" | about {len(todo) / (0.8 * MAX_RPS) / 3600:.1f} h at full speed")
run_batches(todo, fetch_history, HIST_DIR, "hist", HDONE, HFAIL, "nct_id")


# %% [markdown]
# ## Step 5 — Version details for Phase 2 and 3 trials
# For trials in `DETAIL_PHASES`, downloads the first version, the latest version, and every version where the status or design section changed. From these come the original and updated completion dates, enrollment targets, the sponsor on each date and the reason a trial stopped. Skipped if `GET_VERSION_DETAILS = False`.


# %% VERSIONS
VDONE = os.path.join(OUT_DIR, "version_done.txt")
VFAIL = os.path.join(OUT_DIR, "version_failures.csv")

def load_shards(folder, prefix):
    paths = sorted(glob.glob(os.path.join(folder, prefix + "_*.parquet")))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()

def flatten_version(p):
    st = lambda *k: _g(p, "statusModule", *k)
    return {
        "overall_status": st("overallStatus"), "why_stopped": st("whyStopped"),
        "status_verified": st("statusVerifiedDate"), "last_update_submitted": st("lastUpdateSubmitDate"),
        "start_date": st("startDateStruct", "date"), "start_type": st("startDateStruct", "type"),
        "pcd": st("primaryCompletionDateStruct", "date"), "pcd_type": st("primaryCompletionDateStruct", "type"),
        "completion_date": st("completionDateStruct", "date"), "completion_type": st("completionDateStruct", "type"),
        "enrollment": _g(p, "designModule", "enrollmentInfo", "count"),
        "enrollment_type": _g(p, "designModule", "enrollmentInfo", "type"),
        "phases": "|".join(_g(p, "designModule", "phases", default=[]) or []),
        "lead_sponsor": _g(p, "sponsorCollaboratorsModule", "leadSponsor", "name"),
        "n_primary_outcomes": len(_g(p, "outcomesModule", "primaryOutcomes", default=[]) or []),
        "n_secondary_outcomes": len(_g(p, "outcomesModule", "secondaryOutcomes", default=[]) or []),
        "primary_outcomes": json.dumps([o.get("measure") for o in (_g(p, "outcomesModule", "primaryOutcomes", default=[]) or [])])[:3000],
    }

def fetch_version(key):
    nct, v = key.split("|")
    try:
        r = CT.get(f"https://clinicaltrials.gov/api/int/studies/{nct}/history/{v}")
        if r.status_code != 200:
            return key, None, f"HTTP {r.status_code}"
        p = _g(r.json() or {}, "study", "protocolSection")
        if not p:
            return key, None, "no protocolSection in response"
        return key, [dict(nct_id=nct, version=int(v), **flatten_version(p))], None
    except SiteBlocked:
        raise
    except Exception as e:
        return key, None, f"{type(e).__name__}: {e}"[:300]

if GET_VERSION_DETAILS:
    hist = load_shards(HIST_DIR, "hist")
    if hist.empty:
        print("No histories yet; run Step 4 first.")
    else:
        detail_trials = set(tf.loc[tf["phases"].fillna("").map(lambda s: bool(set(s.split("|")) & DETAIL_PHASES)), "nct_id"])
        h = hist[hist["nct_id"].isin(detail_trials) & hist["version"].notna()].copy()
        last_v = h.groupby("nct_id")["version"].transform("max")
        pick = (h["version"] == 0) | (h["version"] == last_v) | h["module_labels"].fillna("").str.contains("status|design", case=False) \
               | (h["module_labels"].fillna("") == "")
        rank = {n: i for i, n in enumerate(order)}
        h = h[pick].assign(r=h["nct_id"].map(rank)).sort_values(["r", "version"])
        keys = [f"{n}|{int(v)}" for n, v in zip(h["nct_id"], h["version"])]
        vdone, vfails = read_lines(VDONE), fail_counts(VFAIL, "key")
        vtodo = [k for k in keys if k not in vdone and vfails.get(k, 0) < 3]
        if VERSION_BUDGET:
            vtodo = vtodo[:VERSION_BUDGET]
        print(f"{h['nct_id'].nunique():,} Phase {'/'.join(sorted(p[-1] for p in DETAIL_PHASES))} trials; "
              f"{len(keys):,} versions selected; {len(vdone):,} done; {len(vtodo):,} to fetch now | "
              f"about {len(vtodo) / (0.8 * MAX_RPS) / 3600:.1f} h at full speed")
        run_batches(vtodo, fetch_version, VER_DIR, "ver", VDONE, VFAIL, "key")
else:
    print("Version details skipped (GET_VERSION_DETAILS = False).")


# %% [markdown]
# ## Step 6 — Check the results
# Check this cell's output.


# %% SUMMARY
print(f"Run: {'DRY RUN' if DRY_RUN else 'FULL'} | requests made: {CT.n_requests:,} | final limit {CT.rps:.2f}/s")
print(f"Industry trials: {len(cur):,} | linked to firms: {trial_firm['nct_id'].nunique():,} | firms: {trial_firm['cik'].nunique():,}")
hist = load_shards(HIST_DIR, "hist")
if hist.empty:
    print("No histories saved yet.")
else:
    hist = hist.drop_duplicates(["nct_id", "version"]).sort_values(["nct_id", "version"])
    per = hist.groupby("nct_id")
    n_status = per["status"].apply(lambda s: (s != s.shift()).sum() - 1)
    print(f"\nHistories: {hist['nct_id'].nunique():,} trials, {len(hist):,} versions "
          f"(median {per.size().median():.0f} per trial)")
    print(f"Trials with at least one status change: {(n_status > 0).mean():.0%}")
    print("Version dates by year:\n" + pd.to_datetime(hist["date"], errors="coerce").dt.year.value_counts().sort_index().to_string())
    labels = hist["module_labels"].fillna("").str.split(";").explode()
    print("\nSections changed most often:\n" + labels[labels != ""].value_counts().head(10).to_string())
    ex = hist[hist["nct_id"] == per.size().idxmax()][["nct_id", "version", "date", "status", "module_labels"]]
    print("\nExample timeline (the trial with the most versions in this run):\n" + ex.head(12).to_string(index=False))
ver = load_shards(VER_DIR, "ver")
if not ver.empty:
    ver = ver.drop_duplicates(["nct_id", "version"]).sort_values(["nct_id", "version"])
    first = ver.drop_duplicates("nct_id", keep="first").set_index("nct_id")
    last = ver.drop_duplicates("nct_id", keep="last").set_index("nct_id")
    d0 = pd.to_datetime(first["pcd"], errors="coerce")
    d1 = pd.to_datetime(last["pcd"], errors="coerce")
    slip = ((d1 - d0).dt.days / 30.4).dropna()
    print(f"\nVersion details: {len(ver):,} versions for {ver['nct_id'].nunique():,} trials")
    if len(slip):
        print(f"Primary completion date, latest vs first registered: median shift {slip.median():.1f} months; "
              f"later by 6+ months: {(slip >= 6).mean():.0%}; earlier: {(slip < 0).mean():.0%}")
    sp = ver.groupby("nct_id")["lead_sponsor"].nunique()
    print(f"Trials whose lead sponsor changed between versions: {(sp > 1).mean():.1%}")
    print("Sample rows:\n" + ver.head(6)[["nct_id", "version", "overall_status", "pcd", "pcd_type", "enrollment", "lead_sponsor"]].to_string(index=False))
for name in ("history_failures.csv", "version_failures.csv"):
    p = os.path.join(OUT_DIR, name)
    if os.path.exists(p):
        fl = pd.read_csv(p)
        print(f"\n{name}: {len(fl):,} rows\n" + fl["error"].str[:60].value_counts().head(5).to_string())
print(f"\nHand-check sample: {os.path.join(CTG_DIR, 'audit_sample.csv')}")


# %% [markdown]
# ## What next
# - **After the dry run:** check Step 6's output, and look over `sponsor_matches.csv` for big sponsors that should map to a firm (for example a subsidiary); add them to `ALIASES`.
# - **Full run:** set `DRY_RUN = False` and *Run all*. If Colab disconnects, reconnect and *Run all*.
# - **Hand check:** fill in the `correct` column of `audit_sample.csv` (yes/no). The share marked yes is the match precision the paper reports.

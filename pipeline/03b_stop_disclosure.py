"""Notebook 3b — Were trial stops disclosed in any SEC filing?

Pipeline stage of the who-knows-first replication code. Cells are separated by '# %%' markers, so the file
runs top to bottom as a script and opens as a notebook in Jupyter, VS Code or (after tools/make_notebooks.py) Colab.
See README.md for inputs, outputs and the run order.
"""


# %% [markdown]
# # Notebook 3b — Were trial stops disclosed in any SEC filing?
#
# Notebook 3 found that most trial stops at small firms were never named in an 8-K. This notebook checks the other SEC channels, for the same stops:
#
# 1. **8-Ks, rechecked.** All 8-Ks the firm filed from a year before to a year after the stop, plus any 8-K naming the trial's NCT number. A filing counts only if **stop wording** ("terminated", "discontinued", "suspended", "withdrawn", "halted", "paused", "deprioritized", "wind down", ...) is in the **same sentence** as the trial's name, the sentence isn't hypothetical ("may", "could", "if", "risk", "often": risk-factor boilerplate), and the stop word isn't about a contract ("license termination", "terminated the agreement").
# 2. **Other filings, downloaded from EDGAR** for the same windows: 10-Q / 10-K, 20-F / 40-F and 6-K (foreign filers), Reg A reports (1-K, 1-SA, 1-U), and prospectuses (424B, S-1, S-3, F-1, F-3).
# 3. **One row per stop**: the first disclosure in each channel, the earliest overall, and a random **audit sample** of stops no filing discloses, to check by hand against press releases and company websites.
#
# The trial's "name" is its NCT number, its acronym (from the title or the registry, e.g. REST-ON), or its drug names and their other names from the registry (e.g. FT218, or RTB101 = dactolisib).
#
# **v3 (after the hand audit):** a year-long window before the registry change, other drug names from ClinicalTrials.gov, combination and dose labels split off drug names, more stop wording (clinical hold, closed enrollment, will not advance), Reg A filings, and a sample limited to stops the firm owned and could have disclosed (Step 1b). The text around every mention is saved and the rules are applied afterwards, so changing a rule never means downloading again (rerun Steps 4-5 only).
#
# **Scope:** stops at firms leading 1–100 trials (`MAX_FIRM_TRIALS`). Big pharma filings rarely name a single trial, and leaving them out saves hours.
#
# **How to run**
# - Set `USER_AGENT` (same as Notebook 1).
# - **First run:** leave `DRY_RUN = True` (15 firms, a few minutes) and *Run all*. Check the last cell's output.
# - **Full run:** set `DRY_RUN = False` and *Run all* (roughly 1–2 hours, mostly EDGAR downloads; it resumes firm by firm if interrupted, and reuses the dry run's firms).
#
# **Output** (in `MyDrive/biopharma_tiv/features/`): `stop_disclosure.parquet`, `stop_disclosure_hits.parquet` (every match with its snippet), `stop_disclosure_audit.csv`, `figures/fig5_stop_disclosure_all_filings.png`.


# %% CONFIG
# ==== Settings ====
import os
PROJECT_DIR = os.environ.get("WKF_PROJECT_DIR", "/content/drive/MyDrive/biopharma_tiv")
USER_AGENT = os.environ.get("SEC_USER_AGENT", "Your Name your.email@example.com")   # same as Notebook 1: the SEC requires a real name and email
MAX_RPS = 8                    # SEC limit is 10 requests/second
MAX_FIRM_TRIALS = 100          # stops at firms leading 1-100 trials
DAYS_BEFORE, DAYS_AFTER = 365, 365   # v3: a year before too (registry updates often come months after the announcement)
CONTEXT = 500                  # characters kept on each side of a trial name (the disclosure rules are applied to these later)
MAX_NAME_FIRMS = 3             # v3: a drug name counts if at most this many firms list it (v2: 1, which dropped drugs that changed hands)
GET_OTHER_NAMES = True         # v3: download each trial's other drug names and registry acronym from ClinicalTrials.gov
DRY_RUN = True                 # True: only the first DRY_RUN_FIRMS firms
DRY_RUN_FIRMS = 15


# %% [markdown]
# ## Step 0 — Setup


# %% SETUP
import os, re, glob, html, json, time, random, threading
import numpy as np
import pandas as pd
import requests

if PROJECT_DIR.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
assert "@" in USER_AGENT and "Your Name" not in USER_AGENT, "Set USER_AGENT to your real name and email first (first code cell)."

EDGAR_DIR = os.path.join(PROJECT_DIR, "edgar")
SHARD_DIR = os.path.join(EDGAR_DIR, "8k_text", "shards")
CTG_DIR   = os.path.join(PROJECT_DIR, "ctgov")
FEAT_DIR  = os.path.join(PROJECT_DIR, "features")
OUT_DIR   = os.path.join(FEAT_DIR, "stop_disclosure_v3")       # v3: new names and window, so nothing from v2 is reused
for sub in ("firms", "history"):
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
SUFFIX = "_dryrun" if DRY_RUN else ""

class FatalSECBlock(Exception):
    pass

class SECClient:
    """Polite SEC client: declared User-Agent, rate limit, retries with backoff (same as Notebook 1)."""
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
CTG = requests.Session()                                        # ClinicalTrials.gov's official API (other drug names)
CTG.headers.update({"User-Agent": USER_AGENT})
print("Saving to:", OUT_DIR)


# %% [markdown]
# ## Step 1 — Stops in scope, and the names each trial goes by


# %% LOAD
def newest(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None

reg = pd.read_parquet(os.path.join(FEAT_DIR, "registry_events.parquet"))
filings = pd.read_parquet(os.path.join(FEAT_DIR, "filings.parquet"))
links = pd.read_parquet(os.path.join(FEAT_DIR, "filing_trial_links.parquet"))
universe = pd.read_parquet(newest(os.path.join(EDGAR_DIR, "universe_*.parquet")))
universe["cik"] = universe["cik"].astype(int)
cur = pd.read_parquet(newest(os.path.join(CTG_DIR, "industry_trials_current*.parquet")))
trial_firm = pd.read_parquet(os.path.join(CTG_DIR, "trial_firm.parquet"))
trial_firm["cik"] = trial_firm["cik"].astype(int)
cik2name = universe.set_index("cik")["name"].to_dict()
cik2ticker = {c: (str(t).split(";")[0] if str(t or "").strip() else None)
              for c, t in zip(universe["cik"], universe["tickers"].fillna(""))}

stops = reg[reg["is_stop"] & reg["firm_lead_trials"].between(1, MAX_FIRM_TRIALS) & reg["cik"].notna()].copy()
stops["cik"] = stops["cik"].astype(int)
stops["registry_date"] = pd.to_datetime(stops["registry_date"])
stops = stops.sort_values(["cik", "registry_date"]).reset_index(drop=True)
stops["stop_id"] = np.arange(len(stops))
firms = sorted(stops["cik"].unique())
if DRY_RUN:
    firms = firms[:DRY_RUN_FIRMS]
    stops = stops[stops["cik"].isin(firms)].copy()
stops["win_start"] = stops["registry_date"] - pd.Timedelta(days=DAYS_BEFORE)
stops["win_end"] = stops["registry_date"] + pd.Timedelta(days=DAYS_AFTER)
cur1 = cur.drop_duplicates("nct_id").set_index("nct_id")
title = cur1["brief_title"].to_dict() if "brief_title" in cur1 else {}
why = cur1["why_stopped"].to_dict() if "why_stopped" in cur1 else {}
inter = cur1["interventions"]

# ---- the names each trial goes by: NCT number, acronyms, drug names and their other names ----
STOP_NAMES = {"placebo", "saline", "standard of care", "best supportive care", "vehicle", "sham", "observation", "control",
              "usual care", "chemotherapy", "radiotherapy", "radiation", "surgery", "matching placebo", "placebo comparator",
              "background therapy", "standard therapy", "investigator's choice", "physician's choice", "diet", "exercise"}
GENERIC_WORDS = set("""nucleoside nucleotide nucleos analog analogs analogue analogues inhibitor inhibitors agonist antagonist
    cell cells t b nk car stem hematopoietic enriched receptor receptors engineered transgenic autologous allogeneic modified
    vaccine peptide protein antibody antibodies monoclonal hyperthermia superficial thermally sensitive liposomal therapy
    therapies infusion injection tablet tablets capsule capsules oral topical gel ointment cream solution suspension film spray
    patch dose doses dosing drug drugs product products standard treatment care agent agents combination regimen conditioning
    lymphodepleting lymphodepletion chemotherapy biopsy imaging scan blood sample collection procedure questionnaire assessment
    mri pet ct radiation high low mid placebo matching active comparator control to of the for in on a an with and plus or
    extended release delayed immediate er xr sr ir dr iv sc im subcutaneous intravenous intramuscular investigational study""".split())
CODE_NAME = re.compile(r"^([a-z]{1,8})[- ]?(\d{2,6}[a-z]?)$")
ACRO_STOP = {"COVID", "HIV", "FDA", "USA", "AIDS", "CNS", "NASH", "MASH", "COPD", "ALS", "AML", "CLL", "NSCLC", "SCLC", "HCC",
             "RCC", "MDD", "GVHD", "IBD", "PAH", "ARDS", "CKD", "DMD", "SMA", "BLA", "NDA", "OLE", "PART", "STUDY", "TRIAL",
             "PHASE", "ADHD", "PTSD", "CIDP", "CRPC", "MCRPC", "HNSCC", "TNBC", "MDS", "NHL", "DLBCL", "ESRD", "RSV", "HCV",
             "HBV", "SARS", "MERS", "CMV", "EBV", "HPV", "DME", "NMOSD", "FSGS", "IGAN", "ATTR", "PBC", "PSC", "CAR-T",
             "COVID-19", "SARS-COV-2", "EXTENSION", "SAFETY", "EFFICACY", "COHORT", "PILOT", "OPEN", "LABEL"}
SPLIT = r"[;()\[\]/,]+|\s+(?:in\s+combination\s+with|combined\s+with|plus|with|and)\s+|\s*\+\s*"     # v3: "X in combination with Y"
DOSE_TAIL = re.compile(r"(?:\s+\d+(?:\.\d+)?\s*(?:mg|mcg|µg|ug|g|ml|%|iu|units?|mg/kg|mg/m2)(?![a-z]).*"
                       r"|\s+(?:dose|cohort|arm|part|level|group|regimen|schedule)\s*[a-z0-9]{0,3}"          # v3: "vtx958 dose a"
                       r"|\s+(?:oral|iv|sc|topical|er|xr|sr|ir|dr)?\s*(?:tablets?|capsules?|injection|infusion|solution|"
                       r"suspension|ointment|cream|gel|film|spray|patch|powder)(?![a-z]).*)$", re.I)
DOSE_HEAD = re.compile(r"^(?:low|high|mid|medium|standard|full|reduced)[- ]dose\s+", re.I)

def name_variants(raw):
    out = set()
    for piece in re.split(SPLIT, str(raw or ""), flags=re.I):
        s = re.sub(r"^(drug|biological|device|other|procedure|combination product|genetic|dietary supplement)\s*:\s*", "",
                   piece.strip(), flags=re.I)
        s = DOSE_HEAD.sub("", s)
        for _ in range(2):
            s = DOSE_TAIL.sub("", s).strip()
        s = s.lower()
        if len(s) >= 4 and re.match(r"[a-z]", s) and not set(re.split(r"[\s-]+", s)) <= GENERIC_WORDS:   # v3: drop generic words
            out.add(s)
    return out

def ok_acronym(a):
    a = str(a or "").strip()
    return (len(a) >= 4 and bool(re.search(r"[A-Z]{2}", a)) and a.upper() not in ACRO_STOP
            and (len(a) >= 5 or bool(re.search(r"[\d-]", a))))

def acronyms(t):
    found = [w for grp in re.findall(r"\(([^()]*)\)", str(t or ""))
             for w in re.findall(r"(?<![\w-])[A-Z][A-Z0-9-]{2,}[A-Z0-9](?![\w-])", grp)]        # v3: also "(LOSVID STUDY)"
    return sorted({a for a in found if ok_acronym(a)})

# other drug names (e.g. the generic name of a code) and the registry acronym: ClinicalTrials.gov's official API, cached
OTHER_PATH = os.path.join(CTG_DIR, "trial_other_names.parquet")
def fetch_other_names(ncts):
    rows = []
    for i in range(0, len(ncts), 100):
        q = {"filter.ids": ",".join(ncts[i:i + 100]), "fields": "NCTId,Acronym,InterventionName,InterventionOtherName",
             "pageSize": 1000, "format": "json"}
        for attempt in range(6):
            try:
                r = CTG.get("https://clinicaltrials.gov/api/v2/studies", params=q, timeout=90)
            except requests.RequestException:
                r = None
            if r is not None and r.status_code == 200:
                break
            if r is not None and r.status_code in (401, 403):
                raise RuntimeError(f"ClinicalTrials.gov refused the request (HTTP {r.status_code}). Set GET_OTHER_NAMES = False "
                                   "and rerun; don't work around the block.")
            time.sleep(min(60, 2 ** attempt))
        else:
            raise RuntimeError("ClinicalTrials.gov did not answer; rerun later, or set GET_OTHER_NAMES = False.")
        for st in r.json().get("studies", []):
            p = st.get("protocolSection", {}) or {}
            nct = (p.get("identificationModule") or {}).get("nctId")
            rows.append((nct, "fetched", "", ""))
            acr = (p.get("identificationModule") or {}).get("acronym")
            if acr:
                rows.append((nct, "acronym", "", acr))
            for iv in (p.get("armsInterventionsModule") or {}).get("interventions", []) or []:
                for o in iv.get("otherNames", []) or []:
                    rows.append((nct, "other_name", iv.get("name", ""), o))
        time.sleep(0.25)
        if (i // 100) % 20 == 19:
            print(f"  {i + 100:,} of {len(ncts):,} trials")
    return pd.DataFrame(rows, columns=["nct_id", "kind", "main_name", "value"])

tf = trial_firm[trial_firm["cik"].isin(firms)][["nct_id", "cik"]].drop_duplicates()
other = pd.read_parquet(OTHER_PATH) if os.path.exists(OTHER_PATH) else pd.DataFrame(columns=["nct_id", "kind", "main_name", "value"])
if GET_OTHER_NAMES:
    todo = sorted(set(tf["nct_id"]) - set(other.loc[other["kind"] == "fetched", "nct_id"]))
    if todo:
        print(f"Downloading other drug names and acronyms for {len(todo):,} trials from ClinicalTrials.gov")
        other = pd.concat([other, fetch_other_names(todo)], ignore_index=True)
        other.to_parquet(OTHER_PATH, index=False)

# a drug name counts if it is a code (ACM-111) or listed by at most MAX_NAME_FIRMS firms (common drugs are other firms' comparators)
all_names = pd.DataFrame([(c, n, v) for c, n, raw in zip(trial_firm["cik"], trial_firm["nct_id"], trial_firm["nct_id"].map(inter))
                          for v in name_variants(raw)], columns=["cik", "nct_id", "name"])
firms_per_name = all_names.groupby("name")["cik"].nunique()
nfirms = lambda v: int(firms_per_name.get(v, 0))
main = pd.DataFrame([(c, n, v) for c, n, raw in zip(tf["cik"], tf["nct_id"], tf["nct_id"].map(inter)) for v in name_variants(raw)],
                    columns=["cik", "nct_id", "name"])
main_ok = main[(main["name"].map(lambda s: bool(CODE_NAME.match(s)))
                | (main["name"].str.len().ge(6) & main["name"].map(nfirms).le(MAX_NAME_FIRMS))) & ~main["name"].isin(STOP_NAMES)]
oth = other[GET_OTHER_NAMES & other["kind"].eq("other_name")].merge(tf, on="nct_id")
alias = pd.DataFrame([(c, n, v) for c, n, raw, mraw in zip(oth["cik"], oth["nct_id"], oth["value"], oth["main_name"])
                      if min([nfirms(m) for m in name_variants(mraw)] or [0]) <= MAX_NAME_FIRMS      # not another firm's comparator
                      for v in name_variants(raw)
                      if v not in STOP_NAMES and nfirms(v) <= MAX_NAME_FIRMS and (CODE_NAME.match(v) or len(v) >= 5)],
                     columns=["cik", "nct_id", "name"])
reg_acr = (other[GET_OTHER_NAMES & other["kind"].eq("acronym") & other["value"].map(ok_acronym)]
           .groupby("nct_id")["value"].agg(lambda s: sorted({x.strip() for x in s})).to_dict())
acr_of = lambda nct: sorted(set(acronyms(title.get(nct))) | set(reg_acr.get(nct, [])))
names = pd.concat([main_ok, alias], ignore_index=True).drop_duplicates()
# never only the NCT number: a trial with no drug name or acronym keeps its least-shared drug names (up to 10 firms)
bare = {(c, n) for c, n in zip(tf["cik"], tf["nct_id"])} - set(zip(names["cik"], names["nct_id"]))
bare = {k for k in bare if not acr_of(k[1])}
weak = main[np.array([k in bare for k in zip(main["cik"], main["nct_id"])], dtype=bool) & main["name"].str.len().ge(6)
            & main["name"].map(nfirms).le(10) & ~main["name"].isin(STOP_NAMES)]
names = pd.concat([names, weak], ignore_index=True).drop_duplicates()
n_name_trials = names.groupby(["cik", "name"])["nct_id"].nunique().to_dict()
names_by_trial = names.groupby(["cik", "nct_id"])["name"].agg(lambda s: sorted(set(s))).to_dict()

def identifiers(cik, nct):
    """(kind, n_trials, regex) in the order they are tried: NCT number, acronyms, drug code names and other drug names."""
    out = [("nct", 1, re.compile(rf"\bNCT\s?{nct[3:]}\b", re.I))]
    out += [("acronym", 1, re.compile(rf"(?<![\w-]){re.escape(a)}(?![\w-])")) for a in acr_of(nct)]
    for n in names_by_trial.get((cik, nct), []):
        m = CODE_NAME.match(n)
        rx = (rf"(?<![\w-]){m.group(1)}[\s-]?{m.group(2)}(?![\w-])" if m else rf"(?<![\w-]){re.escape(n)}(?![\w-])")
        out.append(("drug_name", n_name_trials.get((cik, n), 1), re.compile(rx, re.I)))
    return out

IDENT = {sid: identifiers(c, n) for sid, c, n in zip(stops["stop_id"], stops["cik"], stops["nct_id"])}
NAMES = {sid: acr_of(n) + names_by_trial.get((c, n), []) for sid, c, n in zip(stops["stop_id"], stops["cik"], stops["nct_id"])}
STOP_WORDS = re.compile(
    r"\b(?:terminat(?:e|ed|es|ing|ion)|discontinu(?:e|ed|es|ing|ation)|suspend(?:ed|s|ing)?|"
    r"withdr(?:ew|awn|aw|awal)|halt(?:ed|s|ing)?|stopp(?:ed|ing)|paus(?:ed|ing|e)|"
    r"stops?\s+(?:all\s+|the\s+|further\s+)?(?:development|dosing|enrol?ment|trial|study)|"
    r"deprioriti[sz](?:e|ed|ing|ation)|wind(?:ing)?\s+down|wound\s+down|ceas(?:e|ed|es|ing)|abandon(?:ed|s|ing)?|"
    r"supplant(?:ed|s|ing)?|(?:partial\s+|full\s+)?clinical\s+hold|placed\s+on\s+(?:a\s+)?hold|"
    r"clos(?:e|ed|ing|ure\s+of)\s+(?:\S+\s+){0,6}?(?:trial|study|studies|enrollment|enrolment|program|programme)|"
    r"no\s+longer\s+(?:pursu|develop|enrol|advanc|plan)\w*|decided\s+not\s+to|not\s+to\s+(?:advance|proceed|continue|pursue)|"
    r"will\s+not\s+(?:advance|proceed|continue|pursue|initiate)|"
    r"not\s+(?:plan(?:ning)?|intend(?:ing)?|anticipate)\s+(?:to\s+)?(?:conduct|continu|pursu|advanc|develop|initiat)\w*|"
    r"no\s+(?:further|additional)\s+(?:development|clinical|studies|trials|investment))", re.I)    # v3: holds, closures, "will not advance"

def squash(s):
    return re.sub(r"\s+", " ", s or "").strip()

def scan(text, sids, per_kind=10):
    """Every place a stop's trial is named (up to per_kind per kind of name), with CONTEXT characters on each side."""
    rows = []
    for sid in sids:
        for kind, n_tr, rx in IDENT[sid]:
            for k, m in enumerate(rx.finditer(text)):
                if k >= per_kind:
                    break
                s0 = max(0, m.start() - CONTEXT)
                rows.append({"stop_id": sid, "match_kind": kind, "name_trials": n_tr, "context": text[s0:m.end() + CONTEXT],
                             "pos": m.start() - s0, "mlen": m.end() - m.start()})
    return rows

# ---- when does a mention disclose the stop? stop wording in the same sentence, not hypothetical, not a contract ----
GENERIC = re.compile(r"\bmay\b|(?i:\b(?:might|could|can|cannot|would|should|if|unless|whether|risks?|often|potential(?:ly)?|"
                     r"possib(?:le|ly)|such\s+as|for\s+example)\b)")        # lowercase "may" only: "in May" is a month
CONTRACT = re.compile(r"\b(?:licen[cs]e|agreement|collaboration|contract|employment|lease|merger|partnership|arrangement)\b", re.I)
LIFTED = re.compile(r"\b(?:lift(?:s|ed|ing)?|remov(?:e|ed|al)|resolv(?:e|ed|ing)|releas(?:e|ed))\b", re.I)

def sentence(x, a, b, cap=250):
    """The sentence (or semicolon clause) around x[a:b], at most cap characters on each side."""
    s = max(x.rfind(". ", 0, a), x.rfind("\n", 0, a), x.rfind("; ", 0, a)) + 1
    ends = [i for i in (x.find(". ", b), x.find("\n", b), x.find("; ", b)) if i != -1]
    e = min(ends) if ends else len(x)
    return x[max(s, a - cap):min(e, b + cap)]

def discloses(context, pos, mlen):
    sent = sentence(context, pos, pos + mlen)
    if GENERIC.search(sent):
        return False
    for m in STOP_WORDS.finditer(sent):
        if CONTRACT.search(sent[max(0, m.start() - 50):m.end() + 50]):
            continue
        if "hold" in m.group(0).lower() and LIFTED.search(sent):          # "the FDA lifted the clinical hold"
            continue
        return True
    return False

def stops_in_window(cik, date, by_firm):
    s = by_firm.get(cik)
    if s is None:
        return []
    return s.loc[(s["win_start"] <= date) & (s["win_end"] >= date), "stop_id"].tolist()

stops_by_firm = {c: g for c, g in stops.groupby("cik")}
kinds = pd.Series([{k for k, _, _ in IDENT[s]} for s in stops["stop_id"]])
print(f"Stops in scope: {len(stops):,} at {len(firms):,} firms" + ("  (dry run)" if DRY_RUN else ""))
print(f"With an acronym: {kinds.map(lambda k: 'acronym' in k).mean():.0%} | with a drug name: "
      f"{kinds.map(lambda k: 'drug_name' in k).mean():.0%} | NCT number only: {kinds.map(lambda k: k == {'nct'}).mean():.0%} | "
      f"other names downloaded: {len(alias):,} usable, for {alias['nct_id'].nunique():,} trials")
nct_only = stops[kinds.map(lambda k: k == {"nct"}).to_numpy()]
if len(nct_only):
    print("Examples with the NCT number only (interventions):")
    for n in nct_only["nct_id"].head(5):
        print(f"  {n}: {str(inter.get(n))[:120]}")


# %% [markdown]
# ## Step 1b — Firm filing histories, and which stops belong in the sample
# Downloads each firm's filing list once (saved per firm; Step 3 reuses it). A stop stays in the sample if the firm is the trial's lead sponsor, its name was matched exactly (or by alias, or by a former name in use at the time), the firm was filing SEC reports around the stop, and the stop is not an administrative close (e.g. patients moved to a follow-up study). Left-out stops are kept in the output with `exclude_reason`.


# %% FIRMS
REPORTS = {"10-K", "10-Q", "10-KT", "10-QT", "20-F", "40-F", "8-K", "6-K", "1-K", "1-SA", "1-U"}

def firm_history(cik):
    """Every filing of the firm and its former names with dates, from data.sec.gov/submissions (saved per firm)."""
    fp, fj = (os.path.join(OUT_DIR, "history", f"{cik}{e}") for e in (".parquet", ".json"))
    if os.path.exists(fp) and os.path.exists(fj):
        with open(fj) as f:
            return pd.read_parquet(fp), json.load(f)
    cols = ["form", "filingDate", "accessionNumber", "primaryDocument"]
    r = SEC.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    j = r.json() if r.status_code == 200 else {}
    pages = [j.get("filings", {}).get("recent", {})]
    for f in j.get("filings", {}).get("files", []):
        r2 = SEC.get(f"https://data.sec.gov/submissions/{f['name']}")
        if r2.status_code == 200:
            pages.append(r2.json())
    fl = pd.concat([pd.DataFrame({c: p.get(c, []) for c in cols}) for p in pages], ignore_index=True)
    fl["filingDate"] = pd.to_datetime(fl["filingDate"], errors="coerce")
    former = j.get("formerNames", []) or []
    fl.to_parquet(fp, index=False)
    with open(fj, "w") as f:
        json.dump(former, f)
    return fl, former

t0, hist = time.time(), {}
for k, c in enumerate(firms, 1):
    hist[c] = firm_history(c)
    if k % 100 == 0 or k == len(firms):
        print(f"  filing histories: {k}/{len(firms)} firms | {time.time() - t0:.0f}s")

# ---- which stops belong in the sample ----
SUFFIXES = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "llc", "lp", "llp", "plc", "sa",
            "ag", "nv", "bv", "se", "gmbh", "kk", "spa", "ab", "as", "asa", "oyj", "holdings", "holding", "group", "and", "the",
            "usa", "us"}
CANON = {"pharmaceuticals": "pharma", "pharmaceutical": "pharma", "pharm": "pharma", "therapeutic": "therapeutics",
         "biosciences": "bioscience", "technologies": "technology", "laboratories": "labs", "laboratory": "labs",
         "international": "intl", "biopharmaceuticals": "biopharma", "biopharmaceutical": "biopharma"}
def norm_name(s):                                                      # same as Notebooks 2 and 3
    s = str(s or "").lower()
    s = re.sub(r"/[a-z]{1,4}/", " ", s).replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = [CANON.get(t, t) for t in s.split()]
    while toks and toks[-1] in SUFFIXES:
        toks.pop()
    while toks and toks[0] == "the":
        toks.pop(0)
    return " ".join(toks)

def _date(x):
    t = pd.to_datetime(x, errors="coerce", utc=True)
    return pd.NaT if pd.isna(t) else t.tz_localize(None)

sm = pd.read_csv(os.path.join(CTG_DIR, "sponsor_matches.csv"))
sm = sm[sm["cik"].notna()]
sm_method = {(s, int(c)): str(m).split(":")[0] for s, c, m in zip(sm["sponsor"], sm["cik"], sm["method"])}
cur_lead = cur1["lead_sponsor"].to_dict() if "lead_sponsor" in cur1 else {}
if "owner_source" not in stops:
    stops["owner_source"] = "sponsor_at_time"
if "lead_sponsor" not in stops:
    stops["lead_sponsor"] = stops["nct_id"].map(cur_lead)

def owner_check(r, former):
    """How the trial's sponsor was matched to this firm: alias / exact / exact_former (name in use then) or a problem."""
    if r.owner_source == "collaborator":
        return "collaborator"
    sponsor = r.lead_sponsor if r.owner_source == "sponsor_at_time" else cur_lead.get(r.nct_id)
    n = norm_name(sponsor)
    m = sm_method.get((sponsor, int(r.cik)))
    if m in (None, "nan"):
        m = ("exact" if n == norm_name(cik2name.get(int(r.cik))) else
             "exact_former" if n in {norm_name(f.get("name")) for f in former} else "fuzzy")
    if m == "exact_former":
        spans = [(_date(f.get("from")), _date(f.get("to"))) for f in former if norm_name(f.get("name")) == n]
        if spans and not any((pd.isna(lo) or r.registry_date >= lo - pd.Timedelta(days=365))
                             and (pd.isna(hi) or r.registry_date <= hi + pd.Timedelta(days=730)) for lo, hi in spans):
            return "former_out_of_date"
    return m

def reporting_at(fl, d):
    """Filed a 10-K/10-Q/20-F/40-F/8-K/6-K or Reg A report from 400 days before to 30 days after the date."""
    x = fl.loc[fl["form"].isin(REPORTS), "filingDate"]
    return bool(((x >= d - pd.Timedelta(days=400)) & (x <= d + pd.Timedelta(days=30))).any())

ADMIN = re.compile(r"(?i)enrol?ment\s+(?:\w+\s+){0,3}?(?:was\s+|has\s+been\s+)?completed|completed\s+(?:the\s+)?(?:phase|part|"
                   r"dose[- ]escalation|enrol?ment)|completion\s+of\s+the\s+dose[- ]escalation|followed\s+up\s+in|"
                   r"long[- ]term\s+follow|roll[- ]?over|crossover\s+arm|transferred\s+to|moved\s+to\s+(?:a|another|the)\s+")
NEGATIVE = re.compile(r"(?i)terminat|discontinu|halt|futil|safety|efficacy|business|financ|fund|strateg|corporate|commercial|"
                      r"priorit|sponsor|toxic|death|adverse")
stops["owner_method"] = [owner_check(r, hist[r.cik][1]) for r in stops.itertuples()]
stops["reporting_at_stop"] = [reporting_at(hist[r.cik][0], r.registry_date) for r in stops.itertuples()]
w = stops["nct_id"].map(why).fillna("").astype(str)
stops["administrative"] = w.str.contains(ADMIN) & ~w.str.contains(NEGATIVE)
stops["exclude_reason"] = np.select(
    [stops["owner_method"].eq("collaborator"), stops["owner_method"].eq("fuzzy"), stops["owner_method"].eq("former_out_of_date"),
     ~stops["reporting_at_stop"], stops["administrative"]],
    ["collaborator only", "sponsor matched by close spelling", "former name not in use then",
     "firm not filing reports then (private, acquired or deregistered)", "administrative close"], default="")
stops["in_sample"] = stops["exclude_reason"].eq("")
stops_by_firm = {c: g for c, g in stops.groupby("cik")}
print(f"In the sample: {stops['in_sample'].sum():,} of {len(stops):,} stops")
print(stops["exclude_reason"].replace("", "kept").value_counts().to_string())
fz = stops[stops["owner_method"].eq("fuzzy")].drop_duplicates("lead_sponsor").head(10)
if len(fz):
    print("\nClose-spelling matches left out (add correct ones to ALIASES in Notebook 2):")
    for r in fz.itertuples():
        print(f"  {r.lead_sponsor!r} -> {cik2name.get(r.cik)!r}")


# %% [markdown]
# ## Step 2 — Recheck the 8-Ks for stop wording
# Reads Notebook 1's text shards: every 8-K the firm filed from a year before to a year after each stop, plus any 8-K that names the trial's NCT number.


# %% CHECK_8K
COLS = ["stop_id", "match_kind", "name_trials", "context", "pos", "mlen", "accession", "form", "filing_date", "channel"]
HITS_8K = os.path.join(OUT_DIR, f"hits_8k{SUFFIX}.parquet")
if os.path.exists(HITS_8K):
    hits_8k = pd.read_parquet(HITS_8K)
    print(f"Loaded earlier 8-K check: {len(hits_8k):,} rows")
else:
    f8 = filings[filings["cik"].isin(firms)][["accession", "cik", "event_date"]].copy()
    want = {}                                                     # accession -> stops it could disclose
    for acc, c, d in zip(f8["accession"], f8["cik"], f8["event_date"]):
        sids = stops_in_window(c, d, stops_by_firm)
        if sids:
            want[acc] = set(sids)
    nct_links = links[(links["link_type"] == "nct") & links["nct_id"].isin(set(stops["nct_id"]))]
    nct_links = nct_links.merge(filings[["accession", "event_date"]], on="accession")
    for acc, nct, d in zip(nct_links["accession"], nct_links["nct_id"], nct_links["event_date"]):
        s = stops[(stops["nct_id"] == nct) & (stops["win_start"] <= d) & (stops["win_end"] >= d)]["stop_id"]
        if len(s):
            want.setdefault(acc, set()).update(s)
    acc_date = filings.set_index("accession")["event_date"].to_dict()
    rows, t0 = [], time.time()
    shard_paths = sorted(glob.glob(os.path.join(SHARD_DIR, "docs_*.parquet")))
    for k, p in enumerate(shard_paths, 1):
        accs = pd.read_parquet(p, columns=["accession"])["accession"]
        need = set(accs) & set(want)
        if need:
            docs = pd.read_parquet(p, columns=["accession", "text"])
            docs = docs[docs["accession"].isin(need)]
            for acc, g in docs.groupby("accession"):
                for r in scan("\n".join(g["text"].fillna("")), sorted(want[acc])):
                    rows.append({**r, "accession": acc, "form": "8-K", "filing_date": acc_date[acc], "channel": "8-K"})
                rows.append({"stop_id": -1, "accession": acc, "form": "8-K", "filing_date": acc_date[acc], "channel": "8-K"})
        if k % 25 == 0 or k == len(shard_paths):
            print(f"  shard {k}/{len(shard_paths)} | {time.time() - t0:.0f}s")
    hits_8k = pd.DataFrame(rows, columns=COLS)
    hits_8k.to_parquet(HITS_8K, index=False)
    print(f"8-Ks read: {(hits_8k['stop_id'] < 0).sum():,} | naming a stopped trial: "
          f"{hits_8k.loc[hits_8k['stop_id'] >= 0, 'accession'].nunique():,}")


# %% [markdown]
# ## Step 3 — Other filings from EDGAR
# For each firm: every 10-Q / 10-K / 20-F / 40-F / 6-K / Reg A report / prospectus filed inside one of its stops' windows (filing list from Step 1b). Only the matches are kept. Saved firm by firm, so a rerun skips firms already done.


# %% EDGAR
PERIODIC = {"10-Q", "10-K", "10-QT", "10-KT", "20-F", "40-F"}
CURRENT_FOREIGN = {"6-K"}
REG_A = {"1-K", "1-SA", "1-U"}         # v3: Regulation A issuers file these instead of 10-K / 10-Q / 8-K
PROSPECTUS = {"424B1", "424B2", "424B3", "424B4", "424B5", "424B7", "S-1", "S-1/A", "S-3", "F-1", "F-1/A", "F-3"}
FULL_TXT = {"6-K", "40-F", "1-U"}      # the substance is in exhibits (press release, annual information form)
SKIP_DOCS = ("GRAPHIC", "ZIP", "EXCEL", "EX-101", "XML", "JSON", "PDF")
MAX_CHARS = 3_000_000

def channel_of(form):
    return ("10-Q/10-K/20-F/40-F" if form in PERIODIC else "6-K" if form in CURRENT_FOREIGN else "Reg A" if form in REG_A
            else "prospectus")

def to_text(s):
    s = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", s)
    s = re.sub(r"(?i)<br\s*/?>|</(?:p|div|tr|li|h[1-6]|td)>", "\n", s)
    s = html.unescape(re.sub(r"<[^>]+>", " ", s))
    return re.sub(r"[ \t\r\f\v\xa0]+", " ", s)

def full_submission_text(raw):
    parts = []
    for m in re.finditer(r"(?is)<DOCUMENT>(.*?)</DOCUMENT>", raw):
        d = m.group(1)
        typ = re.search(r"<TYPE>([^\n<]+)", d)
        typ = typ.group(1).strip().upper() if typ else ""
        if typ.startswith(SKIP_DOCS) or "<PDF>" in d[:3000].upper():
            continue
        parts.append(to_text(d))
    return "\n".join(parts)

def do_firm(cik):
    s = stops_by_firm[cik]
    fl = hist[cik][0]                                               # filing list from Step 1b
    fl = fl[fl["form"].isin(PERIODIC | CURRENT_FOREIGN | REG_A | PROSPECTUS)].drop_duplicates("accessionNumber")
    rows = []
    for form, d, acc, doc in zip(fl["form"], fl["filingDate"], fl["accessionNumber"], fl["primaryDocument"]):
        sids = stops_in_window(cik, d, stops_by_firm)
        if not sids:
            continue
        base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/"
        r = SEC.get(base + (f"{acc}.txt" if form in FULL_TXT or not doc else doc))
        if r.status_code != 200:
            continue
        raw = r.content.decode("utf-8", "ignore")[:MAX_CHARS * 3]
        text = (full_submission_text(raw) if form in FULL_TXT or not doc else to_text(raw))[:MAX_CHARS]
        for h in scan(text, sids):
            rows.append({**h, "accession": acc, "form": form, "filing_date": d, "channel": channel_of(form)})
        rows.append({"stop_id": -1, "accession": acc, "form": form, "filing_date": d, "channel": channel_of(form)})  # read
    return pd.DataFrame(rows, columns=COLS)

t0, todo = time.time(), [c for c in firms if not os.path.exists(os.path.join(OUT_DIR, "firms", f"{c}.parquet"))]
print(f"Firms to download: {len(todo):,} (already done: {len(firms) - len(todo):,})")
for k, c in enumerate(todo, 1):
    out = do_firm(c)
    out.to_parquet(os.path.join(OUT_DIR, "firms", f"{c}.parquet.tmp"), index=False)
    os.replace(os.path.join(OUT_DIR, "firms", f"{c}.parquet.tmp"), os.path.join(OUT_DIR, "firms", f"{c}.parquet"))
    if k % 10 == 0 or k == len(todo):
        el = time.time() - t0
        print(f"  firm {k}/{len(todo)} | {SEC.n_requests:,} requests | about {el / k * (len(todo) - k) / 60:.0f} min left")
hits_edgar = pd.concat([pd.read_parquet(os.path.join(OUT_DIR, "firms", f"{c}.parquet")).assign(cik=c) for c in firms],
                       ignore_index=True)
read = pd.concat([hits_8k[hits_8k["stop_id"] < 0].assign(cik=hits_8k["accession"].map(filings.set_index("accession")["cik"])),
                  hits_edgar[hits_edgar["stop_id"] < 0]], ignore_index=True)
hits_edgar = hits_edgar[hits_edgar["stop_id"] >= 0].drop(columns=["cik"])
print(f"EDGAR filings read: {len(read):,} ({read['form'].value_counts().head(8).to_dict()})")


# %% [markdown]
# ## Step 4 — One row per stop, and the audit sample
# For each stop and channel: days from the registry date to the first filing that names the trial near stop wording (negative = before the registry). `first_channel` is the earliest; `timing` is `before`, `0-30 days`, `31-365 days` or `none` (`timing_detail` splits `before` at 30 days; `days_first_v2window` is v2's measure). The audit sample (`stop_disclosure_audit_v3.csv`) is 100 random in-sample stops at firms leading 1–20 trials that no filing discloses.


# %% COMBINE
hits = pd.concat([hits_8k[hits_8k["stop_id"] >= 0], hits_edgar], ignore_index=True)
ctx = list(zip(hits["context"], hits["pos"].astype(int), hits["mlen"].astype(int)))
hits["stop_hit"] = [discloses(c, p, n) for c, p, n in ctx]
hits["snippet"] = [squash(sentence(c, p, p + n)) for c, p, n in ctx]
hits["matched"] = [c[p:p + n] for c, p, n in ctx]
# v3.1: a drug-name match doesn't count if the sentence names another of the firm's trials instead
# ("the discontinuation of our bavituximab Phase III SUNRISE trial" says nothing about a different bavituximab trial)
firm_trials = tf.groupby("cik")["nct_id"].agg(list).to_dict()
sid_cik, sid_nct = dict(zip(stops["stop_id"], stops["cik"])), dict(zip(stops["stop_id"], stops["nct_id"]))
_other = {}
def other_trial_rx(sid):
    c, n = sid_cik[sid], sid_nct[sid]
    if (c, n) not in _other:
        own = set(acr_of(n))
        alts = {re.escape(a) for t in firm_trials.get(c, []) if t != n for a in acr_of(t) if a not in own}
        alts |= {rf"NCT\s?{t[3:]}" for t in firm_trials.get(c, []) if t != n}
        _other[(c, n)] = re.compile(r"(?<![\w-])(?:" + "|".join(sorted(alts)) + r")(?![\w-])") if alts else None
    return _other[(c, n)]
hits["other_trial"] = [k == "drug_name" and other_trial_rx(s) is not None and bool(other_trial_rx(s).search(sn))
                       for k, s, sn in zip(hits["match_kind"], hits["stop_id"], hits["snippet"])]
hits["stop_hit"] = hits["stop_hit"] & ~hits["other_trial"]
hits = hits.merge(stops[["stop_id", "registry_date"]], on="stop_id")
hits["days"] = (pd.to_datetime(hits["filing_date"]) - hits["registry_date"]).dt.days
hits.drop(columns=["registry_date"]).to_parquet(os.path.join(FEAT_DIR, f"stop_disclosure_hits{SUFFIX}.parquet"), index=False)

disc = hits[hits["stop_hit"].astype(bool)]
CHANNELS = ["8-K", "10-Q/10-K/20-F/40-F", "6-K", "Reg A", "prospectus"]
dcols = ["days_" + re.sub(r"\W+", "_", c).strip("_").lower() for c in CHANNELS]
first = disc.groupby(["stop_id", "channel"])["days"].min().unstack().reindex(columns=CHANNELS) if len(disc) else \
    pd.DataFrame(columns=CHANNELS, dtype=float)
first.columns = dcols
sd = stops.set_index("stop_id").join(first)
sd["days_first"] = sd[dcols].min(axis=1)
has = sd["days_first"].notna()
sd["first_channel"] = "none"
sd.loc[has, "first_channel"] = sd.loc[has, dcols].idxmin(axis=1).map(dict(zip(dcols, CHANNELS)))
sd["timing"] = np.select([sd["days_first"] < 0, sd["days_first"] <= 30, sd["days_first"].notna()],
                         ["before", "0-30 days", "31-365 days"], "none")
sd["timing_detail"] = np.select([sd["days_first"] < -30, sd["days_first"] < 0, sd["days_first"] <= 30, sd["days_first"].notna()],
                                ["31-365 days before", "1-30 days before", "0-30 days after", "31-365 days after"], "none")
sd["days_first_v2window"] = disc[disc["days"] >= -30].groupby("stop_id")["days"].min().reindex(sd.index)   # v2's window, for comparison
sd["named_any"] = sd.index.isin(hits["stop_id"])
sd["group"] = np.where(sd["firm_8ks_year"] == 0, "files no 8-Ks (foreign / not yet public)",
                       np.where(sd["firm_lead_trials"] <= 20, "8-K filer, 1-20 trials", "8-K filer, 21-100 trials"))
sd["firm"], sd["ticker"] = sd["cik"].map(cik2name), sd["cik"].map(cik2ticker)
sd["brief_title"], sd["why_stopped"] = sd["nct_id"].map(title), sd["nct_id"].map(why)
sd["names_searched"] = sd.index.map(lambda s: ", ".join(NAMES.get(s, [])))
sd = sd.reset_index()
sd.drop(columns=["win_start", "win_end"]).to_parquet(os.path.join(FEAT_DIR, f"stop_disclosure{SUFFIX}.parquet"), index=False)

pool = sd[sd["in_sample"] & (sd["timing"] == "none") & (sd["firm_lead_trials"] <= 20)]
audit = pool.sample(min(100, len(pool)), random_state=0)[["nct_id", "firm", "ticker", "registry_date", "detail", "brief_title",
                                                           "why_stopped", "named_any", "names_searched"]].copy()
audit.insert(1, "ctgov_url", "https://clinicaltrials.gov/study/" + audit["nct_id"])
for c in ["announced_elsewhere (Y/N)", "where (press release / website / call / other)", "date", "url", "notes"]:
    audit[c] = ""
audit.to_csv(os.path.join(FEAT_DIR, f"stop_disclosure_audit_v3{SUFFIX}.csv"), index=False)
s = sd[sd["in_sample"]]
print(f"Stops: {len(sd):,} | in the sample: {len(s):,} | disclosed in some filing: {(s['timing'] != 'none').mean():.0%} "
      f"(v2 window: {s['days_first_v2window'].notna().mean():.0%}) | audit sample: {len(audit)}")


# %% [markdown]
# ## Step 5 — Check the results
# Check this cell's output.


# %% SUMMARY
print(f"Run: {'DRY RUN' if DRY_RUN else 'FULL'} | stops {len(sd):,} | in the sample {sd['in_sample'].sum():,} | "
      f"firms {len(firms):,} | EDGAR requests {SEC.n_requests:,}")
print("\nLeft out of the sample:\n" + sd["exclude_reason"].replace("", "kept").value_counts().to_string())
s = sd[sd["in_sample"]]
order = ["before", "0-30 days", "31-365 days", "none"]
print("\nWhen a filing first names the trial near stop wording (share of stops in the sample):")
print(pd.crosstab(s["group"], s["timing"], normalize="index").reindex(columns=order, fill_value=0).round(2).to_string())
print(pd.crosstab(s["group"], s["timing"], margins=True).reindex(columns=order + ["All"], fill_value=0).to_string())
detail = ["31-365 days before", "1-30 days before", "0-30 days after", "31-365 days after", "none"]
print("\nIn detail:")
print(pd.crosstab(s["group"], s["timing_detail"], normalize="index").reindex(columns=detail, fill_value=0).round(2).to_string())
print("\nDisclosed in some filing, v2 window (30 days before) vs v3 (365 days before):")
print(pd.DataFrame({"v2_window": s.groupby("group")["days_first_v2window"].apply(lambda x: x.notna().mean()),
                    "v3_window": s.groupby("group")["days_first"].apply(lambda x: x.notna().mean())}).round(2).to_string())
print("\nFirst channel (disclosed stops):")
print(pd.crosstab(s["group"], s["first_channel"]).drop(columns=["none"], errors="ignore").to_string())
print("\nShare of stops disclosed in each channel at any point in the window:")
print(s.groupby("group")[dcols].agg(lambda x: x.notna().mean()).round(2).rename(columns=dict(zip(dcols, CHANNELS))).to_string())
old = s["disclosed_before"] | s["days_to_8k"].notna()
rd = read.drop_duplicates("accession").merge(s[["cik", "group"]].drop_duplicates("cik"), on="cik")
print("\nFilings read per firm, by group:", (rd.groupby("group").size() / s.groupby("group")["cik"].nunique()).round(1).to_dict())
print("Share of stops whose trial is named in any filing:", s.groupby("group")["named_any"].mean().round(2).to_dict())
print(f"\nNotebook 3's measure (any 8-K naming the trial): {old.mean():.0%} | 8-Ks with stop wording: "
      f"{s['days_8_k'].notna().mean():.0%} | any filing with stop wording: {(s['timing'] != 'none').mean():.0%}")
ds = disc[disc["stop_id"].isin(s["stop_id"])]
print("Matched by:", ds["match_kind"].value_counts().to_dict(), "| via a drug name shared by 6+ trials:",
      int((ds["name_trials"] > 5).sum()))
print("Names behind the most disclosures (check none is a generic word):",
      ds.drop_duplicates(["stop_id", "matched"])["matched"].str.lower().value_counts().head(15).to_dict())
print("Stop wording that did it:", ds["snippet"].str.extract("(" + STOP_WORDS.pattern + ")", flags=re.I)[0].str.lower()
      .str.replace(r"\s+", " ", regex=True).value_counts().head(15).to_dict())
print(f"Not disclosed, but the trial is named in some filing without stop wording: "
      f"{(s['named_any'] & (s['timing'] == 'none')).mean():.0%} of stops")
print("\nSample matches (check these read as a stop):")
for x in ds.sample(min(12, len(ds)), random_state=1).itertuples():
    print(f"  [{x.form} | {x.match_kind} | {int(x.days):+d} days] {x.snippet[:400]}")

# ---- figure: disclosure across all SEC filings ----
import matplotlib.pyplot as plt
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "legend.frameon": False, "axes.axisbelow": True, "ytick.major.size": 0})
labels = {"before": "Filing first (up to a year before)", "0-30 days": "Filing 0-30 days after",
          "31-365 days": "Filing 31-365 days after", "none": "No filing discloses it"}
share = pd.crosstab(s["group"], s["timing"], normalize="index").reindex(columns=order, fill_value=0) * 100
n = s["group"].value_counts().reindex(share.index)
fig, ax = plt.subplots(figsize=(8, 3.2))
left, y = np.zeros(len(share)), np.arange(len(share))[::-1]
for col, c in zip(order, ["#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"]):
    ax.barh(y, share[col], left=left, height=0.55, color=c, edgecolor=SURF, linewidth=1.5, label=labels[col])
    for yi, l, v in zip(y, left, share[col]):
        if v >= 6:
            ax.text(l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", color="white" if c in ("#2a78d6", "#eb6834") else INK, fontsize=9)
    left += share[col].to_numpy()
ax.set_yticks(y, [f"{g}\n(n = {k:,})" for g, k in zip(share.index, n)])
ax.set_xlim(0, 100); ax.set_xlabel("Share of stops (%)"); ax.grid(axis="y", visible=False)
ax.set_title("Is a trial stop disclosed in any SEC filing?", loc="left", pad=22)
ax.text(0, 1.03, "8-K, 10-Q/10-K, 20-F/40-F, 6-K, Reg A or prospectus naming the trial near stop wording",
        transform=ax.transAxes, color=INK2, fontsize=9)
ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0, -0.28), fontsize=9)
os.makedirs(os.path.join(FEAT_DIR, "figures"), exist_ok=True)
fig.savefig(os.path.join(FEAT_DIR, "figures", f"fig5_stop_disclosure_all_filings{SUFFIX}.png"), dpi=200, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## What next
# - **After the dry run:** check Step 5's output, then set `DRY_RUN = False` and *Run all*.
# - **Audit:** fill in `stop_disclosure_audit.csv` (about a day). It gives the share of "undisclosed" stops that were announced outside SEC filings.
# - **Notebook 4** can start in parallel: prices and volatility around readouts don't depend on this notebook.

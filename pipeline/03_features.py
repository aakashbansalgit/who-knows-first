"""Notebook 3 — Features: 8-K timing and content, trial readouts, registry changes

Pipeline stage of the who-knows-first replication code. Cells are separated by '# %%' markers, so the file
runs top to bottom as a script and opens as a notebook in Jupyter, VS Code or (after tools/make_notebooks.py) Colab.
See README.md for inputs, outputs and the run order.
"""


# %% [markdown]
# # Notebook 3 — Features: 8-K timing and content, trial readouts, registry changes
#
# Turns what Notebooks 1 and 2 downloaded into the tables the tests run on.
#
# 1. **Timing.** Gives each 8-K the trading day whose prices first reflect it (NYSE calendar with holidays and early closes) and whether it arrived before the open, during trading or after the close.
# 2. **Content.** Separates press releases from slide decks and transcripts, flags 8-Ks that report trial results (with a rough positive/negative label), FDA events, and the trials each 8-K mentions: by trial ID (NCT number) or by the drug code names in that firm's trials.
# 3. **Registry changes.** From AACT's monthly snapshots: status changes (e.g. terminated), primary completion dates moved, enrollment changes, primary outcomes edited and results posted, each dated by when ClinicalTrials.gov posted it, and owned by whichever firm sponsored the trial at that time.
# 4. **Disclosure timing.** For each registry change, whether and when the firm mentioned that trial in an 8-K.
# 5. **Readout list.** Every readout 8-K, plus the ones from the last two years for the OCC options-volume download.
#
# **How to run**
# - Needs Notebook 1's files, and Notebook 2's files (Steps 4–5 wait for the full AACT run; the rest runs without it).
# - **First run:** leave `DRY_RUN = True` and *Run all*. Step 3 reads 10 text shards spread over the years. Check the last cell's output.
# - **Full run:** set `DRY_RUN = False` and *Run all* (about 30–60 minutes; Step 3 resumes if interrupted).
#
# **Output** (in `MyDrive/biopharma_tiv/features/`): `filings.parquet`, `filing_text_features.parquet`, `filing_trial_links.parquet`, `registry_events.parquet`, `readouts.parquet`, `readouts_occ_window.csv`.


# %% CONFIG
# ==== Settings ====
import os
PROJECT_DIR = os.environ.get("WKF_PROJECT_DIR", "/content/drive/MyDrive/biopharma_tiv")
SIC_CODES = {"2834", "2835", "2836"}   # same as Notebook 1
DRY_RUN = True                         # True: Step 3 reads only DRY_RUN_SHARDS text shards, spread over the years
DRY_RUN_SHARDS = 10
OCC_YEARS = 2                          # OCC's per-stock options volume goes back about two years


# %% [markdown]
# ## Step 0 — Setup
# Mounts Drive and installs the NYSE trading calendar (`exchange_calendars`) if needed.


# %% SETUP
import os, re, glob, json, time, datetime as dt
import numpy as np
import pandas as pd
try:
    import exchange_calendars as xcals
    from rapidfuzz import process, fuzz
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "exchange_calendars", "rapidfuzz"], check=True)
    import exchange_calendars as xcals
    from rapidfuzz import process, fuzz

if PROJECT_DIR.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")

EDGAR_DIR = os.path.join(PROJECT_DIR, "edgar")
SHARD_DIR = os.path.join(EDGAR_DIR, "8k_text", "shards")
CTG_DIR   = os.path.join(PROJECT_DIR, "ctgov")
FEAT_DIR  = os.path.join(PROJECT_DIR, "features")
TEXT_VERSION = "v6"                    # Step 3 saves per-shard results here; a new version when its rules change
TEXT_DIR  = os.path.join(FEAT_DIR, f"text_shards_{TEXT_VERSION}" + ("_dryrun" if DRY_RUN else ""))
for d in (FEAT_DIR, TEXT_DIR):
    os.makedirs(d, exist_ok=True)
print("Saving to:", FEAT_DIR)


# %% [markdown]
# ## Step 1 — Load Notebook 1 and 2 outputs


# %% LOAD
def newest(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return files[-1] if files else None

UNI_PATH = newest(os.path.join(EDGAR_DIR, "universe_*.parquet"))
FI_PATH = newest(os.path.join(EDGAR_DIR, "filing_index_*.parquet"))
assert UNI_PATH and FI_PATH, "Notebook 1's firm list or filing list is missing."
universe = pd.read_parquet(UNI_PATH)
universe["cik"] = universe["cik"].astype(int)
filing_index = pd.read_parquet(FI_PATH).drop_duplicates("accession")
meta_paths = sorted(glob.glob(os.path.join(SHARD_DIR, "meta_*.parquet")))
doc_paths = sorted(glob.glob(os.path.join(SHARD_DIR, "docs_*.parquet")))
assert meta_paths and doc_paths, f"No 8-K text shards in {SHARD_DIR}: run Notebook 1's full download first."

CUR_PATH = newest(os.path.join(CTG_DIR, "industry_trials_current*.parquet"))
assert CUR_PATH, "Notebook 2's current trial records are missing (Step 2 of Notebook 2)."
cur = pd.read_parquet(CUR_PATH)
trial_firm = pd.read_parquet(os.path.join(CTG_DIR, "trial_firm.parquet"))
trial_firm["cik"] = trial_firm["cik"].astype(int)
sponsor_matches = pd.read_csv(os.path.join(CTG_DIR, "sponsor_matches.csv"))
aact_paths = sorted(glob.glob(os.path.join(CTG_DIR, "aact_monthly", "aact_*.parquet")))

cik2name = universe.set_index("cik")["name"].to_dict()
cik2ticker = {c: (str(t).split(";")[0] if str(t or "").strip() else None)
              for c, t in zip(universe["cik"], universe["tickers"].fillna(""))}
print(f"Firms: {len(universe):,} | 8-Ks in the filing list: {len(filing_index):,} | text shards: {len(doc_paths):,}")
print(f"Industry trials: {len(cur):,} | trial-firm links: {len(trial_firm):,} | AACT monthly files: {len(aact_paths)}"
      + ("" if aact_paths else "  (Steps 4-5 will be skipped until Notebook 2's full run finishes)"))


# %% [markdown]
# ## Step 2 — When each 8-K reached the market
# The header's acceptance time is in Eastern time. Each filing gets the first NYSE session whose close comes after it (`event_date`), and a timing label:
# - `pre_open`: before that day's open, so prices react at the open;
# - `market_hours`: during trading;
# - `after_close`: after a close, at a weekend or on a holiday, so prices react the next session.
#
# Also attaches the 8-K item numbers (e.g. 2.02 earnings, 7.01 Regulation FD, 8.01 other events) and flags filings made under a SIC code outside the universe (e.g. after a reverse merger).


# %% TIMING
meta = pd.concat([pd.read_parquet(p, columns=["accession", "cik", "acceptance_datetime", "company_header", "sic_header",
                                                 "items_header", "n_ex99"]) for p in meta_paths], ignore_index=True)
meta = meta.drop_duplicates("accession")
meta["acceptance_et"] = pd.to_datetime(meta["acceptance_datetime"], format="%Y%m%d%H%M%S", errors="coerce")

cal = xcals.get_calendar("XNYS", start=str((meta["acceptance_et"].min() - pd.Timedelta(days=10)).date()),
                         end=str((meta["acceptance_et"].max() + pd.Timedelta(days=30)).date()))
sched = cal.schedule
opens = sched["open"].dt.tz_convert("America/New_York").dt.tz_localize(None).to_numpy()
closes = sched["close"].dt.tz_convert("America/New_York").dt.tz_localize(None).to_numpy()
sessions = sched.index.tz_localize(None) if sched.index.tz is not None else sched.index

ok = meta["acceptance_et"].notna().to_numpy()
t = meta["acceptance_et"].to_numpy()
i = np.searchsorted(closes, t[ok], side="right")              # first session whose close is after the filing
event = np.full(len(meta), np.datetime64("NaT"), dtype="datetime64[ns]")
timing = np.full(len(meta), None, dtype=object)
event[ok] = sessions.to_numpy()[i]
same_day = pd.to_datetime(t[ok]).normalize().to_numpy() == sessions.to_numpy()[i]
timing[ok] = np.where(t[ok] >= opens[i], "market_hours", np.where(same_day, "pre_open", "after_close"))
meta["event_date"] = event
meta["timing"] = timing

fi = filing_index[["accession", "filing_date", "items_json"]].copy()
meta = meta.merge(fi, on="accession", how="left")
meta["items"] = meta["items_json"].fillna("").astype(str).str.findall(r"\d\.\d{2}").str.join(";")
meta["sic_ok"] = meta["sic_header"].isin(SIC_CODES)
filings = meta[["accession", "cik", "company_header", "acceptance_et", "event_date", "timing", "filing_date",
                "items", "sic_header", "sic_ok", "n_ex99"]].copy()
filings["cik"] = filings["cik"].astype(int)
filings.to_parquet(os.path.join(FEAT_DIR, "filings.parquet"), index=False)

print(f"{len(filings):,} 8-Ks | missing acceptance time: {filings['acceptance_et'].isna().mean():.1%} "
      f"| filed under another SIC code: {(~filings['sic_ok']).mean():.1%}")
print(filings["timing"].value_counts(normalize=True).round(3).to_string())
items = filings["items"].str.split(";").explode()
print("\nMost common 8-K items:\n" + items[items != ""].value_counts().head(8).to_string())


# %% [markdown]
# ## Step 3 — What each 8-K says
# Reads the text shards one at a time (saved per shard, so it resumes) and records for each filing:
# - **document kinds:** the 8-K itself (`form_8k`); and for each Exhibit 99: press release (description, wire service, "today announced", dateline or ticker), slide deck (description, or a layout of many short lines), transcript, or other;
# - **readout:** results language ("met its primary endpoint", "topline results", "results from the Phase 2 trial", "statistically significant") in the headline or first paragraph of a press release (or of the 8-K's items when there is none), stopping at the "About the company" and forward-looking-statements sections. Skipped when the sentence looks ahead ("expects topline data in the first half", "will present data") or when the headline is about something else first (quarterly results, business update, FDA action, financing, a deal, management, restructuring, preclinical data, a trial starting or finishing enrollment, a conference). `readout_kind` is `topline_or_endpoint` (primary endpoint, topline or statistical-significance wording) or `data_update` (e.g. updated data at a conference). Tone (positive / negative / mixed / unclear) is a keyword count in the first 6,000 characters;
# - **FDA news:** approval, complete response letter, advisory committee, PDUFA date, in the headline area (first 1,000 characters);
# - **trial links:** NCT numbers anywhere in the filing, and the drug code names (e.g. "VX-548") or drug names unique to that firm's own trials, with how many of the firm's trials share that name.
#
# These are screening rules; the paper's measure comes later. Check `headline` and `readout_snippet` in `readouts.parquet` when auditing.


# %% TEXT
# ---- drug names per firm, from the firm's own trials ----
STOP_NAMES = {"placebo", "saline", "standard of care", "best supportive care", "vehicle", "sham", "observation", "control",
              "usual care", "chemotherapy", "radiotherapy", "radiation", "surgery", "matching placebo", "placebo comparator",
              "background therapy", "standard therapy", "investigator's choice", "physician's choice", "diet", "exercise"}
CODE_NAME = re.compile(r"^[a-z]{1,8}[- ]?\d{2,6}[a-z]?$")

def name_variants(raw):
    out = set()
    for piece in re.split(r"[;()\[\]/,]+", str(raw or "")):
        s = re.sub(r"^(drug|biological|device|other|procedure|combination product|genetic|dietary supplement)\s*:\s*", "",
                   piece.strip(), flags=re.I)
        s = re.sub(r"\s+\d+(\.\d+)?\s*(mg|mcg|µg|ug|g|ml|%|iu|units?|mg/kg|mg/m2)\b.*$", "", s, flags=re.I).strip().lower()
        if len(s) >= 4 and re.match(r"[a-z]", s):          # a name starts with a letter ("2.8 mg" is a dose, not a drug)
            out.add(s)
    return out

tf_all = trial_firm[["nct_id", "cik"]].drop_duplicates().merge(cur[["nct_id", "interventions"]], on="nct_id", how="left")
pairs = [(c, n, v) for c, n, raw in zip(tf_all["cik"], tf_all["nct_id"], tf_all["interventions"]) for v in name_variants(raw)]
names = pd.DataFrame(pairs, columns=["cik", "nct_id", "name"])
firms_per_name = names.groupby("name")["cik"].nunique()
keep_name = names["name"].map(lambda s: bool(CODE_NAME.match(s))) | (
    names["name"].str.len().ge(6) & names["name"].map(firms_per_name).eq(1) & ~names["name"].isin(STOP_NAMES))
names = names[keep_name & ~names["name"].isin(STOP_NAMES)]
name2ncts = names.groupby(["cik", "name"])["nct_id"].agg(lambda s: sorted(set(s))).to_dict()

CODE_TOKEN = re.compile(r"(?<![\w-])([A-Za-z]{1,8})[- ]?(\d{2,6}[A-Za-z]?)(?![\w-])")
firm_names = {}                                              # cik -> (code names by key, single words, phrases)
for c, grp in names.groupby("cik"):
    ns = set(grp["name"])
    codes = {re.sub(r"[- ]", "", n): n for n in ns if CODE_NAME.match(n)}
    rest = ns - set(codes.values())
    words = {n for n in rest if re.fullmatch(r"[a-z][a-z0-9]+", n)}
    firm_names[c] = (codes, words, sorted(rest - words))

def drug_names_in(cik, text):
    fn = firm_names.get(cik)
    if not fn:
        return set()
    codes, words, phrases = fn
    found = set()
    if codes:
        for m in CODE_TOKEN.finditer(text):
            key = (m.group(1) + m.group(2)).lower()
            if key in codes:
                found.add(codes[key])
    low = text.lower()
    if words:
        found |= words & set(re.findall(r"[a-z][a-z0-9]+", low))
    found |= {ph for ph in phrases if ph in low}
    return found

print(f"Drug names used for linking: {names['name'].nunique():,} across {len(firm_names):,} firms")

# ---- patterns ----
LEAD_CHARS, HEAD_CHARS, FDA_CHARS, TONE_CHARS = 2000, 600, 1000, 6000   # where each rule looks, from the top of a document
WIRE = re.compile(r"GLOBE\s?NEWSWIRE|BUSINESS\s?WIRE|PR\s?NEWSWIRE|ACCESS\s?(?:NEWS)?WIRE|MARKETWIRED?|PRWEB|EQS\s?NEWSWIRE", re.I)
PR_STRONG = re.compile(r"for\s+immediate\s+release|\bpress\s+release\b|\bnews\s+release\b|"
                       r"\btoday\s+(?:announced|reported|provided|issued|unveiled|presented)\b", re.I)
PR_WEAK = [re.compile(r"\((?:nasdaq|nyse|amex|tsx|otc)[a-z\s]*:\s*[a-z.]{1,6}\)", re.I),                 # (Nasdaq: ACME)
           re.compile(r"\b[A-Z][A-Z .'-]{2,},\s+(?:[A-Z][a-z]+\.?,\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                      r"[a-z]*\.?\s+\d{1,2},?\s+(?:19|20)\d{2}"),                                         # BOSTON, Jan. 8, 2025
           re.compile(r"\b(?:investor|media)\s+(?:relations|contacts?)\b|\bcontacts?\s*:", re.I)]
PR_HINT = re.compile(r"press|release|\bnews\b", re.I)
DECK_HINT = re.compile(r"presentation|slides?\b|deck|corporate overview|investor day|poster|r&d day", re.I)
TRANSCRIPT = re.compile(r"transcript", re.I)
OPERATOR = re.compile(r"^\s*Operator\s*:?\s*$|\bOperator\s*:", re.M)
ITEM_HEAD = re.compile(r"\bItem\s+\d{1,2}\.\d{2}\b", re.I)

PHASE = re.compile(r"\bphase\s*(?:[1-4]|iv|i{1,3})(?:\s*[/-]\s*(?:[1-4]|i{1,3}))?[ab]?\b", re.I)
STUDYWORD = re.compile(r"\b(?:trials?|stud(?:y|ies))\b", re.I)
READ_RX = {   # results language; a readout needs one of these in the first LEAD_CHARS, not in a forward-looking sentence
    "endpoint": re.compile(r"\b(?:met|meets|achieved|achieves|reached|hit|did\s+not\s+(?:meet|achieve|reach)|does\s+not\s+meet|"
                           r"fail(?:s|ed)?\s+to\s+(?:meet|achieve|reach)|miss(?:es|ed))\s+(?:the\s+|its\s+|their\s+|both\s+|all\s+|"
                           r"each\s+|one\s+of\s+(?:the\s+|its\s+)?|of\s+its\s+)?(?:pre-?specified\s+)?(?:co-?)?primary\s+"
                           r"(?:efficacy\s+|safety\s+)?(?:and\s+(?:key\s+)?secondary\s+)?(?:end\s?-?points?|objectives?)|"
                           r"primary\s+(?:efficacy\s+)?end\s?-?points?\s+(?:was|were|has\s+been)\s+(?:not\s+)?(?:met|achieved|reached)", re.I),
    "topline": re.compile(r"\btop[- ]?line\s+(?:\w+\s+){0,3}?(?:results?|data|read[- ]?outs?)\b", re.I),
    "results_from": re.compile(r"\b(?:results?|data)\s+(?:from|of|in|for)\s+(?:the\s+|its\s+|our\s+|a\s+|an\s+)?[^.\n]{0,80}?"
                               r"\b(?:phase|trials?|stud(?:y|ies))\b", re.I),
    "stat_sig": re.compile(r"\bstatistically\s+significant|\bstatistical\s+significance|\bp\s*[<=≤]\s*0?\.\d", re.I),
}
# results phrase followed later in the same clause by "expected", "to be reported" ... ("Topline results of 470 patients expected in Q1")
FUTURE_AFTER = re.compile(r"\b(?:expected|anticipated|to\s+be\s+(?:reported|presented|announced|released)|"
                          r"will\s+be\s+(?:reported|presented|announced|released|available))\b", re.I)
FUTURE = re.compile(r"\b(?:expect(?:s|ed|ing)?|anticipat(?:e|es|ed|ing)|plan(?:s|ned)?\s+to|intend(?:s|ed)?\s+to|will|shall|"
                    r"to\s+be\s+(?:presented|reported|announced|released)|to\s+(?:present|report|announce|release|share|host)|"
                    r"upcoming|on\s+track|scheduled|guidance|(?:first|second)\s+half|by\s+(?:the\s+)?end\s+of|year[- ]end|"
                    r"later\s+(?:this|in)|next\s+year|designed\s+to|powered\s+to|first\s+patients?\s+(?:dosed|enrolled|randomized))\b", re.I)
OTHER_TOPICS = [   # if one of these comes first in the headline area, the document is about something else
    re.compile(r"\b(?:q[1-4]|(?:first|second|third|fourth)[- ]quarter)\s+(?:(?:and\s+)?(?:full[- ]year\s+)?(?:fiscal\s+)?(?:19|20)\d{2}\s+)?"
               r"(?:financial|results|earnings|operating|business|update|report)|\b(?:first|second|third|fourth)[- ]quarter\s+"
               r"(?:and\s+(?:full|fiscal)[- ]year\s+)?(?:19|20)\d{2}|\bfull[- ]year\s+(?:19|20)\d{2}|\bfiscal\s+(?:year\s+)?(?:19|20)\d{2}|"
               r"\bfinancial\s+results\b|\bresults\s+of\s+operations\b|\bearnings\b|\b(?:business|corporate|pipeline)\s+"
               r"(?:update|highlights|progress|review)\b|\b(?:shareholder|stockholder)\s+letter|\bletter\s+to\s+(?:share|stock)holders", re.I),
    re.compile(r"\b(?:FDA|EMA|CHMP|PMDA|NMPA|MHRA|PDUFA|NDA|BLA|sNDA|sBLA|MAA|IND|CRL|ODAC)\b"),
    re.compile(r"\b(?:Food\s+and\s+Drug\s+Administration|European\s+Commission|priority\s+review|breakthrough\s+therapy|fast\s+track|"
               r"orphan\s+drug|accelerated\s+approval|marketing\s+authori[sz]ation|complete\s+response|advisory\s+committee|"
               r"approv(?:es|ed|al)|clinical\s+hold|designation)\b", re.I),
    re.compile(r"\b(?:public\s+offering|underwritten\s+offering|private\s+placement|registered\s+direct|pricing\s+of|"
               r"at-the-market|convertible\s+(?:senior\s+)?notes|credit\s+facility|term\s+loan|unregistered\s+sales)\b", re.I),
    re.compile(r"\b(?:acquir(?:e|es|ed|ing)|acquisition|merger|tender\s+offer|definitive\s+agreement|licen[cs]e\s+agreement|"
               r"collaboration\s+(?:and\s+licen[cs]e\s+)?agreement|strategic\s+(?:collaboration|partnership|alternatives)|"
               r"exclusive\s+licen[cs]e)\b", re.I),
    re.compile(r"\b(?:appoints?|appointed|appointment\s+of|elect(?:s|ed)\s+to|resign(?:s|ed|ation)|retire(?:s|ment)|steps?\s+down|"
               r"departure\s+of|names\s+(?:new\s+)?(?:chief|ceo|cfo|president|head)|(?:management|leadership|executive)\s+"
               r"(?:changes?|transition))\b", re.I),
    re.compile(r"\b(?:restructuring|reduction\s+in\s+(?:force|workforce)|workforce\s+reduction|cost\s+reduction|wind[- ]down|"
               r"strategic\s+(?:actions|update|review|realignment|reprioriti[sz]ation|priorities))\b", re.I),
    re.compile(r"\b(?:pre-?clinical|non-?clinical|animal\s+(?:models?|stud(?:y|ies))|in\s+vivo|murine|mouse\s+models?)\b", re.I),
    re.compile(r"\b(?:initiat(?:es|ed|ion\s+of)|commenc(?:es|ed|ement\s+of)|first\s+(?:patients?|subjects?|participants?)\s+(?:dosed|"
               r"enrolled|randomi[sz]ed|treated)|enrollment\s+of\s+(?:the\s+)?first|(?:completes?|completed|completion\s+of|achieves|"
               r"reaches|surpasses|exceeds)\b[^.\n]{0,40}?\b(?:enrollment|recruitment|dosing)|completion\s+of\s+(?:the\s+)?phase|"
               r"doses?\s+first)\b", re.I),
    re.compile(r"\b(?:to\s+(?:present|participate|host|webcast)|will\s+(?:present|participate|host)|participat(?:e|es|ion)\s+in|"
               r"fireside\s+chat|investor\s+conferences?|annual\s+(?:shareholder\s+|stockholder\s+)?meeting|r&d\s+day|"
               r"investor\s+day|publication\b|published\s+(?:in|online))", re.I),
]
POS = re.compile(r"\b(?:met|meets|achieved|reached|hit)\s+(?:the\s+|its\s+|their\s+|both\s+|all\s+|each\s+)?(?:of\s+its\s+)?(?:co-?)?"
                 r"primary\s+(?:efficacy\s+)?end\s?-?points?|primary\s+(?:efficacy\s+)?end\s?-?points?\s+(?:was|were)\s+(?:met|achieved)|"
                 r"primary\s+end\s?-?point\s+of\s+[^.]{0,80}?\bmet\b|positive\s+(?:[\w-]+\s+){0,3}?(?:results|data|outcomes?|findings)|"
                 r"(?:encouraging|favorable|compelling)\s+(?:[\w-]+\s+){0,2}?(?:results|data)|achieved\s+statistical\s+significance|"
                 r"statistically\s+significant\s+(?:and\s+clinically\s+meaningful\s+)?(?:improvement|reduction|benefit|increase|difference)|"
                 r"clinically\s+meaningful\s+(?:improvement|reduction|benefit)", re.I)
NEGATED = re.compile(r"\b(?:no|not|without|lack(?:ed)?\s+of|failed\s+to\s+show)\b[^.]{0,30}$", re.I)
NEG = re.compile(r"did\s+not\s+(?:meet|achieve|reach|show|demonstrate|improve|separate)|does\s+not\s+meet|"
                 r"failed\s+to\s+(?:meet|achieve|reach|demonstrate|show)|primary\s+(?:efficacy\s+)?end\s?-?points?\s+(?:was|were)\s+not\s+"
                 r"(?:met|achieved)|miss(?:ed|es)\s+(?:the\s+|its\s+)?(?:co-?)?primary|not\s+statistically\s+significant|"
                 r"no\s+(?:statistically\s+)?significant\s+(?:difference|improvement|benefit)|"
                 r"(?:negative|disappointing)\s+(?:[\w-]+\s+){0,3}?(?:results|data)|"
                 r"(?:stopped|halted|terminated|discontinued)\s+(?:\w+\s+){0,4}?for\s+futility|"
                 r"discontinu(?:e|es|ed|ing)\s+(?:the\s+|its\s+|further\s+)?(?:development|program|trial|study)", re.I)
FDA = {"fda_approval": re.compile(r"\b(?:FDA|Food\s+and\s+Drug\s+Administration)\s+(?:has\s+)?(?:granted\s+(?:accelerated\s+|full\s+|"
                                  r"traditional\s+)?approval|approve[sd])\b|\breceive[sd]?\s+(?:U\.S\.\s+)?(?:FDA|Food\s+and\s+Drug\s+"
                                  r"Administration)\s+(?:accelerated\s+|full\s+|marketing\s+)?approval\b|\b(?:FDA|Food\s+and\s+Drug\s+"
                                  r"Administration)\s+(?:accelerated\s+|full\s+)?approval\s+(?:of|for)\b|\bapproval\s+(?:from|by)\s+the\s+"
                                  r"(?:U\.S\.\s+)?(?:FDA|Food\s+and\s+Drug\s+Administration)\b", re.I),
       "fda_crl": re.compile(r"(?i:\bcomplete\s+response\s+letter)|\bCRL\b"),
       "fda_adcom": re.compile(r"\b(?:FDA|ODAC)\b[^.\n]{0,60}(?i:advisory\s+committee)|(?i:advisory\s+committee\b[^.\n]{0,60}\bvote[sd]?\b)|"
                               r"(?i:advisory\s+committee)\b[^.\n]{0,60}\bFDA\b|\bODAC\b"),
       "fda_pdufa": re.compile(r"\bPDUFA\b")}
FDA_NOT_FUTURE = {"fda_approval", "fda_crl"}     # "expects FDA approval by ..." is not an approval
NCT = re.compile(r"\bNCT\s?0?(\d{7,8})\b", re.I)
# company boilerplate and legal text: rules stop reading here ("About Acme ...", "Forward-Looking Statements", ...)
BOILER = re.compile(r"(?:^|\n|[.!?]\s+)\s*(?:About|ABOUT)\s+(?!\d)(?:the\s+|THE\s+)?[A-Z]|"
                    r"(?i:forward[- ]looking\s+statements|safe\s+harbou?r\b|cautionary\s+(?:note|statement))")
STRONG_RULES = {"endpoint", "topline", "stat_sig"}

def squash(s):
    return re.sub(r"\s+", " ", s or "").strip()

def content(t, n):
    """The first n characters, cut where the boilerplate starts (if it starts after the first 200)."""
    x = t[:n]
    m = BOILER.search(x, 200)
    return x[:m.start()] if m else x

def headline_of(t):
    return squash(re.sub(r"^.{0,80}?\bEX(?:HIBIT)?[\s.-]*99(?:\.\d+)?\s*", "", t[:500], flags=re.I | re.S))[:300]

def clause_end(x, m, after):
    ends = [i for i in (x.find(". ", m.end()), x.find("\n", m.end()), x.find("; ", m.end())) if i != -1]
    return min(ends + [len(x), m.end() + after])

def clause_start(x, m):
    return max(x.rfind(". ", 0, m.start()), x.rfind("\n", 0, m.start()), x.rfind("; ", 0, m.start())) + 1

def window(x, m, after=40):
    """The clause holding match m: from the previous sentence break to the next one, at most `after` characters past the match."""
    return x[clause_start(x, m):clause_end(x, m, after)]

def is_future(x, m):
    """Forward-looking: a future word in the clause up to 40 characters past the match, or "expected" etc. up to 150 past it."""
    return bool(FUTURE.search(window(x, m)) or FUTURE_AFTER.search(x[m.end():clause_end(x, m, 150)]))

def doc_kind(doc_type, desc, fname, text):
    if not str(doc_type or "").upper().startswith("EX-99"):
        return "form_8k"                                      # the 8-K itself (cover page, items) and non-99 exhibits
    d, top = f"{desc or ''} {fname or ''}", text[:3000]
    if TRANSCRIPT.search(d) or OPERATOR.search(text[:20000]):
        return "transcript"
    if PR_HINT.search(desc or ""):
        return "press_release"
    if DECK_HINT.search(d):
        return "slides"
    if WIRE.search(top) or PR_STRONG.search(top) or sum(bool(rx.search(top)) for rx in PR_WEAK) >= 2:
        return "press_release"
    lines = [l.strip() for l in text[:20000].split("\n") if l.strip()]
    if len(lines) >= 60 and np.median([len(l) for l in lines]) < 40 and np.mean([l[-1] in ".!?" for l in lines]) < 0.3:
        return "slides"
    return "other_ex99"

def body(kind, text):
    """Where the content starts: the first Item heading for the 8-K itself (skips the cover page), else the top."""
    if kind == "form_8k":
        m = ITEM_HEAD.search(text)
        if m:
            return text[m.start():]
    return text

PARA_CHARS = 900   # how far past the dateline the first paragraph is read

def first_para_end(x):
    """End of the first paragraph: PARA_CHARS past the dateline or wire tag, or 1,200 characters when there is none."""
    ms = [m for rx in (WIRE, PR_WEAK[1]) for m in [rx.search(x)] if m]
    return min(m.end() for m in ms) + PARA_CHARS if ms else 1200

def readout_hit(t):
    """(rule, match, strong) for the first results statement in the headline or first paragraph, or None."""
    lead = content(t, LEAD_CHARS)
    lead = lead[:first_para_end(lead)]
    if not (PHASE.search(lead) or STUDYWORD.search(lead)):
        return None
    hits = sorted(((m.start(), rule, m) for rule, rx in READ_RX.items() for m in rx.finditer(lead)), key=lambda h: h[0])
    if not hits:
        return None
    first = hits[0][2]
    if first.start() < HEAD_CHARS and FUTURE.search(lead[clause_start(lead, first):first.start()]):
        return None                     # the headline previews results still to come ("to Report Data from Phase 2 Trial in Q4")
    done = [h for h in hits if not is_future(lead, h[2])]                  # results already in, not "expects" / "will present"
    if not done:
        return None
    other = [m.start() for rx in OTHER_TOPICS for m in [rx.search(lead[:HEAD_CHARS])] if m]
    if other and min(other) < hits[0][0]:                                  # headline is about something else first
        return None
    return done[0][1], done[0][2], any(h[1] in STRONG_RULES for h in done)

def n_pos(x):
    return sum(1 for m in POS.finditer(x)
               if not NEGATED.search(x[max(0, m.start() - 40):m.start()]) and not re.search(r"\bnot\b", m.group(), re.I))

def features(acc, cik, docs):
    texts = [r.text or "" for r in docs.itertuples()]
    kinds = [doc_kind(r.doc_type, r.description, r.filename, t) for r, t in zip(docs.itertuples(), texts)]
    by = {}
    for k, t in zip(kinds, texts):
        by.setdefault(k, []).append(body(k, t))
    order = [k for k in ("press_release", "other_ex99", "form_8k") if k in by]
    src = order[0] if order else ("slides" if "slides" in by else "transcript")
    cands = [t for k in order for t in by[k]] or [t for ts in by.values() for t in ts]   # read in this order
    hit = None
    for t in cands:
        h = readout_hit(t)
        if h:
            hit = (t, h)
            break
    pos = neg = 0
    tone = rule = kind = None
    snippet = ""
    head = headline_of(cands[0]) if cands else ""
    if hit:
        t, (rule, m, strong) = hit
        kind = "topline_or_endpoint" if strong else "data_update"
        head = headline_of(t)
        win = content(t, TONE_CHARS)
        pos, neg = n_pos(win), len(NEG.findall(win))
        tone = "mixed" if pos and neg else "positive" if pos else "negative" if neg else "unclear"
        snippet = squash(t[max(0, m.start() - 100):m.end() + 250])
    alltext = "\n".join(texts)
    ncts = sorted({"NCT" + m.zfill(8) for m in NCT.findall(alltext)})
    row = {"accession": acc, "cik": cik, "n_docs": len(docs), "kinds": ";".join(sorted(set(kinds))), "text_source": src,
           "lead_chars": sum(len(t) for t in by.get(src, [])), "has_press_release": "press_release" in by,
           "has_slides": "slides" in by, "phase_mention": bool(PHASE.search(cands[0][:LEAD_CHARS])) if cands else False,
           "readout": hit is not None, "readout_kind": kind, "readout_rule": rule, "readout_tone": tone, "pos_hits": pos,
           "neg_hits": neg, "nct_ids": ";".join(ncts), "n_nct": len(ncts), "headline": head, "readout_snippet": snippet}
    for k, rx in FDA.items():
        row[k] = any(m for t in cands for x in [content(t, FDA_CHARS)] for m in rx.finditer(x)
                     if k not in FDA_NOT_FUTURE or not FUTURE.search(window(x, m)))
    links = [(acc, n, "nct", n, np.nan) for n in ncts]
    for n in drug_names_in(cik, alltext):
        hits_n = name2ncts.get((cik, n), [])
        for nct in hits_n:
            links.append((acc, nct, "drug_name", n, len(hits_n)))
    return row, links

acc2cik = filing_index.set_index("accession")["cik"].astype(int).to_dict()
if DRY_RUN:   # shards spread over the sample period, so the dry run covers several years
    todo = [doc_paths[i] for i in sorted(set(np.linspace(0, len(doc_paths) - 1, DRY_RUN_SHARDS).round().astype(int)))]
else:
    todo = doc_paths
t0 = time.time()
for k, p in enumerate(todo, 1):
    sid = re.search(r"(\d+)", os.path.basename(p)).group(1)
    out_f, out_l = os.path.join(TEXT_DIR, f"feat_{sid}.parquet"), os.path.join(TEXT_DIR, f"links_{sid}.parquet")
    if os.path.exists(out_f) and os.path.exists(out_l):
        continue
    docs = pd.read_parquet(p, columns=["accession", "doc_type", "description", "filename", "text"])
    rows, links = [], []
    for acc, grp in docs.groupby("accession", sort=False):
        r, l = features(acc, acc2cik.get(acc, -1), grp)
        rows.append(r); links += l
    pd.DataFrame(rows).to_parquet(out_f + ".tmp", index=False); os.replace(out_f + ".tmp", out_f)
    pd.DataFrame(links, columns=["accession", "nct_id", "link_type", "matched", "n_name_trials"]).to_parquet(out_l + ".tmp", index=False)
    os.replace(out_l + ".tmp", out_l)
    if k % 10 == 0 or k == len(todo):
        el = time.time() - t0
        print(f"  shard {k}/{len(todo)} | about {el / k * (len(todo) - k) / 60:.0f} min left")

text_feat = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(TEXT_DIR, "feat_*.parquet")))], ignore_index=True)
links = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(TEXT_DIR, "links_*.parquet")))], ignore_index=True)
links = links.drop_duplicates(["accession", "nct_id", "link_type"])
text_feat.to_parquet(os.path.join(FEAT_DIR, f"filing_text_features{'_dryrun' if DRY_RUN else ''}.parquet"), index=False)
links.to_parquet(os.path.join(FEAT_DIR, f"filing_trial_links{'_dryrun' if DRY_RUN else ''}.parquet"), index=False)
print(f"\n{len(text_feat):,} filings read | readouts: {text_feat['readout'].sum():,} | "
      f"filings linked to a trial: {links['accession'].nunique():,} ({links['nct_id'].nunique():,} trials)")


# %% [markdown]
# ## Step 4 — Registry changes from AACT's monthly snapshots
# Compares each trial's record with the previous month's and keeps:
# - `status`: overall status changed (e.g. to `TERMINATED`, `SUSPENDED`, `WITHDRAWN`, `COMPLETED`);
# - `pcd_later` / `pcd_earlier`: primary completion date moved by 3+ months (compared at month level, since old snapshots give month only);
# - `enrollment_down` / `enrollment_up`: enrollment changed by 20%+;
# - `outcome_edit`: the list of primary outcomes changed;
# - `results_posted`: results appeared.
#
# Each change is dated by the record's *last update posted* date when that falls between the two snapshots, else by the *last update submitted* date, else by the snapshot (`date_source` says which); results are dated by their own posting date and count once, when first posted after the archive starts (2017 posts are picked up from the 2018-01 snapshot, when AACT added the field). A **stop** (`is_stop`) is a recruiting, active or suspended trial becoming terminated, suspended or withdrawn, leaving out stale records being cleaned up (`stale_record`: not updated for 2+ years, or primary completion 2+ years past; `stop_any` keeps them). `firm_lead_trials` counts the trials the firm leads today, to separate small firms from big pharma. The firm is the lead sponsor at that time, matched the same way as Notebook 2. If that name doesn't match a firm (e.g. an academic-led trial with an industry collaborator), it falls back to today's link from Notebook 2, and `owner_source` says which (`sponsor_at_time`, `current_lead` or `collaborator`), so collaborator-only trials can be filtered out.


# %% REGISTRY
registry_events = None
if not aact_paths:
    print("AACT monthly files not found yet: run this step after Notebook 2's full run.")
else:
    panel = pd.concat([pd.read_parquet(p) for p in aact_paths], ignore_index=True)
    panel["snap"] = pd.to_datetime(panel["snapshot_date"])
    panel = panel.sort_values(["nct_id", "snap"]).reset_index(drop=True)
    for c in ["overall_status", "primary_completion_date", "enrollment", "primary_outcomes_hash", "results_first_posted_date",
              "last_update_posted_date", "last_update_submitted_date", "phase", "lead_sponsor"]:
        if c not in panel:
            panel[c] = np.nan
    cols = ["snap", "overall_status", "primary_completion_date", "enrollment", "primary_outcomes_hash", "results_first_posted_date",
            "last_update_posted_date", "last_update_submitted_date"]
    prev = panel.groupby("nct_id")[cols].shift(1)
    has = prev["snap"].notna()

    def ym(s):
        d = pd.to_datetime(s, errors="coerce")
        return d.dt.year * 12 + d.dt.month

    new_s, old_s = panel["overall_status"], prev["overall_status"]
    pcd_move = ym(panel["primary_completion_date"]) - ym(prev["primary_completion_date"])
    en_new = pd.to_numeric(panel["enrollment"], errors="coerce")
    en_old = pd.to_numeric(prev["enrollment"], errors="coerce")
    en_ratio = en_new / en_old.where(en_old > 0)
    hash_new = panel["primary_outcomes_hash"]
    hash_old = prev["primary_outcomes_hash"]
    rp_new = pd.to_datetime(panel["results_first_posted_date"], errors="coerce")
    checks = {
        "status": has & new_s.notna() & old_s.notna() & (new_s != old_s),
        "pcd_later": has & (pcd_move >= 3),
        "pcd_earlier": has & (pcd_move <= -3),
        "enrollment_down": has & (en_ratio <= 0.8),
        "enrollment_up": has & (en_ratio >= 1.25),
        "outcome_edit": has & hash_new.notna() & hash_old.notna() & (hash_new != hash_old),
        # results first seen in this snapshot, posted after the archive starts (2017 posts first appear in 2018-01, when
        # AACT added the field); results posted before the archive are not events
        "results_posted": has & rp_new.notna() & prev["results_first_posted_date"].isna()
                          & (rp_new > panel["snap"].min() - pd.Timedelta(days=7)) & (rp_new <= panel["snap"]),
    }
    ev = []
    for kind, mask in checks.items():
        sub = panel.loc[mask, ["nct_id", "snap", "lead_sponsor", "overall_status", "last_update_posted_date",
                               "last_update_submitted_date", "results_first_posted_date", "phase"]].copy()
        sub["event_type"] = kind
        sub["status_from"] = old_s[mask] if kind == "status" else None
        sub["status_to"] = new_s[mask] if kind == "status" else None
        sub["prev_update"] = prev.loc[mask, "last_update_posted_date"].fillna(prev.loc[mask, "last_update_submitted_date"])
        sub["prev_pcd"] = prev.loc[mask, "primary_completion_date"]
        sub["prev_snap"] = prev.loc[mask, "snap"]
        sub["detail"] = {"status": old_s[mask].astype(str) + " -> " + new_s[mask].astype(str),
                         "pcd_later": pcd_move[mask].astype("Int64").astype(str) + " months",
                         "pcd_earlier": pcd_move[mask].astype("Int64").astype(str) + " months",
                         "enrollment_down": en_old[mask].astype("Int64").astype(str) + " -> " + en_new[mask].astype("Int64").astype(str),
                         "enrollment_up": en_old[mask].astype("Int64").astype(str) + " -> " + en_new[mask].astype("Int64").astype(str),
                         "outcome_edit": "primary outcomes changed",
                         "results_posted": "results first posted"}[kind]
        ev.append(sub)
    registry_events = pd.concat(ev, ignore_index=True)
    lup = pd.to_datetime(registry_events["last_update_posted_date"], errors="coerce")
    lus = pd.to_datetime(registry_events["last_update_submitted_date"], errors="coerce")
    inwin = lambda d: d.notna() & (d > registry_events["prev_snap"]) & (d <= registry_events["snap"])
    registry_events["registry_date"] = np.where(inwin(lup), lup, np.where(inwin(lus), lus, registry_events["snap"]))
    registry_events["registry_date"] = pd.to_datetime(registry_events["registry_date"])
    registry_events["date_source"] = np.where(inwin(lup), "posted", np.where(inwin(lus), "submitted", "snapshot"))
    is_rp = registry_events["event_type"] == "results_posted"      # results carry their own posting date
    registry_events.loc[is_rp, "registry_date"] = pd.to_datetime(registry_events.loc[is_rp, "results_first_posted_date"])
    registry_events.loc[is_rp, "date_source"] = "results_posted"
    registry_events = registry_events[~(is_rp & registry_events.duplicated(["nct_id", "event_type"]))]   # first posting only
    registry_events = registry_events.drop(columns=["results_first_posted_date"]).reset_index(drop=True)
    # a record nobody had touched for 2+ years, or whose primary completion date had passed 2+ years earlier: a clean-up
    two_years = pd.Timedelta(days=730)
    registry_events["stale_record"] = (
        (registry_events["registry_date"] - pd.to_datetime(registry_events["prev_update"], errors="coerce") > two_years)
        | (registry_events["registry_date"] - pd.to_datetime(registry_events["prev_pcd"], errors="coerce") > two_years))
    # a stop: a running (or suspended) trial becomes terminated, suspended or withdrawn; not stale records (UNKNOWN, COMPLETED)
    ACTIVE = {"RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION", "SUSPENDED"}
    STOPPED = {"TERMINATED", "SUSPENDED", "WITHDRAWN"}
    registry_events["stop_any"] = ((registry_events["event_type"] == "status") & registry_events["status_from"].isin(ACTIVE)
                                   & registry_events["status_to"].isin(STOPPED))
    registry_events["is_stop"] = registry_events["stop_any"] & ~registry_events["stale_record"]

    # ---- owner at the time: match that month's lead sponsor name (Notebook 2's method) ----
    SUFFIXES = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "llc", "lp", "llp", "plc", "sa",
                "ag", "nv", "bv", "se", "gmbh", "kk", "spa", "ab", "as", "asa", "oyj", "holdings", "holding", "group", "and", "the",
                "usa", "us"}
    CANON = {"pharmaceuticals": "pharma", "pharmaceutical": "pharma", "pharm": "pharma", "therapeutic": "therapeutics",
             "biosciences": "bioscience", "technologies": "technology", "laboratories": "labs", "laboratory": "labs",
             "international": "intl", "biopharmaceuticals": "biopharma", "biopharmaceutical": "biopharma"}
    def norm_name(s):
        s = str(s or "").lower()
        s = re.sub(r"/[a-z]{1,4}/", " ", s).replace("&", " and ")
        s = re.sub(r"[^a-z0-9 ]+", " ", s)
        toks = [CANON.get(t, t) for t in s.split()]
        while toks and toks[-1] in SUFFIXES:
            toks.pop()
        while toks and toks[0] == "the":
            toks.pop(0)
        return " ".join(toks)
    variants = [(c, n, False) for c, n in zip(universe["cik"], universe["name"])]
    variants += [(c, f.strip(), True) for c, fs in zip(universe["cik"], universe["former_names"].fillna(""))
                 for f in str(fs).split(";") if f.strip()]
    vn = pd.DataFrame(variants, columns=["cik", "variant", "former"])
    vn["norm"] = vn["variant"].map(norm_name)
    vn = vn[vn["norm"].str.len() >= 3]
    norm2ciks = vn.groupby("norm")["cik"].agg(lambda s: sorted(set(s))).to_dict()
    choices = list(norm2ciks)
    exact = {s: int(c) for s, c in zip(sponsor_matches["sponsor"], sponsor_matches["cik"]) if pd.notna(c)}
    alias = {norm_name(s): int(c) for s, c, m in zip(sponsor_matches["sponsor"], sponsor_matches["cik"], sponsor_matches["method"])
             if m == "alias" and pd.notna(c)}
    def owner(name):
        if name in exact:
            return exact[name]
        n = norm_name(name)
        if n in alias:
            return alias[n]
        if n in norm2ciks and len(norm2ciks[n]) == 1:
            return norm2ciks[n][0]
        if len(n) >= 6:
            hit = process.extractOne(n, choices, scorer=fuzz.token_sort_ratio, score_cutoff=92)
            if hit and len(norm2ciks[hit[0]]) == 1:
                return norm2ciks[hit[0]][0]
        return None
    owners = {s: owner(s) for s in registry_events["lead_sponsor"].dropna().unique()}
    registry_events["cik"] = registry_events["lead_sponsor"].map(owners)
    # fallback: today's link from Notebook 2, lead sponsor first, else an industry collaborator (e.g. academic-led trials)
    first_link = (trial_firm.assign(_r=(trial_firm["role"] != "lead").astype(int))
                  .sort_values(["nct_id", "_r"]).drop_duplicates("nct_id").set_index("nct_id"))
    fallback_src = registry_events["nct_id"].map(first_link["role"]).map({"lead": "current_lead", "collaborator": "collaborator"})
    registry_events["owner_source"] = np.where(registry_events["cik"].notna(), "sponsor_at_time", fallback_src)
    registry_events["cik"] = registry_events["cik"].fillna(registry_events["nct_id"].map(first_link["cik"])).astype("Int64")
    # how many trials the firm leads today: a stop matters more to a firm with 5 trials than to one with 2,000
    lead_n = trial_firm[trial_firm["role"] == "lead"].groupby("cik")["nct_id"].nunique()
    registry_events["firm_lead_trials"] = registry_events["cik"].astype("float").map(lead_n.rename(index=float)).fillna(0).astype(int)

    print(f"{len(panel):,} trial-months | {len(registry_events):,} registry changes")
    print(registry_events["event_type"].value_counts().to_string())
    print("\nDated by:\n" + registry_events["date_source"].value_counts(normalize=True).round(3).to_string())
    print("Owner from:\n" + registry_events["owner_source"].value_counts(normalize=True).round(3).to_string())


# %% [markdown]
# ## Step 5 — Did the firm disclose it, and when?
# For each registry change, finds 8-Ks that mention the same trial: by NCT number, or by a drug name used in at most 5 of the firm's trials (`MAX_NAME_TRIALS`). Keeps:
# - `days_to_8k`: days from the registry date to the first such 8-K on or after it (within a year);
# - `disclosed_before`: an 8-K mentioning the trial in the 30 days before the registry date;
# - `same_day_8k`: any 8-K by the same firm within one trading day of the registry date;
# - `firm_8ks_year`: 8-Ks the firm filed within a year either side. Zero means a foreign filer (6-K / 20-F) or a firm not yet or no longer public: leave these out of disclosure tests.
#
# These are the inputs to the "registry first, press release later" test.


# %% DISCLOSE
MAX_NAME_TRIALS = 5   # a drug name shared by more of the firm's trials is too vague to say which trial an 8-K is about
if registry_events is not None:
    precise = links[(links["link_type"] == "nct") | (links["n_name_trials"] <= MAX_NAME_TRIALS)]
    fl = filings[["accession", "cik", "event_date"]].merge(precise[["accession", "nct_id"]].drop_duplicates(), on="accession")
    m = registry_events.reset_index().rename(columns={"index": "eid"})[["eid", "nct_id", "registry_date"]].merge(
        fl[["nct_id", "event_date"]], on="nct_id", how="left")
    gap = (m["event_date"] - m["registry_date"]).dt.days
    after = gap.where(gap.between(0, 365))
    before = gap.between(-30, -1)
    agg = pd.DataFrame({"eid": m["eid"], "after": after, "before": before}).groupby("eid").agg(
        days_to_8k=("after", "min"), disclosed_before=("before", "any"))
    registry_events = registry_events.join(agg)
    registry_events["disclosed_before"] = registry_events["disclosed_before"].fillna(False).astype(bool)
    same = registry_events[["cik", "registry_date"]].reset_index().merge(
        filings[["cik", "event_date"]], on="cik", how="inner")
    near = same[(same["event_date"] - same["registry_date"]).dt.days.between(-1, 1)]["index"].unique()
    registry_events["same_day_8k"] = registry_events.index.isin(near)
    # 8-Ks the firm filed within a year either side: foreign filers (6-K / 20-F) have none, so they cannot disclose by 8-K
    by_firm = {c: np.sort(g.dropna().to_numpy()) for c, g in filings.groupby("cik")["event_date"]}
    n8k = np.zeros(len(registry_events), dtype=int)
    rd = registry_events["registry_date"].to_numpy()
    for c, idx in registry_events.groupby("cik").indices.items():
        d = by_firm.get(int(c))
        if d is not None:
            n8k[idx] = (np.searchsorted(d, rd[idx] + np.timedelta64(365, "D"), side="right")
                        - np.searchsorted(d, rd[idx] - np.timedelta64(365, "D"), side="left"))
    registry_events["firm_8ks_year"] = n8k
    out = registry_events.drop(columns=["prev_snap"])
    out.to_parquet(os.path.join(FEAT_DIR, f"registry_events{'_dryrun' if DRY_RUN else ''}.parquet"), index=False)

    focus = registry_events[registry_events["is_stop"] | ((registry_events["event_type"] == "pcd_later")
                                                          & registry_events["detail"].str.extract(r"(-?\d+)")[0].astype(float).ge(6))]
    print(f"Changes with an 8-K mentioning the trial within a year after: {registry_events['days_to_8k'].notna().mean():.1%}"
          + ("  (dry run: only part of the 8-K text was read)" if DRY_RUN else ""))
    print("\nStops (terminated/suspended/withdrawn) and delays of 6+ months:")
    print(f"  {len(focus):,} events | 8-K within a year: {focus['days_to_8k'].notna().mean():.1%} | "
          f"median days to it: {focus['days_to_8k'].median():.0f} | mentioned in the 30 days before: {focus['disclosed_before'].mean():.1%}")


# %% [markdown]
# ## Step 6 — Readout list
# Every 8-K flagged as a readout, with its market date and ticker, saved as `readouts.parquet`; the ones from the last `OCC_YEARS` years go to `readouts_occ_window.csv` for the OCC options-volume download (`occ_window(ticker, event_date)`).


# %% READOUTS
ro = filings.merge(text_feat.drop(columns=["cik"]), on="accession")
ro = ro[ro["readout"]].copy()
ro["ticker"] = ro["cik"].map(cik2ticker)
ro["company"] = ro["cik"].map(cik2name).fillna(ro["company_header"])
keep = ["accession", "cik", "company", "ticker", "acceptance_et", "event_date", "timing", "items", "sic_ok", "readout_kind", "readout_rule",
        "readout_tone", "pos_hits", "neg_hits", "nct_ids", "fda_approval", "fda_crl", "fda_adcom", "fda_pdufa", "text_source",
        "headline", "readout_snippet"]
ro = ro[keep].sort_values("acceptance_et")
ro.to_parquet(os.path.join(FEAT_DIR, f"readouts{'_dryrun' if DRY_RUN else ''}.parquet"), index=False)
oldest = pd.Timestamp(dt.date.today()) - pd.DateOffset(years=OCC_YEARS) + pd.Timedelta(days=3)
occ = ro[(ro["event_date"] >= oldest) & ro["ticker"].notna()][["ticker", "event_date", "accession", "readout_kind", "readout_tone"]]
occ.to_csv(os.path.join(FEAT_DIR, f"readouts_occ_window{'_dryrun' if DRY_RUN else ''}.csv"), index=False)
print(f"{len(ro):,} readout 8-Ks from {ro['cik'].nunique():,} firms | in the OCC window with a ticker: {len(occ):,}")


# %% [markdown]
# ## Step 7 — Check the results
# Check this cell's output.


# %% SUMMARY
print(f"Run: {'DRY RUN' if DRY_RUN else 'FULL'} | Step 3 rules: {TEXT_VERSION}")
f = filings.merge(text_feat.drop(columns=["cik"]), on="accession")
print(f"8-Ks with text read: {len(f):,} of {len(filings):,} | years: {f['event_date'].dt.year.min()}-{f['event_date'].dt.year.max()}")
print("Document kinds (share of filings containing each):")
kinds = f["kinds"].str.split(";").explode()
print((kinds.value_counts() / len(f)).round(3).to_string())
print("Text read for readouts (first choice):\n" + f["text_source"].value_counts(normalize=True).round(3).to_string())
r = f[f["readout"]]
print(f"\nReadouts: {len(r):,} ({f['readout'].mean():.1%} of filings) | with an NCT number: {(r['n_nct'] > 0).mean():.0%}")
print("Found by:\n" + r["readout_rule"].value_counts().to_string())
print("Kind x tone:\n" + pd.crosstab(r["readout_kind"], r["readout_tone"], margins=True).to_string())
print("Per year:\n" + r.groupby(r["event_date"].dt.year).size().to_string())
print("Timing:\n" + r["timing"].value_counts(normalize=True).round(3).to_string())
print(f"Readouts in 8-Ks with item 2.02 (earnings): {r['items'].str.contains('2.02', regex=False).sum():,}")
print("\nFDA news in the headline area:", {k: int(f[k].sum()) for k in FDA})
lk = links.merge(filings[["accession", "cik"]], on="accession").merge(
    trial_firm[["nct_id", "cik"]].drop_duplicates().assign(own=True), on=["nct_id", "cik"], how="left")
prec = (lk["link_type"] == "nct") | (lk["n_name_trials"] <= 5)
print(f"Trial links: {len(lk):,} ({lk['link_type'].value_counts().to_dict()}) | precise (NCT, or a name with at most 5 trials): "
      f"{prec.sum():,} | filings with a precise link: {lk.loc[prec, 'accession'].nunique() / len(f):.1%}")
print(f"Links where the trial belongs to the filing firm: {lk['own'].eq(True).mean():.0%}")
top = links[links["link_type"] == "drug_name"].groupby("matched").agg(filings=("accession", "nunique"), trials=("n_name_trials", "max"))
print("Drug names linking the most filings (check none is a generic word):\n"
      + top.sort_values("filings", ascending=False).head(12).to_string())
for kind, tone in [("topline_or_endpoint", "positive"), ("topline_or_endpoint", "negative"), ("topline_or_endpoint", "unclear"),
                   ("data_update", None)]:
    ex = r[(r["readout_kind"] == kind) & ((r["readout_tone"] == tone) if tone else True)]
    for x in ex.sample(min(2, len(ex)), random_state=2).itertuples():
        print(f"\n[{kind} | {x.readout_tone} | {x.readout_rule}] {x.company_header} | {str(x.event_date)[:10]} | items {x.items or '-'}"
              f"\n  HEADLINE: {x.headline[:200]}\n  MATCH: {x.readout_snippet[:300]}")
if registry_events is not None:
    print("\nRegistry changes by type and year:")
    print(pd.crosstab(registry_events["event_type"], registry_events["registry_date"].dt.year).to_string())
    stops = registry_events[registry_events["is_stop"]]
    print(f"\nStops: {len(stops):,} (plus {int((registry_events['stop_any'] & registry_events['stale_record']).sum()):,} "
          f"clean-ups of stale records, left out) | 8-K mentioning the trial within a year: {stops['days_to_8k'].notna().mean():.0%} | "
          f"median days: {stops['days_to_8k'].median():.0f} | 8-K by the firm within a day: {stops['same_day_8k'].mean():.0%}")
    print((stops["status_from"] + " -> " + stops["status_to"]).value_counts().head(6).to_string())
    print(f"Stops at firms with no 8-K within a year (foreign filers, private then): {(stops['firm_8ks_year'] == 0).mean():.0%}")
    stops = stops[stops["firm_8ks_year"] > 0]
    size = pd.cut(stops["firm_lead_trials"], [-1, 0, 20, 100, np.inf], labels=["no lead trials", "1-20", "21-100", "100+"])
    print("Firms filing 8-Ks, by how many trials the firm leads:\n" + stops.groupby(size, observed=False).agg(
        stops=("nct_id", "size"), with_8k_within_year=("days_to_8k", lambda s: s.notna().mean()),
        firm_8k_within_day=("same_day_8k", "mean")).round(2).to_string())
    small = stops[stops["firm_lead_trials"].between(1, 20)].copy()
    small["firm"] = small["cik"].astype("float").map({float(k): v for k, v in cik2name.items()})
    print("Stops at 8-K filers leading 1-20 trials (sample):")
    print(small.sample(min(8, len(small)), random_state=3).sort_values("registry_date")[
        ["nct_id", "firm", "registry_date", "detail", "phase", "firm_8ks_year", "days_to_8k", "same_day_8k"]].to_string(index=False))


# %% [markdown]
# ## What next
# - **After the dry run:** check Step 7's output, then set `DRY_RUN = False` and *Run all*.
# - **OCC:** run `occ_window(ticker, event_date)` for each row of `readouts_occ_window.csv` in the scratch notebook.
# - **Next notebook:** daily stock prices and the first tests (realized volatility around readouts; registry changes before disclosure).

"""Notebook 1 — Download biopharma 8-K press releases from EDGAR

Pipeline stage of the who-knows-first replication code. Cells are separated by '# %%' markers, so the file
runs top to bottom as a script and opens as a notebook in Jupyter, VS Code or (after tools/make_notebooks.py) Colab.
See README.md for inputs, outputs and the run order.
"""


# %% [markdown]
# # Notebook 1 — Download biopharma 8-K press releases from EDGAR
#
# Part of the biopharma text-implied volatility pilot. This notebook:
#
# 1. Finds every SEC filer whose industry code (SIC) is 2834, 2835 or 2836: pharmaceutical preparations, diagnostics and biologics.
# 2. Lists their 8-K filings from 2010 to 2025.
# 3. Downloads each filing once and keeps the 8-K's own text plus its press releases (exhibits 99.x), along with the exact time the SEC accepted the filing.
# 4. Saves everything to Google Drive as small Parquet files.
#
# **How to run**
#
# - **Runtime:** CPU is enough; no GPU needed.
# - **Edit `USER_AGENT`** in the first code cell: your name and email. The SEC refuses scripts that don't identify themselves.
# - **First run:** leave `DRY_RUN = True` and choose *Runtime → Run all*. It fetches 200 random filings into a test folder. Most of the time goes to a one-time download and scan of the SEC's bulk company file. Check the output of the last cell.
# - **Full run:** set `DRY_RUN = False` and *Run all* again. Step 2 prints a time estimate; expect many hours. Keep the browser tab open.
# - **If Colab disconnects:** reconnect and *Run all*. Finished filings are skipped, so it resumes where it stopped.
#
# **Output** (in `MyDrive/biopharma_tiv/edgar/`): the firm universe, the filing list, and `8k_text/shards/` with one `meta_*.parquet` (one row per filing) and one `docs_*.parquet` (one row per document) per batch.


# %% CONFIG
# ==== Settings: edit USER_AGENT; leave the rest unless you have a reason ====
import os
USER_AGENT = os.environ.get("SEC_USER_AGENT", "Your Name your.email@example.com")   # the SEC requires a real name and email

SIC_CODES    = {"2834", "2835", "2836"}   # pharma preparations, diagnostics, biologics
INCLUDE_8731 = False                       # also add SIC 8731 (commercial biological research; many are not drug developers)
START_DATE   = "2010-01-01"
END_DATE     = "2025-12-31"
FORMS        = {"8-K"}                     # add "8-K/A" to include amendments

DRY_RUN   = True    # first run: True (200 random filings into a test folder). Then set to False.
DRY_RUN_N = 200

MAX_RPS   = 8       # requests per second; the SEC allows at most 10
N_WORKERS = 4       # parallel downloads (the rate limit above still applies)
CHUNK     = 500     # filings per saved batch

PROJECT_DIR = os.environ.get("WKF_PROJECT_DIR", "/content/drive/MyDrive/biopharma_tiv")


# %% [markdown]
# ## Step 0 — Setup
# Mounts Google Drive (approve the pop-up) and creates the output folders.


# %% SETUP
import os, re, io, json, time, gzip, glob, zipfile, random, threading, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import requests
import pandas as pd
from bs4 import BeautifulSoup
import warnings
try:
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:
    pass

assert "@" in USER_AGENT and "Your Name" not in USER_AGENT, "Set USER_AGENT to your real name and email first (first code cell)."
if INCLUDE_8731:
    SIC_CODES = set(SIC_CODES) | {"8731"}

try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    pass  # not in Colab

START, END = pd.Timestamp(START_DATE), pd.Timestamp(END_DATE)
TAG = "sic" + "-".join(sorted(SIC_CODES))                       # names files by universe definition
EDGAR_DIR = os.path.join(PROJECT_DIR, "edgar")
OUT_DIR   = os.path.join(EDGAR_DIR, "8k_text_dryrun" if DRY_RUN else "8k_text")
SHARD_DIR = os.path.join(OUT_DIR, "shards")
LOCAL_TMP = "/content/tmp" if os.path.isdir("/content") else "/tmp/tiv_tmp"
for d in (EDGAR_DIR, OUT_DIR, SHARD_DIR, LOCAL_TMP):
    os.makedirs(d, exist_ok=True)
print("Saving to:", OUT_DIR)
print("Universe SIC codes:", sorted(SIC_CODES), "| forms:", sorted(FORMS), "|", START_DATE, "to", END_DATE)


# %% HTTP
# ---- A polite SEC client: declared User-Agent, global rate limit, retries with backoff ----
class FatalSECBlock(Exception):
    pass

class SECClient:
    def __init__(self, user_agent, max_rps=8):
        self.ua = user_agent
        self.min_gap = 1.0 / max_rps
        self._lock = threading.Lock()
        self._next = 0.0
        self._local = threading.local()
        self.consecutive_fail = 0
        self.n_requests = 0

    def _session(self):
        s = getattr(self._local, "s", None)
        if s is None:
            s = requests.Session()
            s.headers.update({"User-Agent": self.ua, "Accept-Encoding": "gzip, deflate"})
            self._local.s = s
        return s

    def _wait_turn(self):
        with self._lock:
            t = max(time.monotonic(), self._next)
            self._next = t + self.min_gap
        delay = t - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def _fetch(self, url, stream, timeout):
        return self._session().get(url, timeout=timeout, stream=stream)

    def get(self, url, stream=False, timeout=60, max_tries=6):
        last = None
        for attempt in range(max_tries):
            self._wait_turn()
            try:
                r = self._fetch(url, stream, timeout)
                self.n_requests += 1
            except requests.RequestException as e:
                last = f"{type(e).__name__}: {e}"
            else:
                if r.status_code in (200, 404):
                    if r.status_code == 200:
                        with self._lock:
                            self.consecutive_fail = 0
                    return r                      # the caller decides what a 404 means
                if r.status_code == 403 and not stream and "Undeclared Automated Tool" in r.text:
                    raise FatalSECBlock("The SEC refused the request because it doesn't accept the User-Agent. "
                                        "Put your real name and email in USER_AGENT and run again.")
                last = f"HTTP {r.status_code}"
            with self._lock:
                self.consecutive_fail += 1
                pause = self.consecutive_fail >= 25
                if pause:
                    self.consecutive_fail = 0
            if pause:
                print("Many failed requests in a row: pausing 10 minutes in case the SEC is throttling this machine.")
                time.sleep(600)
            time.sleep(min(60, 2 ** attempt + random.random()))
        raise RuntimeError(f"GET failed after {max_tries} tries: {url} ({last})")

SEC = SECClient(USER_AGENT, MAX_RPS)
print("SEC client ready:", USER_AGENT, f"| max {MAX_RPS} requests/second")


# %% [markdown]
# ## Step 1 — Firm universe
# Downloads the SEC's bulk company file (`submissions.zip`, a large file, saved on the Colab machine rather than Drive) and keeps every filer with the chosen SIC codes. SIC is the code the SEC holds today, so a firm that changed industry code is classified by its current one.
#
# If the bulk file can't be downloaded, a slower fallback lists 8-K filers from the SEC's quarterly indexes and looks each one up (roughly an hour, cached on Drive).
#
# Skipped automatically if the universe file already exists.


# %% UNIVERSE
SUB_ZIP_URL   = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"
ZIP_PATH      = os.path.join(LOCAL_TMP, "submissions.zip")
SUB_CACHE     = os.path.join(EDGAR_DIR, "submissions_json")      # used only by the fallback
UNIVERSE_PATH = os.path.join(EDGAR_DIR, f"universe_{TAG}.parquet")

def download_file(url, path):
    if os.path.exists(path) and zipfile.is_zipfile(path):
        print("Already downloaded:", path)
        return path
    r = SEC.get(url, stream=True, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    total = int(r.headers.get("Content-Length") or 0)
    got, t0, next_report = 0, time.time(), 0
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        for block in r.iter_content(8 << 20):
            f.write(block)
            got += len(block)
            if got >= next_report:
                pct = f"{100 * got / total:.0f}%" if total else ""
                print(f"  {got / 1e9:.2f} GB {pct} ({time.time() - t0:.0f}s)")
                next_report += 500_000_000
    os.replace(tmp, path)
    if not zipfile.is_zipfile(path):
        raise RuntimeError("Downloaded file is not a valid zip")
    return path

class SubmissionsSource:
    """EDGAR submissions JSON, read from the bulk zip when present, else from data.sec.gov (cached on Drive)."""
    def __init__(self, zip_path=None, cache_dir=None):
        self.zf = zipfile.ZipFile(zip_path) if zip_path and os.path.exists(zip_path) else None
        self.names = set(self.zf.namelist()) if self.zf else set()
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def close(self):
        if self.zf is not None:
            self.zf.close()
            self.zf, self.names = None, set()

    def get(self, name):
        if name in self.names:
            return json.loads(self.zf.read(name))
        p = os.path.join(self.cache_dir, name) if self.cache_dir else None
        if p and os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        r = SEC.get("https://data.sec.gov/submissions/" + name)
        if r.status_code != 200:
            return None
        obj = r.json()
        if p:
            with open(p, "w") as f:
                json.dump(obj, f)
        return obj

SIC_RE = re.compile(rb'"sic"\s*:\s*"(\d*)"')

def universe_names_from_zip(zip_path, sic_codes):
    hits = []
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if re.fullmatch(r"CIK\d{10}\.json", n)]
        print(f"Scanning {len(names):,} filers for SIC codes {sorted(sic_codes)} ...")
        for i, n in enumerate(names):
            with zf.open(n) as fh:
                m = SIC_RE.search(fh.read(4096))       # "sic" sits near the top of each file
            if m is None:
                m = SIC_RE.search(zf.read(n))
            if m and m.group(1).decode() in sic_codes:
                hits.append(n)
            if i and i % 200_000 == 0:
                print(f"  {i:,} scanned, {len(hits):,} matches")
    return hits

def parse_master_idx(text):
    """Rows of an EDGAR quarterly master index: (cik, name, form, date_filed, filename)."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("CIK|Company Name|")), None)
    if start is None:
        return
    for l in lines[start + 1:]:
        if not l or l.startswith("-"):
            continue
        parts = l.split("|")
        if len(parts) < 5:
            continue
        yield int(parts[0]), "|".join(parts[1:-3]), parts[-3], parts[-2], parts[-1]

def fetch_master_index(year, q):
    base = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/"
    r = SEC.get(base + "master.gz")
    if r.status_code == 200:
        return gzip.decompress(r.content).decode("latin-1")
    r = SEC.get(base + "master.idx")
    return r.content.decode("latin-1") if r.status_code == 200 else ""

def universe_names_fallback(source):
    ciks = set()
    for y in range(START.year, END.year + 1):
        for q in range(1, 5):
            if pd.Timestamp(year=y, month=3 * q - 2, day=1) > min(END, pd.Timestamp.today()):
                continue
            n_before = len(ciks)
            for cik, _, form, date, _ in parse_master_idx(fetch_master_index(y, q)):
                if form in FORMS and START <= pd.Timestamp(date) <= END:
                    ciks.add(cik)
            print(f"  {y} Q{q}: {len(ciks) - n_before:,} new 8-K filers (total {len(ciks):,})")
    print(f"Looking up SIC codes for {len(ciks):,} filers (about {len(ciks) / MAX_RPS / 60:.0f} minutes; cached) ...")
    hits = []
    for i, cik in enumerate(sorted(ciks)):
        name = f"CIK{cik:010d}.json"
        obj = source.get(name)
        if obj and str(obj.get("sic", "")) in SIC_CODES:
            hits.append(name)
        if i and i % 1000 == 0:
            print(f"  {i:,} looked up, {len(hits):,} matches")
    return hits

def universe_row(obj):
    return {
        "cik": int(obj.get("cik")),
        "name": obj.get("name"),
        "sic": str(obj.get("sic", "")),
        "sic_description": obj.get("sicDescription"),
        "entity_type": obj.get("entityType"),
        "tickers": ";".join(obj.get("tickers") or []),
        "exchanges": ";".join(x for x in (obj.get("exchanges") or []) if x),
        "state_of_incorporation": obj.get("stateOfIncorporation"),
        "former_names": ";".join(f.get("name", "") for f in (obj.get("formerNames") or [])),
    }

route = None
if os.path.exists(UNIVERSE_PATH):
    universe = pd.read_parquet(UNIVERSE_PATH)
    print(f"Loaded universe: {len(universe):,} filers ({UNIVERSE_PATH})")
else:
    try:
        print("Downloading", SUB_ZIP_URL)
        download_file(SUB_ZIP_URL, ZIP_PATH)
        route = "zip"
    except FatalSECBlock:
        raise
    except Exception as e:
        print("Bulk download failed:", e, "\nUsing the slower fallback.")
        route = "api"
    source = SubmissionsSource(ZIP_PATH if route == "zip" else None, SUB_CACHE)
    names = universe_names_from_zip(ZIP_PATH, SIC_CODES) if route == "zip" else universe_names_fallback(source)
    universe = pd.DataFrame([universe_row(source.get(n)) for n in names])
    universe.to_parquet(UNIVERSE_PATH, index=False)
    print(f"Universe: {len(universe):,} filers saved to {UNIVERSE_PATH}")
print(universe["sic"].value_counts().to_string())


# %% [markdown]
# ## Step 2 — Filing list
# Lists every 8-K each universe firm filed in the date window, from the same company files, and prints a time estimate for the download. Skipped if the list already exists.


# %% FILINGS
FILING_INDEX_PATH = os.path.join(
    EDGAR_DIR, f"filing_index_{TAG}_{'-'.join(sorted(FORMS)).replace('/', '')}_{START_DATE}_{END_DATE}.parquet")
FIELDS = ["accessionNumber", "filingDate", "acceptanceDateTime", "form", "items",
          "primaryDocument", "primaryDocDescription", "size"]

def _columns(block):
    n = len(block.get("accessionNumber", []))
    return {k: list(block.get(k, [None] * n)) for k in FIELDS}

def filings_from_submission(obj, source):
    blocks = [obj.get("filings", {}).get("recent", {})]
    for f in obj.get("filings", {}).get("files", []):
        extra = source.get(f["name"])
        if extra:
            blocks.append(extra["filings"].get("recent", {}) if "filings" in extra else extra)
    df = pd.concat([pd.DataFrame(_columns(b)) for b in blocks], ignore_index=True)
    df.insert(0, "cik", int(obj["cik"]))
    return df

if os.path.exists(FILING_INDEX_PATH):
    filing_index = pd.read_parquet(FILING_INDEX_PATH)
    print(f"Loaded filing list: {len(filing_index):,} rows ({FILING_INDEX_PATH})")
else:
    if "source" not in globals() or source is None:
        source = SubmissionsSource(ZIP_PATH if os.path.exists(ZIP_PATH) else None, SUB_CACHE)
    frames = []
    for i, cik in enumerate(universe["cik"]):
        obj = source.get(f"CIK{int(cik):010d}.json")
        if obj:
            frames.append(filings_from_submission(obj, source))
        if i and i % 500 == 0:
            print(f"  {i:,} / {len(universe):,} firms")
    allf = pd.concat(frames, ignore_index=True)
    allf["filingDate"] = pd.to_datetime(allf["filingDate"], errors="coerce")
    keep = allf["form"].isin(FORMS) & allf["filingDate"].between(START, END)
    filing_index = (allf[keep]
                    .rename(columns={"accessionNumber": "accession", "filingDate": "filing_date",
                                     "acceptanceDateTime": "acceptance_json", "items": "items_json",
                                     "primaryDocument": "primary_doc", "primaryDocDescription": "primary_doc_desc"})
                    .drop_duplicates(["cik", "accession"])
                    .sort_values(["filing_date", "cik"])
                    .reset_index(drop=True))
    filing_index.to_parquet(FILING_INDEX_PATH, index=False)
    print(f"Filing list saved: {FILING_INDEX_PATH}")
assert len(filing_index), "No filings found: check SIC_CODES, FORMS and the date window."
if "source" in globals() and source is not None:
    source.close()                 # release the bulk zip (Windows keeps open files locked)

n_acc = filing_index["accession"].nunique()
print(f"{n_acc:,} filings from {filing_index['cik'].nunique():,} firms")
print(filing_index.groupby(filing_index["filing_date"].dt.year).size().rename("8-Ks per year").to_string())
print(f"Estimated full download time at ~{0.8 * MAX_RPS:.0f} filings/second: {n_acc / (0.8 * MAX_RPS) / 3600:.1f} hours")


# %% [markdown]
# ## Step 3 — Parsing functions
# Each filing is one SEC text file holding a header and every document. We keep:
# - the header: acceptance time (Eastern time), the 8-K item list and the SIC code at filing;
# - the 8-K's own text and every exhibit 99.x (press releases), converted from HTML to plain text.
#
# Images, XBRL and other exhibits are dropped. The cell ends with a small self-test.


# %% PARSE
DOC_RE     = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.S | re.I)
TEXT_RE    = re.compile(r"<TEXT>(.*?)</TEXT>", re.S | re.I)
HTML_HINT  = re.compile(r"<(html|body|div|p|table|font|br|span)\b", re.I)
HIDDEN     = re.compile(r"display\s*:\s*none", re.I)
BLOCK_TAGS = ["p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "center", "pre"]
BREAK      = ""   # placeholder for a real line break while source whitespace is collapsed

def decode(raw):
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")

def clean_text(text):
    text = text.replace("\xa0", " ").replace("​", "")
    out, blank = [], False
    for line in text.splitlines():
        line = re.sub(r"[ \t\r\f\v]+", " ", line).strip()
        if line:
            out.append(line)
            blank = False
        elif out and not blank:
            out.append("")
            blank = True
    return "\n".join(out).strip()

def html_to_text(html):
    soup = BeautifulSoup(html, "lxml")
    for t in soup.find_all(["script", "style", "head"]):
        t.decompose()
    for t in soup.find_all(lambda x: x.name is not None and x.name.lower() in ("ix:header", "header")):
        t.decompose()
    for t in soup.find_all(style=HIDDEN):
        t.decompose()
    for t in soup.find_all(["td", "th"]):
        t.append(" ")
    for t in soup.find_all(BLOCK_TAGS):
        t.append(BREAK)
    text = soup.get_text("")
    text = re.sub(r"[ \t\r\n\f\v\xa0]+", " ", text)   # line breaks in the HTML source are not real breaks
    return clean_text(text.replace(BREAK, "\n"))

def _tag(block, tag):
    m = re.search(rf"<{tag}>([^\n<]*)", block, re.I)
    return m.group(1).strip() if m else ""

def _hdr(pattern, hdr):
    m = re.search(pattern, hdr)
    return m.group(1).strip() if m else None

def parse_submission(raw, keep_forms):
    text = decode(raw)
    m = re.search(r"<SEC-HEADER>(.*?)</SEC-HEADER>", text, re.S | re.I)
    hdr = m.group(1) if m else text[:20000]
    sic = re.search(r"STANDARD INDUSTRIAL CLASSIFICATION:\s*(.*?)\[(\d{4})\]", hdr)
    meta = {
        "acceptance_datetime": _hdr(r"<ACCEPTANCE-DATETIME>\s*(\d{14})", text[:5000]),
        "form_header": _hdr(r"CONFORMED SUBMISSION TYPE:\s*(\S+)", hdr),
        "filed_as_of": _hdr(r"FILED AS OF DATE:\s*(\d{8})", hdr),
        "period_of_report": _hdr(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})", hdr),
        "items_header": "; ".join(s.strip() for s in re.findall(r"ITEM INFORMATION:\s*(.+)", hdr)),
        "company_header": _hdr(r"COMPANY CONFORMED NAME:\s*(.+)", hdr),
        "sic_header": sic.group(2) if sic else None,
    }
    keep = {f.upper() for f in keep_forms}
    docs, n_total = [], 0
    for dm in DOC_RE.finditer(text):
        n_total += 1
        block = dm.group(1)
        dtype = _tag(block, "TYPE").upper()
        if not (dtype in keep or dtype.startswith("EX-99")):
            continue
        tm = TEXT_RE.search(block)
        body = re.sub(r"</?XBRL>", "", tm.group(1) if tm else block, flags=re.I)
        if body.lstrip()[:6].lower().startswith("begin "):      # uuencoded binary
            continue
        is_html = bool(HTML_HINT.search(body[:20000]))
        txt = html_to_text(body) if is_html else clean_text(re.sub(r"</?PAGE>", "", body, flags=re.I))
        docs.append({
            "doc_type": dtype,
            "sequence": _tag(block, "SEQUENCE"),
            "filename": _tag(block, "FILENAME"),
            "description": _tag(block, "DESCRIPTION"),
            "is_html": is_html,
            "n_chars": len(txt),
            "text": txt,
        })
    meta["n_docs_total"] = n_total
    meta["n_ex99"] = sum(d["doc_type"].startswith("EX-99") for d in docs)
    return meta, docs

# ---- self-test on a tiny made-up filing ----
_demo = b"""<SEC-DOCUMENT>0000000000-20-000001.txt : 20200102
<SEC-HEADER>0000000000-20-000001.hdr.sgml : 20200102
<ACCEPTANCE-DATETIME>20200102163005
CONFORMED SUBMISSION TYPE:\t8-K
ITEM INFORMATION:\t\tOther Events
FILED AS OF DATE:\t\t20200102
\t\tSTANDARD INDUSTRIAL CLASSIFICATION:\tPHARMACEUTICAL PREPARATIONS [2834]
</SEC-HEADER>
<DOCUMENT>
<TYPE>8-K
<SEQUENCE>1
<FILENAME>d1.htm
<TEXT>
<html><body><div style="display:none"><ix:header>hidden facts</ix:header></div><p>Item 8.01 Other Events.</p></body></html>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-99.1
<SEQUENCE>2
<FILENAME>ex991.htm
<DESCRIPTION>PRESS RELEASE
<TEXT>
<html><body><p>Acme&nbsp;Bio announces <b>positive</b>
Phase&nbsp;3 topline
data.</p><table><tr><td>Revenue</td><td>100</td></tr></table></body></html>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>GRAPHIC
<SEQUENCE>3
<FILENAME>logo.jpg
<TEXT>
begin 644 logo.jpg
M_]C_X``02D9)1@`!`0$`8`!@``#_VP!#``,"`@,"`@,#`P,$`P,$!0@%!00$
end
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>"""
_m, _d = parse_submission(_demo, {"8-K"})
assert _m["acceptance_datetime"] == "20200102163005" and _m["sic_header"] == "2834", _m
assert [d["doc_type"] for d in _d] == ["8-K", "EX-99.1"], _d
assert "hidden facts" not in _d[0]["text"] and "Item 8.01" in _d[0]["text"], _d[0]["text"]
assert "Acme Bio announces positive Phase 3 topline data." in _d[1]["text"], _d[1]["text"]
assert "Revenue 100" in _d[1]["text"], _d[1]["text"]
print("Parser self-test passed.")


# %% [markdown]
# ## Step 4 — Download and parse
# Downloads each filing once, at most `MAX_RPS` requests per second, and saves a batch every `CHUNK` filings. Re-running skips finished filings. A filing that fails three times (on separate runs) is skipped and listed in `failures.csv`.
#
# If the SEC throttles this machine (repeated HTTP 403 or 429), the code pauses and retries. If that keeps happening, *Runtime → Disconnect and delete runtime* gives you a new machine; then *Run all*.


# %% DOWNLOAD
DONE_PATH = os.path.join(OUT_DIR, "done_accessions.txt")
FAIL_PATH = os.path.join(OUT_DIR, "failures.csv")
acc2cik = filing_index.drop_duplicates("accession").set_index("accession")["cik"].to_dict()

def filing_url(cik, acc):
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{acc}.txt"

def load_done():
    if not os.path.exists(DONE_PATH):
        return set()
    with open(DONE_PATH) as f:
        return {l.strip() for l in f if l.strip()}

def load_fail_counts():
    if not os.path.exists(FAIL_PATH):
        return {}
    return pd.read_csv(FAIL_PATH)["accession"].value_counts().to_dict()

def write_parquet_atomic(df, path):
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)

def next_shard_id():
    ids = [int(re.search(r"(\d+)", os.path.basename(p)).group(1)) for p in glob.glob(os.path.join(SHARD_DIR, "meta_*.parquet"))]
    return max(ids, default=0) + 1

def process_one(acc):
    cik = acc2cik[acc]
    try:
        r = SEC.get(filing_url(cik, acc))
        if r.status_code != 200:
            return acc, None, None, f"HTTP {r.status_code}"
        meta, docs = parse_submission(r.content, FORMS)
        meta.update(accession=acc, cik=int(cik), raw_bytes=len(r.content),
                    fetched_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
        for d in docs:
            d["accession"] = acc
        return acc, meta, docs, None
    except FatalSECBlock:
        raise
    except Exception as e:
        return acc, None, None, f"{type(e).__name__}: {e}"[:300]

done, fail_counts = load_done(), load_fail_counts()
todo = [a for a in acc2cik if a not in done and fail_counts.get(a, 0) < 3]
if DRY_RUN:
    todo = random.Random(0).sample(todo, min(DRY_RUN_N, len(todo)))
print(f"{len(done):,} filings already done; {len(todo):,} to download now" + (" (dry run sample)" if DRY_RUN else ""))

t0, n_ok, n_fail = time.time(), 0, 0
try:
    with ThreadPoolExecutor(N_WORKERS) as ex:
        for start in range(0, len(todo), CHUNK):
            chunk = todo[start:start + CHUNK]
            metas, docs, fails = [], [], []
            for acc, meta, dlist, err in ex.map(process_one, chunk):
                if err:
                    fails.append({"accession": acc, "error": err,
                                  "when_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
                else:
                    metas.append(meta)
                    docs.extend(dlist)
            sid = next_shard_id()
            if metas:
                write_parquet_atomic(pd.DataFrame(metas), os.path.join(SHARD_DIR, f"meta_{sid:05d}.parquet"))
            if docs:
                write_parquet_atomic(pd.DataFrame(docs), os.path.join(SHARD_DIR, f"docs_{sid:05d}.parquet"))
            with open(DONE_PATH, "a") as f:
                f.writelines(m["accession"] + "\n" for m in metas)
            if fails:
                pd.DataFrame(fails).to_csv(FAIL_PATH, mode="a", header=not os.path.exists(FAIL_PATH), index=False)
            n_ok, n_fail = n_ok + len(metas), n_fail + len(fails)
            rate = (n_ok + n_fail) / max(time.time() - t0, 1e-9)
            left = len(todo) - start - len(chunk)
            print(f"batch {sid:05d}: {n_ok:,} saved, {n_fail:,} failed this run | {rate:.1f} filings/s | "
                  f"{left:,} left, about {left / max(rate, 1e-9) / 3600:.1f} h")
except FatalSECBlock as e:
    print("STOPPED:", e)
print(f"Done for now: {n_ok:,} saved and {n_fail:,} failed in {(time.time() - t0) / 60:.1f} minutes.")


# %% [markdown]
# ## Step 5 — Check the results
# Check this cell's output.


# %% SUMMARY
metas = [pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(SHARD_DIR, "meta_*.parquet")))]
if not metas:
    print("Nothing saved yet.")
else:
    meta = pd.concat(metas, ignore_index=True).drop_duplicates("accession")
    acc_ts = pd.to_datetime(meta["acceptance_datetime"], format="%Y%m%d%H%M%S", errors="coerce")
    print(f"Run: {'DRY RUN' if DRY_RUN else 'FULL'} | {len(meta):,} filings from {meta['cik'].nunique():,} firms saved")
    print(f"Missing acceptance time: {acc_ts.isna().mean():.1%} | with at least one exhibit 99: {(meta['n_ex99'] > 0).mean():.1%}")
    print("\nFilings by year:\n" + acc_ts.dt.year.value_counts().sort_index().to_string())
    print("\nAcceptance hour (Eastern), share of filings:\n" + acc_ts.dt.hour.value_counts(normalize=True).sort_index().round(3).to_string())
    j = filing_index.drop_duplicates("accession").set_index("accession")["acceptance_json"]
    jd = pd.to_datetime(meta["accession"].map(j).astype(str).str[:10], errors="coerce")
    print(f"\nHeader date matches the bulk file's acceptance date: {(jd.dt.date == acc_ts.dt.date).mean():.1%}")
    print(f"SIC in header equals universe SIC: {meta['sic_header'].isin(SIC_CODES).mean():.1%}")
    items = meta["items_header"].fillna("").str.split("; ").explode()
    print("\nMost common 8-K items:\n" + items[items != ""].value_counts().head(8).to_string())
    doc_paths = sorted(glob.glob(os.path.join(SHARD_DIR, "docs_*.parquet")))
    if doc_paths:
        docs = pd.concat([pd.read_parquet(p) for p in doc_paths[-3:]], ignore_index=True)
        ex = docs[docs["doc_type"].str.startswith("EX-99")]
        print(f"\nExhibit 99 documents in the last batches: {len(ex):,} | median length {ex['n_chars'].median():,.0f} characters")
        for _, row in ex.sample(min(2, len(ex)), random_state=1).iterrows():
            print(f"\n--- {row['accession']} {row['doc_type']} {row['description']} ---\n{row['text'][:600]}")
    if os.path.exists(FAIL_PATH):
        fl = pd.read_csv(FAIL_PATH)
        print(f"\nFailures logged: {len(fl):,}\n" + fl["error"].str[:60].value_counts().head(5).to_string())
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump({"run": "dry" if DRY_RUN else "full", "sic_codes": sorted(SIC_CODES), "forms": sorted(FORMS),
                   "start": START_DATE, "end": END_DATE, "filings_saved": int(len(meta)),
                   "firms": int(meta["cik"].nunique()), "written_utc": dt.datetime.now(dt.timezone.utc).isoformat()}, f, indent=2)


# %% [markdown]
# ## What next
# - **After the dry run:** check Step 5's output. If it looks right, set `DRY_RUN = False` and *Run all*. Steps 1 and 2 are skipped because their files are saved.
# - **During the full run:** progress prints after every batch. If Colab disconnects, reconnect and *Run all*; it resumes.
# - **Next notebook:** ClinicalTrials.gov version histories for the same firms.

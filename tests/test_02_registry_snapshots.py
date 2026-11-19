"""Offline test of the merged Notebook 2 (matching + AACT monthly snapshots)."""
import io, os, re, sys, glob, json, shutil, zipfile, contextlib, types
import pandas as pd
import requests as real_requests

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "tools"))
from cells import load_stage
C = load_stage("02_registry_snapshots")
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _fixtures_02 as T          # reuse firm list, ClinicalTrials.gov API fixtures and runner pieces

FIX = "/tmp/aact_fixtures"
shutil.rmtree(FIX, ignore_errors=True)
os.makedirs(FIX)

NEW_STUDY_COLS = ["nct_id", "nlm_download_date_description", "study_first_submitted_date", "results_first_submitted_date",
                  "last_update_submitted_date", "start_date_type", "start_month_year", "start_date", "completion_date_type", "completion_date",
                  "primary_completion_date_type", "primary_completion_month_year", "primary_completion_date", "study_type", "brief_title",
                  "overall_status", "last_known_status", "phase", "enrollment", "enrollment_type", "why_stopped", "verification_date"]
OLD_STUDY_COLS = ["nct_id", "first_received_date", "last_changed_date", "first_received_results_date", "start_month_year",
                  "completion_month_year", "completion_date_type", "primary_completion_month_year", "primary_completion_date_type",
                  "verification_month_year", "study_type", "brief_title", "overall_status", "phase", "enrollment", "enrollment_type",
                  "why_stopped"]

def studies_txt(cols, rows, bad_line=False):
    lines = ["|".join(cols)]
    for r in rows:
        lines.append("|".join(str(r.get(c, "")) for c in cols))
    if bad_line:
        lines.insert(2, "NCT00000099|x|y|" + "|".join(["z"] * 40))   # too many fields
    return "\n".join(lines) + "\n"

def month_year(d):
    return pd.Timestamp(d).strftime("%B %Y")

def rows_for(snap, status_map, pcd_map, enroll, upper=False, blank_start=()):
    out = []
    for i in range(1, 10):
        n = f"NCT0000000{i}"
        pcd = pcd_map.get(n, "2013-12-15")
        st = status_map.get(n, "Recruiting")
        r = {"nct_id": n, "study_first_submitted_date": "2009-12-20", "last_update_submitted_date": snap,
             "start_date": "" if n in blank_start else "2010-01-15", "start_month_year": "January 2010", "start_date_type": "Actual",
             "primary_completion_date": pcd, "primary_completion_month_year": month_year(pcd),
             "primary_completion_date_type": "Anticipated", "study_type": "Interventional", "brief_title": f"Trial 'quoted' \"x\" {i}",
             "overall_status": st, "phase": "Phase 3", "enrollment": str(enroll), "enrollment_type": "Anticipated",
             "verification_date": snap, "first_received_date": "2009-12-20", "last_changed_date": snap,
             "verification_month_year": month_year(snap), "completion_month_year": "June 2014", "completion_date_type": "Anticipated"}
        if upper:
            r.update(overall_status=st.upper().replace(" ", "_").replace(",", ""), phase="PHASE3", enrollment_type="ESTIMATED",
                     primary_completion_date_type="ESTIMATED", start_date_type="ACTUAL", study_type="INTERVENTIONAL")
        out.append(r)
    return out

def make_zip(name, studies, sponsors, outcomes, folder=""):
    with zipfile.ZipFile(os.path.join(FIX, name), "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(folder + "studies.txt", studies)
        z.writestr(folder + "sponsors.txt", sponsors)
        if outcomes is not None:
            z.writestr(folder + "design_outcomes.txt", outcomes)
        z.writestr(folder + "conditions.txt", "id|nct_id|name\n1|NCT00000001|Cancer\n")

SPONS = ("id|nct_id|agency_class|lead_or_collaborator|name\n"
         "1|NCT00000001|Industry|lead|Acme Bio, Inc.\n2|NCT00000002|Industry|lead|Acme Pharmaceuticals\n"
         "3|NCT00000004|Other|lead|University X\n4|NCT00000004|Industry|collaborator|BigBio Corporation\n"
         "5|NCT00000003|Industry|lead|Pfizer\n6|NCT00000005|Industry|lead|Novartis Pharmaceuticals\n")
OUTC = ("id|nct_id|outcome_type|measure|time_frame\n1|NCT00000001|primary|Overall survival|5 years\n"
        "2|NCT00000001|secondary|PFS|2 years\n3|NCT00000002|primary|ORR|1 year\n")
OUTC_EDIT = OUTC.replace("Overall survival", "Overall survival at 3 years")

# 2017: oldest era (first_received/last_changed names, month-year text dates, no design_outcomes table)
make_zip("20170105_pipe-delimited-export.zip",
         studies_txt(OLD_STUDY_COLS, rows_for("2016-12-20", {"NCT00000001": "Not yet recruiting"}, {"NCT00000001": "2012-06-30"}, 200)),
         SPONS, None)
# 2020: middle era (current names, a blank start_date that start_month_year fills, title-case codes, one bad line)
make_zip("20200101_export_ctgov.zip",
         studies_txt(NEW_STUDY_COLS, rows_for("2019-12-28", {"NCT00000001": "Active, not recruiting"}, {"NCT00000001": "2013-06-30"}, 250,
                                              blank_start={"NCT00000003"}), bad_line=True),
         SPONS, OUTC, folder="20200101_export_ctgov/")
# 2026: newest era (upper-case codes)
make_zip("20260101_export_ctgov.zip",
         studies_txt(NEW_STUDY_COLS, rows_for("2025-12-15", {"NCT00000001": "Completed", "NCT00000002": "Terminated"},
                                              {"NCT00000001": "2013-12-15"}, 350, upper=True)),
         SPONS.replace("2|NCT00000002|Industry|lead|Acme Pharmaceuticals", "2|NCT00000002|Industry|lead|Acme Bio, Inc."), OUTC_EDIT)
for src, dst in [("20170105_pipe-delimited-export.zip", "20180101_pipe-delimited-export.zip"),
                 ("20200101_export_ctgov.zip", "20210101_export_ctgov.zip"), ("20200101_export_ctgov.zip", "20230101_export_ctgov.zip"),
                 ("20260101_export_ctgov.zip", "20240101_export_ctgov.zip"), ("20260101_export_ctgov.zip", "20250101_export_ctgov.zip"),
                 ("20260101_export_ctgov.zip", "20260105_export_ctgov.zip")]:
    shutil.copy(os.path.join(FIX, src), os.path.join(FIX, dst))

API = "https://aact.ctti-clinicaltrials.org/api/v1"
STORE = "https://ctti-aact.nyc3.digitaloceanspaces.com/archive/"
KEY = "test-key-123"
os.environ["AACT_API_KEY"] = KEY
SNAPS = [  # archive of flat snapshots (id, created_at, file)
    (90, "2009-01-01T06:00:00Z", "20090101_export_ctgov.zip"),
    (100, "2017-01-05T06:10:00Z", "20170105_pipe-delimited-export.zip"),
    (110, "2018-01-01T06:10:00Z", "20180101_pipe-delimited-export.zip"),
    (200, "2020-01-01T06:10:00Z", "20200101_export_ctgov.zip"),
    (210, "2021-01-01T06:10:00Z", "20210101_export_ctgov.zip"),
    (230, "2023-01-01T06:10:00Z", "20230101_export_ctgov.zip"),
    (240, "2024-01-01T06:10:00Z", "20240101_export_ctgov.zip"),
    (250, "2025-01-01T06:10:00Z", "20250101_export_ctgov.zip"),
    (300, "2026-01-01T06:10:00Z", "20260101_export_ctgov.zip"),
    (301, "2026-01-05T06:10:00Z", "20260105_export_ctgov.zip"),
]

class FakeResp:
    def __init__(self, status, content=b"", text="", js=None, headers=None):
        self.status_code, self._content, self.text, self._js = status, content, text or json.dumps(js or {}), js
        self.headers = headers or {}
    def json(self):
        if self._js is None:
            raise ValueError("no json")
        return self._js
    def iter_content(self, n):
        for i in range(0, len(self._content), n):
            yield self._content[i:i + n]
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False

class FakeRequests:
    Session = real_requests.Session
    RequestException = real_requests.RequestException
    def __init__(self, snaps=SNAPS, key=KEY, broken=(), busy_once=True):
        self.snaps, self.key, self.broken, self.busy = snaps, key, set(broken), busy_once
        self.downloads, self.api_calls, self.leaked = [], [], False
    def get(self, url, stream=False, timeout=None, headers=None, params=None, allow_redirects=True):
        headers = headers or {}
        if url.startswith(API):
            self.api_calls.append(url)
            if headers.get("Authorization") != "Bearer " + self.key:
                return FakeResp(401, js={"error": {"code": "invalid_api_key", "message": "The API key is missing, invalid, or revoked."}})
            path = url[len(API):]
            if path == "/snapshots":
                return FakeResp(200, js={"data": [{"type": "postgres"}, {"type": "flat"}]})
            if path == "/snapshots/flat/monthly":
                if self.busy:
                    self.busy = False
                    return FakeResp(429, js={"error": {"code": "rate_limited", "message": "slow down"}}, headers={"Retry-After": "3"})
                lo, hi = params["from"], params["to"]
                data = [{"id": i, "type": "flat", "created_at": c, "size_bytes": 2_100_000_000, "extra": "ignored"}
                        for i, c, _ in self.snaps if lo <= c[:10] <= hi]
                return FakeResp(200, js={"data": sorted(data, key=lambda d: d["created_at"], reverse=True)})
            m = re.fullmatch(r"/snapshots/flat/(\d+)/download", path)
            if m:
                assert allow_redirects is False
                name = {i: f for i, _, f in self.snaps}[int(m.group(1))]
                return FakeResp(302, headers={"Location": STORE + name + "?X-Amz-Signature=abc"})
            return FakeResp(404, js={"error": {"code": "not_found", "message": "The path does not exist."}})
        if "Authorization" in headers or KEY in url:
            self.leaked = True
        name = url.split("?")[0].rsplit("/", 1)[-1]
        self.downloads.append(name)
        p = os.path.join(FIX, name)
        if name in self.broken or not os.path.exists(p):
            return FakeResp(403, b"denied")
        return FakeResp(200, open(p, "rb").read())

import re
ALL = [C.CONFIG, C.SETUP, C.HTTP, C.FIRMS, C.CURRENT, C.MATCH, C.FIND_AACT, C.AACT, C.SUMMARY]

def run(P, overrides, fake_req, cells=ALL, quiet=False):
    ns, buf = {}, io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            for src in cells:
                exec(compile(src, "<cell>", "exec"), ns)
                if src is C.CONFIG:
                    ns.update(USER_AGENT="Test User test@example.com", PROJECT_DIR=P, MAX_RPS=1000.0)
                    ns.update(overrides)
                if src is C.SETUP:
                    ns["time"] = T.fast_time
                    ns["requests"] = fake_req
                if src is C.HTTP:
                    ns["CT"]._fetch = T.MockCT()
    except Exception:
        print(buf.getvalue())
        raise
    out = buf.getvalue()
    if not quiet:
        print(out)
    return ns, out

# === A: dry run lists the archive through the API, processes four months across the eras ===
P = "/tmp/nb2m_A"
T.make_project(P)
fr = FakeRequests()
ns, out = run(P, {"DRY_RUN": True}, fr)
m = ns["monthly"]
assert [str(x) for x in m["month"]] == ["2017-01", "2018-01", "2020-01", "2021-01", "2023-01", "2024-01", "2025-01", "2026-01"], m
assert m.loc[m["month"] == pd.Period("2026-01"), "url"].iloc[0] == API + "/snapshots/flat/300/download"
assert "AACT API key works. Snapshot types: postgres, flat" in out
assert "8 monthly flat-file snapshots from 2017-01 to 2026-01 (17 GB in all" in out, out
assert "The archive starts at 2017-01" in out and "Months with no snapshot in the archive" in out
assert sorted(fr.downloads) == ["20170105_pipe-delimited-export.zip", "20200101_export_ctgov.zip",
                                "20240101_export_ctgov.zip", "20260101_export_ctgov.zip"], fr.downloads
assert not fr.leaked, "API key sent to the storage link"
assert KEY not in out
assert "dry run: four months spread over the archive" in out
D = os.path.join(P, "ctgov", "aact_dryrun")
d17 = pd.read_parquet(os.path.join(D, "aact_201701.parquet")).set_index("nct_id")
assert sorted(d17.index) == ["NCT00000001", "NCT00000002", "NCT00000003", "NCT00000004", "NCT00000007", "NCT00000008"], d17.index
r1 = d17.loc["NCT00000001"]
assert r1["overall_status"] == "NOT_YET_RECRUITING" and r1["phase"] == "PHASE3" and r1["primary_completion_date_type"] == "ESTIMATED"
assert r1["primary_completion_date"] == "2012-06-01" and r1["start_date"] == "2010-01-01" and r1["completion_date"] == "2014-06-01"
assert r1["study_first_submitted_date"] == "2009-12-20" and r1["last_update_submitted_date"] == "2016-12-20"
assert r1["verification_date"] == "2016-12-01" and pd.isna(r1["start_date_type"]) and r1["lead_sponsor"] == "Acme Bio, Inc."
assert "n_primary_outcomes" not in d17.columns
d20 = pd.read_parquet(os.path.join(D, "aact_202001.parquet")).set_index("nct_id")
assert d20.loc["NCT00000001", "overall_status"] == "ACTIVE_NOT_RECRUITING" and d20.loc["NCT00000003", "start_date"] == "2010-01-01"
assert d20.loc["NCT00000002", "start_date"] == "2010-01-15" and "NCT00000099" not in d20.index
d26 = pd.read_parquet(os.path.join(D, "aact_202601.parquet")).set_index("nct_id")
assert d26.loc["NCT00000001", "overall_status"] == "COMPLETED" and d26.loc["NCT00000001", "phase"] == "PHASE3"
cl = pd.read_csv(os.path.join(D, "column_log.csv"), dtype=str)
assert ((cl["month"] == "201701") & (cl["column"] == "studies.start_date") & (cl["note"] == "from start_month_year")).any()
assert ((cl["month"] == "201701") & (cl["column"] == "studies.last_update_submitted_date") & (cl["note"] == "from last_changed_date")).any()
assert ((cl["month"] == "202001") & (cl["column"] == "studies.start_date") & (cl["note"] == "from start_month_year")).any()
assert ((cl["month"] == "201701") & (cl["column"] == "studies.last_update_posted_date") & (cl["note"] == "missing")).any()
assert not ((cl["month"] == "202601") & (cl["column"] == "studies.start_date")).any()
hl = pd.read_csv(os.path.join(D, "headers.csv"), dtype=str)
assert "first_received_date" in hl[(hl["month"] == "201701") & (hl["table"] == "studies")]["columns"].iloc[0]
assert "Status codes by snapshot" in out and "Share of rows with each field filled, by snapshot" in out
assert "Columns missing, or filled from an older name" in out and "from last_changed_date" in out
assert "Example trial:" in out
assert not os.path.exists(os.path.join(ns["LOCAL_TMP"], "aact_201701.zip"))
f = ns["find_firm"]("acme")
assert list(f["cik"]) == [1] and bool(f["files_8k"].iloc[0])
print("TEST A OK\n" + "=" * 70)

# === B: full run, then resume ===
fr = FakeRequests()
ns, out = run(P, {"DRY_RUN": False}, fr)
assert len(fr.downloads) == 8 and "20260105_export_ctgov.zip" not in fr.downloads, fr.downloads
assert not fr.leaked
F = os.path.join(P, "ctgov", "aact_monthly")
d20 = pd.read_parquet(os.path.join(F, "aact_202001.parquet")).set_index("nct_id")
assert d20.loc["NCT00000001", "n_primary_outcomes"] == 1 and d20.loc["NCT00000001", "n_secondary_outcomes"] == 1
d26 = pd.read_parquet(os.path.join(F, "aact_202601.parquet")).set_index("nct_id")
assert d26.loc["NCT00000001", "primary_outcomes_hash"] != d20.loc["NCT00000001", "primary_outcomes_hash"]
assert d26.loc["NCT00000002", "lead_sponsor"] == "Acme Bio, Inc."
assert "Months: 8 (2017-01 to 2026-01)" in out, out
# only real changes count: NCT01 (NYR -> ANR -> COMPLETED) and NCT02 (-> TERMINATED); case changes don't
panel = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(F, "aact_*.parquet")))])
assert set(panel["overall_status"].dropna()) == {"NOT_YET_RECRUITING", "RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED", "TERMINATED"}
assert "Trials whose status changed between snapshots: 33%" in out, out      # 2 of 6
assert "Primary completion date moved between first and last snapshot for 1 trials" in out
assert "no stale-looking months" in out
fr2 = FakeRequests()
ns, out = run(P, {"DRY_RUN": False}, fr2, quiet=True)
assert fr2.downloads == [] and "8 months done | 0 months to process" in out, out
print("TEST B OK\n" + "=" * 70)

# === C: one file link fails -> that month FAILED, the rest continue ===
P = "/tmp/nb2m_C"
T.make_project(P)
fr = FakeRequests(broken={"20200101_export_ctgov.zip"})
ns, out = run(P, {"DRY_RUN": False}, fr, quiet=True)
assert "202001: FAILED" in out and "201701:" in out and "202601:" in out, out
assert fr.downloads.count("20200101_export_ctgov.zip") == 5     # retried, with a fresh link each time
assert sum(u.endswith("/200/download") for u in fr.api_calls) == 5
print("TEST C OK\n" + "=" * 70)

# === D: wrong key -> clear message, nothing downloaded ===
P = "/tmp/nb2m_D"
T.make_project(P)
fr = FakeRequests(key="the-real-key")
try:
    run(P, {"DRY_RUN": True}, fr, quiet=True)
    raise AssertionError("should have stopped")
except Exception as e:
    assert type(e).__name__ == "AACTError" and "401 invalid_api_key" in str(e) and "Colab Secrets" in str(e), e
    assert KEY not in str(e)
assert fr.downloads == []
print("TEST D OK\n" + "=" * 70)

# === E: key revoked mid-run -> Step 5 stops instead of failing every month ===
P = "/tmp/nb2m_E"
T.make_project(P)
fr = FakeRequests()
orig_get = fr.get
def revoke_after_list(url, **kw):
    if url.endswith("/download"):
        fr.key = "revoked"
    return orig_get(url, **kw)
fr.get = revoke_after_list
ns, out = run(P, {"DRY_RUN": False}, fr, quiet=True)
assert "201701: STOPPED" in out and "201801:" not in out and fr.downloads == [], out
print("TEST E OK\n" + "=" * 70)

# === F: nothing in the archive for the window -> message, no crash ===
P = "/tmp/nb2m_F"
T.make_project(P)
ns, out = run(P, {"DRY_RUN": True, "AACT_FIRST_MONTH": "2011-01", "AACT_LAST_MONTH": "2012-12"}, FakeRequests(), quiet=True)
assert "no flat-file snapshots between 2011-01-01 and 2012-12-31" in out and "No monthly snapshots to process yet" in out and "No monthly AACT files yet" in out, out
print("TEST F OK")

# === G: harmonizers on their own ===
iso, code = ns["iso_dates"], ns["code"]
assert list(iso(pd.Series(["2013-12-15", "June 2012", "January 5, 2010", None, "", "2017-01-05 00:00:00", "2014-03"])).fillna("-")) == \
    ["2013-12-15", "2012-06-01", "2010-01-05", "-", "-", "2017-01-05", "2014-03-01"]
assert list(code(pd.Series(["Active, not recruiting", "Unknown status", "Phase 1/Phase 2", "Early Phase 1", "N/A", "Anticipated",
                            "PHASE2", "", None])).fillna("-")) == \
    ["ACTIVE_NOT_RECRUITING", "UNKNOWN", "PHASE1/PHASE2", "EARLY_PHASE1", "NA", "ESTIMATED", "PHASE2", "-", "-"]
print("TEST G OK")
print("\nALL TESTS PASSED")

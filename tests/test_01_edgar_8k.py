"""Offline test: run the notebook's code cells against fixtures with a mocked SEC."""
import gzip, io, json, os, shutil, sys, zipfile, contextlib
import pandas as pd

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "tools"))
from cells import load_stage
C = load_stage("01_edgar_8k")

LOCAL_TMP = "/tmp/tiv_tmp"

# ---------------- fixtures ----------------
def sub_json(cik, sic, name, recent, files=(), pad_before_sic=0):
    d = {"cik": str(cik), "entityType": "operating"}
    if pad_before_sic:
        d["padding"] = "x" * pad_before_sic          # pushes "sic" beyond the first 4 KB
    d.update({"sic": sic, "sicDescription": "desc", "name": name, "tickers": ["T%d" % cik],
              "exchanges": ["Nasdaq"], "stateOfIncorporation": "DE", "formerNames": [{"name": "Old %d" % cik}],
              "filings": {"recent": recent, "files": [{"name": f} for f in files]}})
    return d

def arrays(rows):
    keys = ["accessionNumber", "filingDate", "acceptanceDateTime", "form", "items", "primaryDocument",
            "primaryDocDescription", "size", "reportDate"]
    return {k: [r.get(k, "") for r in rows] for k in keys}

def fr(acc, date, form="8-K", items="8.01,9.01"):
    return {"accessionNumber": acc, "filingDate": date, "acceptanceDateTime": date + "T16:30:05.000Z",
            "form": form, "items": items, "primaryDocument": "d.htm", "primaryDocDescription": "8-K", "size": 1000}

J1 = sub_json(1, "2834", "Acme Bio Inc", arrays([
    fr("0000000001-15-000010", "2015-04-03"),
    fr("0000000001-15-000020", "2015-05-01", form="10-Q"),
    fr("0000000001-09-000001", "2009-06-01"),
]), files=["CIK0000000001-submissions-001.json"])
J1_OLD = arrays([fr("0000000001-12-000002", "2012-01-05", items="2.02,9.01"),
                 fr("0000000001-12-000003", "2012-02-01", form="8-K/A")])
J2 = sub_json(2, "7372", "Softco", arrays([fr("0000000002-15-000001", "2015-03-01")]))
J3 = sub_json(3, "2836", "Bigbio Corp", arrays([
    fr("0000000003-20-000005", "2020-07-01"),
    fr("0000000003-21-000007", "2021-03-01"),
    fr("0000000003-24-000009", "2024-11-11"),
]), pad_before_sic=6000)
J4 = {"cik": "4", "entityType": "individual", "sic": "", "name": "A Person", "filings": {"recent": arrays([]), "files": []}}

def make_zip(path):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("CIK0000000001.json", json.dumps(J1))
        z.writestr("CIK0000000001-submissions-001.json", json.dumps(J1_OLD))
        z.writestr("CIK0000000002.json", json.dumps(J2))
        z.writestr("CIK0000000003.json", json.dumps(J3))
        z.writestr("CIK0000000004.json", json.dumps(J4))

def submission_txt(acc, accept, sic_line, docs, items=("Other Events", "Financial Statements and Exhibits")):
    hdr = (f"<SEC-DOCUMENT>{acc}.txt : {accept[:8]}\n<SEC-HEADER>{acc}.hdr.sgml : {accept[:8]}\n"
           f"<ACCEPTANCE-DATETIME>{accept}\nACCESSION NUMBER:\t\t{acc}\nCONFORMED SUBMISSION TYPE:\t8-K\n"
           f"PUBLIC DOCUMENT COUNT:\t\t{len(docs)}\nCONFORMED PERIOD OF REPORT:\t{accept[:8]}\n"
           + "".join(f"ITEM INFORMATION:\t\t{i}\n" for i in items) +
           f"FILED AS OF DATE:\t\t{accept[:8]}\nFILER:\n\n\tCOMPANY DATA:\t\n\t\tCOMPANY CONFORMED NAME:\t\t\tACME BIO INC\n"
           f"\t\tCENTRAL INDEX KEY:\t\t\t0000000001\n\t\tSTANDARD INDUSTRIAL CLASSIFICATION:\t{sic_line}\n</SEC-HEADER>\n")
    body = ""
    for i, (typ, fname, desc, text) in enumerate(docs, 1):
        body += f"<DOCUMENT>\n<TYPE>{typ}\n<SEQUENCE>{i}\n<FILENAME>{fname}\n" + (f"<DESCRIPTION>{desc}\n" if desc else "") + f"<TEXT>\n{text}\n</TEXT>\n</DOCUMENT>\n"
    return (hdr + body + "</SEC-DOCUMENT>\n").encode("utf-8")

IXBRL_MAIN = """<XBRL>
<?xml version='1.0' encoding='ASCII'?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><head><title>8-K</title><style>p{margin:0}</style></head>
<body><div style="display: none"><ix:header><ix:hidden><ix:nonNumeric name="dei:EntityCentralIndexKey">0000000001</ix:nonNumeric></ix:hidden></ix:header></div>
<p style="font-family:Times">UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p>
<p><font>Item&#160;8.01</font><font> Other Events.</font></p>
<p>On April&nbsp;3, 2015, the Company <span>announced</span> topline results.</p>
</body></html>
</XBRL>"""
EX991 = """<html><body>
<p align="center"><b>Acme Bio Announces Positive Phase 3 Results for ACM-101</b></p>
<p>CAMBRIDGE, Mass., April
3, 2015 &#8211; Acme Bio,
Inc. (Nasdaq: ACME) today announced that its pivotal trial met its primary endpoint.</p>
<table><tr><td>Revenue</td><td>$</td><td>1,234</td></tr><tr><td>Net loss</td><td>$</td><td>(5,678)</td></tr></table>
<p>Forward-looking statements<br>This press release contains...</p>
</body></html>"""
EX992_PLAIN = "<PAGE>\n   Investor presentation slide text\n\n\n   Page 2   text   \n</PAGE>"
GRAPHIC = "begin 644 logo.jpg\nM_]C_X``02D9)1@`!`0$`8`!@``#_VP!#``,\"`@,\"`@,#`P,$`P,$!0@%!00$\nend"
EX101 = "<XBRL><?xml version='1.0'?><xs:schema/></XBRL>"
PLAIN_MAIN = "                      UNITED STATES\n  FORM 8-K\n\nItem 2.02  Results of Operations\n\n  The Company reported revenue.\n"

TXT = {
    "0000000001-15-000010": submission_txt("0000000001-15-000010", "20150403163005", "PHARMACEUTICAL PREPARATIONS [2834]",
        [("8-K", "d8k.htm", "8-K", IXBRL_MAIN), ("EX-99.1", "ex991.htm", "PRESS RELEASE", EX991),
         ("EX-99.2", "ex992.txt", "SLIDES", EX992_PLAIN), ("GRAPHIC", "logo.jpg", None, GRAPHIC), ("EX-101.SCH", "x.xsd", None, EX101)]),
    "0000000001-12-000002": submission_txt("0000000001-12-000002", "20120105070101", "PHARMACEUTICAL PREPARATIONS [2834]",
        [("8-K", "d.txt", None, PLAIN_MAIN), ("EX-99.1", "ex.htm", "EARNINGS RELEASE", EX991)],
        items=("Results of Operations and Financial Condition", "Financial Statements and Exhibits")),
    "0000000003-21-000007": submission_txt("0000000003-21-000007", "20210301200000", "BIOLOGICAL PRODUCTS (NO DIAGNOSTIC SUBSTANCES) [2836]",
        [("8-K", "d.htm", "8-K", "<html><body><p>Item 5.02 Departure of Directors.</p></body></html>")],
        items=("Departure of Directors or Certain Officers",)),
    "0000000003-24-000009": submission_txt("0000000003-24-000009", "20241111081500", "BIOLOGICAL PRODUCTS (NO DIAGNOSTIC SUBSTANCES) [2836]",
        [("8-K", "d.htm", "8-K", "<html><body><p>Item 7.01 Regulation FD.</p></body></html>"),
         ("EX-99.1", "e.htm", "PRESS RELEASE", "<html><body><p>Bigbio receives FDA approval.</p></body></html>")]),
}

MASTER_2015Q1 = ("Description: Master Index of EDGAR Dissemination Feed\nLast Data Received: March 31, 2015\n\n"
                 "CIK|Company Name|Form Type|Date Filed|Filename\n"
                 "--------------------------------------------------------------------------------\n"
                 "1|Acme Bio Inc|8-K|2015-03-02|edgar/data/1/0000000001-15-000005.txt\n"
                 "2|Softco|8-K|2015-03-01|edgar/data/2/0000000002-15-000001.txt\n"
                 "4|A Person|4|2015-03-01|edgar/data/4/0000000004-15-000001.txt\n"
                 "5|Weird | Name LLC|8-K|2015-02-01|edgar/data/5/0000000005-15-000001.txt\n")

# ---------------- mock HTTP ----------------
class Resp:
    def __init__(self, status, content=b"", headers=None):
        self.status_code, self.content, self.headers = status, content, headers or {}
    @property
    def text(self):
        return self.content.decode("utf-8", "replace")
    def json(self):
        return json.loads(self.content)
    def iter_content(self, n):
        for i in range(0, len(self.content), n):
            yield self.content[i:i + n]

class MockSEC:
    def __init__(self, mode="normal"):
        self.mode, self.calls, self.flaky_seen = mode, [], set()
    def __call__(self, url, stream, timeout):
        self.calls.append(url)
        if self.mode == "blocked":
            return Resp(403, b"<html>Your Request Originates from an Undeclared Automated Tool</html>")
        if url.endswith("submissions.zip"):
            return Resp(404)
        if "full-index/2015/QTR1/master.gz" in url:
            return Resp(200, gzip.compress(MASTER_2015Q1.encode()))
        if "full-index" in url:
            return Resp(404)
        if "data.sec.gov/submissions/" in url:
            name = url.rsplit("/", 1)[1]
            table = {"CIK0000000001.json": J1, "CIK0000000002.json": J2, "CIK0000000005.json": J2,
                     "CIK0000000001-submissions-001.json": J1_OLD}
            return Resp(200, json.dumps(table[name]).encode()) if name in table else Resp(404)
        acc = url.rsplit("/", 1)[1].replace(".txt", "")
        if acc == "0000000003-21-000007" and acc not in self.flaky_seen:
            self.flaky_seen.add(acc)
            return Resp(429)
        return Resp(200, TXT[acc]) if acc in TXT else Resp(404)

# ---------------- runner ----------------
def run(project_dir, overrides, mock, cells, ns=None, quiet=False):
    ns = ns if ns is not None else {}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            for src in cells:
                exec(compile(src, "<cell>", "exec"), ns)
                if src is C.CONFIG:
                    ns.update(USER_AGENT="Test User test@example.com", PROJECT_DIR=project_dir, MAX_RPS=1000, CHUNK=2)
                    ns.update(overrides)
                if src is C.HTTP:
                    ns["SEC"]._fetch = mock
    except Exception:
        print(buf.getvalue())
        raise
    out = buf.getvalue()
    if not quiet:
        print(out)
    return ns, out

ALL = [C.CONFIG, C.SETUP, C.HTTP, C.UNIVERSE, C.FILINGS, C.PARSE, C.DOWNLOAD, C.SUMMARY]

def fresh(dirname):
    shutil.rmtree(dirname, ignore_errors=True)
    shutil.rmtree(LOCAL_TMP, ignore_errors=True)
    os.makedirs(LOCAL_TMP, exist_ok=True)

# === Test A: zip route, full run, resume, failure cap ===
P = "/tmp/tiv_testA"
fresh(P)
make_zip(os.path.join(LOCAL_TMP, "submissions.zip"))
mock = MockSEC()
ns, out = run(P, {"DRY_RUN": False}, mock, ALL)
u = ns["universe"]
assert sorted(u["cik"]) == [1, 3], u
fi = ns["filing_index"]
assert sorted(fi["accession"]) == sorted(["0000000001-15-000010", "0000000001-12-000002", "0000000003-20-000005",
                                          "0000000003-21-000007", "0000000003-24-000009"]), fi
shards = os.path.join(P, "edgar", "8k_text", "shards")
meta = pd.concat([pd.read_parquet(p) for p in sorted(__import__("glob").glob(shards + "/meta_*.parquet"))])
docs = pd.concat([pd.read_parquet(p) for p in sorted(__import__("glob").glob(shards + "/docs_*.parquet"))])
assert len(meta) == 4 and "0000000003-20-000005" not in set(meta["accession"]), meta
m10 = meta.set_index("accession").loc["0000000001-15-000010"]
assert m10["acceptance_datetime"] == "20150403163005" and m10["sic_header"] == "2834" and m10["n_ex99"] == 2, m10
assert m10["items_header"] == "Other Events; Financial Statements and Exhibits", m10["items_header"]
d10 = docs[docs["accession"] == "0000000001-15-000010"].set_index("doc_type")
assert list(d10.index) == ["8-K", "EX-99.1", "EX-99.2"], d10.index
main = d10.loc["8-K", "text"]
assert "0000000001" not in main and "p{margin" not in main and "Item 8.01 Other Events." in main, main
assert "On April 3, 2015, the Company announced topline results." in main, main
pr = d10.loc["EX-99.1", "text"]
assert "CAMBRIDGE, Mass., April 3, 2015 \u2013 Acme Bio, Inc. (Nasdaq: ACME) today announced" in pr and "Revenue $ 1,234" in pr, pr
assert "Forward-looking statements\nThis press release" in pr, pr
assert d10.loc["EX-99.2", "text"] == "Investor presentation slide text\n\nPage 2 text", repr(d10.loc["EX-99.2", "text"])
plain = docs[(docs["accession"] == "0000000001-12-000002") & (docs["doc_type"] == "8-K")]["text"].iloc[0]
assert plain.startswith("UNITED STATES\nFORM 8-K\n\nItem 2.02 Results"), repr(plain)
assert "Failures logged: 1" in out and "HTTP 404" in out, out
assert "Header date matches" in out
print("TEST A first run OK\n" + "=" * 70)

# resume twice: 404 filing retried until 3 failures, then skipped
for k in range(3):
    ns, out = run(P, {"DRY_RUN": False}, mock, [C.CONFIG, C.SETUP, C.HTTP, C.UNIVERSE, C.FILINGS, C.PARSE, C.DOWNLOAD], quiet=True)
    line = [l for l in out.splitlines() if "to download now" in l][0]
    print("resume", k + 1, "->", line)
assert "4 filings already done; 0 to download now" in line, line
fails = pd.read_csv(os.path.join(P, "edgar", "8k_text", "failures.csv"))
assert (fails["accession"] == "0000000003-20-000005").sum() == 3, fails
assert "Loaded universe" in out and "Loaded filing list" in out
print("TEST A resume OK\n" + "=" * 70)

# === Test B: dry run in a fresh folder ===
P = "/tmp/tiv_testB"
fresh(P)
make_zip(os.path.join(LOCAL_TMP, "submissions.zip"))
ns, out = run(P, {"DRY_RUN": True, "DRY_RUN_N": 3}, MockSEC(), ALL, quiet=True)
assert "3 to download now (dry run sample)" in out, out
assert os.path.isdir(os.path.join(P, "edgar", "8k_text_dryrun", "shards"))
assert not os.path.exists(os.path.join(P, "edgar", "8k_text"))
print("TEST B dry run OK\n" + "=" * 70)

# === Test C: fallback route (no bulk zip), one quarter of index ===
P = "/tmp/tiv_testC"
fresh(P)
mock = MockSEC()
ns, out = run(P, {"DRY_RUN": False, "START_DATE": "2015-01-01", "END_DATE": "2015-12-31"}, mock,
              [C.CONFIG, C.SETUP, C.HTTP, C.UNIVERSE, C.FILINGS], quiet=False)
assert sorted(ns["universe"]["cik"]) == [1], ns["universe"]
assert list(ns["filing_index"]["accession"]) == ["0000000001-15-000010"], ns["filing_index"]
assert os.path.exists(os.path.join(P, "edgar", "submissions_json", "CIK0000000001.json"))
print("TEST C fallback OK\n" + "=" * 70)

# === Test D: SEC refuses the User-Agent ===
P = "/tmp/tiv_testD"
fresh(P)
make_zip(os.path.join(LOCAL_TMP, "submissions.zip"))
ns, out = run(P, {"DRY_RUN": False}, MockSEC(), [C.CONFIG, C.SETUP, C.HTTP, C.UNIVERSE, C.FILINGS, C.PARSE], quiet=True)
ns["SEC"]._fetch = MockSEC("blocked")
ns, out = run(P, {}, None, [C.DOWNLOAD], ns=ns, quiet=True)
assert "STOPPED: The SEC refused" in out, out
print("TEST D block handling OK")

# === Test E: the placeholder User-Agent is rejected ===
try:
    ns = {}
    exec(C.CONFIG, ns); ns["PROJECT_DIR"] = "/tmp/tiv_testE"
    exec(C.SETUP, ns)
    raise SystemExit("placeholder UA was not rejected")
except AssertionError as e:
    print("TEST E placeholder rejected:", e)
print("\nALL TESTS PASSED")

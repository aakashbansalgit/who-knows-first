"""Offline test of Notebook 3b (v3) with a made-up project folder, a fake EDGAR and a fake ClinicalTrials.gov."""
import io, os, sys, json, shutil, contextlib
from urllib.parse import urlencode
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "tools"))
from cells import load_stage
C = load_stage("03b_stop_disclosure")

P = "/tmp/nb3b_test"
shutil.rmtree(P, ignore_errors=True)
E, S, G, F = f"{P}/edgar", f"{P}/edgar/8k_text/shards", f"{P}/ctgov", f"{P}/features"
for d in (E, S, G, F):
    os.makedirs(d)

pd.DataFrame({"cik": [1, 5, 7, 9], "name": ["ACME BIO INC", "EURO BIO PLC", "OTHERCO INC", "BIGPHARMA INC"],
              "tickers": ["ACME", "EURB", "OTH", "BIGP"], "former_names": ["", "", "", ""]}).to_parquet(f"{E}/universe_x.parquet")
K = [  # accession, cik, market date, text, shard
    ("K1", 1, "2023-03-20", "ACME Bio today announced it has discontinued the Phase 2 STRIDE trial of its pain drug.", 1),
    ("K2", 1, "2023-05-20", "ACME Bio has suspended the BRAVO study after a safety review.", 1),
    ("K3", 1, "2023-09-10", "ACME presents data on acmeprofen at a conference.", 2),
    ("K4", 9, "2023-03-20", "BigPharma terminated NCT00000004.", 2),
    ("K5", 7, "2023-03-25", "Our partner's trial NCT00000001 was terminated.", 2),
    ("K6", 1, "2021-01-01", "ACME discontinued ACM-111 years ago.", 2),
]
for s in (1, 2):
    pd.DataFrame([{"accession": a, "doc_type": "EX-99.1", "text": t} for a, c, d, t, sh in K if sh == s]).to_parquet(
        f"{S}/docs_{s:05d}.parquet")
pd.DataFrame({"accession": [k[0] for k in K], "cik": [k[1] for k in K],
              "event_date": pd.to_datetime([k[2] for k in K])}).to_parquet(f"{F}/filings.parquet")
pd.DataFrame({"accession": ["K5", "K4"], "nct_id": ["NCT00000001", "NCT00000004"], "link_type": "nct",
              "matched": ["NCT00000001", "NCT00000004"], "n_name_trials": np.nan}).to_parquet(f"{F}/filing_trial_links.parquet")
ev = [  # nct, cik, registry date, lead trials, 8-Ks in a year, detail, is_stop, lead sponsor then, owner source
    ("NCT00000001", 1, "2023-03-15", 5, 10, "RECRUITING -> TERMINATED", True, "Acme Bio, Inc.", "sponsor_at_time"),
    ("NCT00000002", 1, "2023-06-01", 5, 10, "ACTIVE_NOT_RECRUITING -> SUSPENDED", True, "Acme Bio, Inc.", "sponsor_at_time"),
    ("NCT00000003", 5, "2023-04-01", 3, 0, "RECRUITING -> TERMINATED", True, "Euro Bio plc", "sponsor_at_time"),
    ("NCT00000004", 9, "2023-03-15", 500, 40, "RECRUITING -> TERMINATED", True, "BigPharma", "sponsor_at_time"),
    ("NCT00000005", 1, "2023-09-01", 5, 10, "RECRUITING -> WITHDRAWN", True, "Acme Bio, Inc.", "sponsor_at_time"),
    ("NCT00000006", 7, "2023-03-01", 2, 1, "RECRUITING -> TERMINATED", True, "OtherCo Inc", "sponsor_at_time"),     # not filing then
    ("NCT00000007", 1, "2023-08-01", 5, 10, "RECRUITING -> TERMINATED", True, "Acme Bio, Inc.", "sponsor_at_time"),  # alias + acronym
    ("NCT00000008", 1, "2023-07-01", 5, 10, "ACTIVE_NOT_RECRUITING -> TERMINATED", True, "Acme Bio, Inc.", "sponsor_at_time"),  # admin
    ("NCT00000009", 1, "2023-07-01", 5, 10, "RECRUITING -> TERMINATED", True, "Univ Hospital", "collaborator"),
    ("NCT00000010", 1, "2023-07-01", 5, 10, "RECRUITING -> TERMINATED", True, "Acme Biologix Inc", "sponsor_at_time"),  # fuzzy
    ("NCT00000011", 5, "2023-05-01", 3, 0, "RECRUITING -> TERMINATED", True, "Old Euro Pharma", "sponsor_at_time"),  # former, out of date
    ("NCT00000001", 1, "2022-01-01", 5, 10, "9 months", False, "Acme Bio, Inc.", "sponsor_at_time"),
]
n_ev = len(ev)
pd.DataFrame({"nct_id": [e[0] for e in ev], "cik": pd.array([e[1] for e in ev], dtype="Int64"),
              "registry_date": pd.to_datetime([e[2] for e in ev]), "firm_lead_trials": [e[3] for e in ev],
              "firm_8ks_year": [e[4] for e in ev], "detail": [e[5] for e in ev], "is_stop": [e[6] for e in ev],
              "lead_sponsor": [e[7] for e in ev], "owner_source": [e[8] for e in ev],
              "event_type": ["status"] * (n_ev - 1) + ["pcd_later"], "phase": "PHASE2",
              "days_to_8k": [5.0] + [np.nan] * (n_ev - 1),
              "disclosed_before": [False, True] + [False] * (n_ev - 2)}).to_parquet(f"{F}/registry_events.parquet")
TR = {  # nct: (title, why stopped, interventions, lead sponsor today)
    "NCT00000001": ("A Study of ACM-111 in Pain (STRIDE)", "Business decision", "ACM-111 dose A; Placebo", "Acme Bio, Inc."),
    "NCT00000002": ("Phase 3 Study of Acmezil (BRAVO)", "Safety review", "acmezil 10 mg tablets", "Acme Bio, Inc."),
    "NCT00000003": ("EUR-55 in Psoriasis", None, "EUR-55", "Euro Bio plc"),
    "NCT00000004": ("BP-9 Study (BIGSTUDY)", None, "BP-9", "BigPharma"),
    "NCT00000005": ("Acmeprofen in Migraine (NASH)", "Sponsor decision", "acmeprofen in combination with pembrolizumab",
                    "Acme Bio, Inc."),
    "NCT00000006": ("OTH-66 in Asthma", "Funding", "OTH-66", "OtherCo Inc"),
    "NCT00000007": ("ACM-777 in Gout", "Strategic decision", "ACM-777", "Acme Bio, Inc."),
    "NCT00000008": ("ACM-888 Rollover Study", "Patients to be followed up in the long-term follow-up study", "ACM-888",
                    "Acme Bio, Inc."),
    "NCT00000009": ("Investigator Study of ACM-999", "Funding", "ACM-999", "Univ Hospital"),
    "NCT00000010": ("ACM-101 in Pain", "Business decision", "ACM-101", "Acme Biologix Inc"),
    "NCT00000011": ("EUR-11 in Acne", "Business decision", "EUR-11", "Old Euro Pharma"),
    "NCT00000012": ("Generic Study", None, "nucleoside analogue inhibitors; pembrolizumab", "Acme Bio, Inc."),
    "NCT00000013": ("Big study", None, "pembrolizumab", "BigPharma"),
    "NCT00000014": ("Other big study", None, "pembrolizumab", "OtherCo Inc"),
    "NCT00000015": ("Euro combo", None, "pembrolizumab", "Euro Bio plc"),
}
pd.DataFrame({"nct_id": list(TR), "brief_title": [v[0] for v in TR.values()], "why_stopped": [v[1] for v in TR.values()],
              "interventions": [v[2] for v in TR.values()], "lead_sponsor": [v[3] for v in TR.values()]}).to_parquet(
    f"{G}/industry_trials_current_withcollab.parquet")
tfirm = [("NCT00000001", 1), ("NCT00000002", 1), ("NCT00000005", 1), ("NCT00000003", 5), ("NCT00000004", 9),
         ("NCT00000006", 7), ("NCT00000007", 1), ("NCT00000008", 1), ("NCT00000009", 1), ("NCT00000010", 1),
         ("NCT00000011", 5), ("NCT00000012", 1), ("NCT00000013", 9), ("NCT00000014", 7), ("NCT00000015", 5)]
pd.DataFrame({"nct_id": [t[0] for t in tfirm], "cik": [t[1] for t in tfirm],
              "role": ["collaborator" if t[0] == "NCT00000009" else "lead" for t in tfirm],
              "sponsor": "x", "method": "exact"}).to_parquet(f"{G}/trial_firm.parquet")
pd.DataFrame({"sponsor": ["Acme Bio, Inc.", "Euro Bio plc", "Acme Biologix Inc", "Old Euro Pharma", "OtherCo Inc"],
              "cik": [1, 5, 1, 5, 7], "method": ["exact", "exact", "fuzzy", "exact_former", "exact"],
              "score": [100, 100, 93, 100, 100]}).to_csv(f"{G}/sponsor_matches.csv", index=False)

def subs(rows, files=(), former=()):
    return {"formerNames": list(former),
            "filings": {"recent": {"form": [r[0] for r in rows], "filingDate": [r[1] for r in rows],
                                   "accessionNumber": [r[2] for r in rows], "primaryDocument": [r[3] for r in rows]},
                        "files": list(files)}}
filler = "Other programs advanced as planned. " * 12
A = "https://www.sec.gov/Archives/edgar/data"
FAKE = {
    "https://data.sec.gov/submissions/CIK0000000001.json": json.dumps(subs(
        [("10-Q", "2023-05-10", "0000000001-23-000010", "q1.htm"), ("424B5", "2023-07-01", "0000000001-23-000020", "p.htm"),
         ("10-Q", "2023-11-10", "0000000001-23-000030", "q3.htm"), ("8-K", "2023-03-20", "0000000001-23-000002", "k.htm"),
         ("10-Q", "2023-08-10", "0000000001-23-000025", "q2.htm"), ("10-K", "2020-03-01", "0000000001-20-000005", "k20.htm")],
        [{"name": "CIK0000000001-submissions-001.json", "filingFrom": "2015-01-01", "filingTo": "2019-12-31"}])),
    "https://data.sec.gov/submissions/CIK0000000005.json": json.dumps(subs(
        [("6-K", "2023-04-20", "0000000005-23-000003", "cover.htm"), ("20-F", "2024-03-30", "0000000005-24-000001", "20f.htm")],
        [{"name": "CIK0000000005-submissions-001.json", "filingFrom": "2022-01-01", "filingTo": "2023-12-31"}],
        [{"name": "Old Euro Pharma plc", "from": "2008-01-01T00:00:00.000Z", "to": "2015-06-01T00:00:00.000Z"}])),
    "https://data.sec.gov/submissions/CIK0000000005-submissions-001.json": json.dumps(
        {"form": ["6-K"], "filingDate": ["2023-12-01"], "accessionNumber": ["0000000005-23-000009"], "primaryDocument": ["c2.htm"]}),
    "https://data.sec.gov/submissions/CIK0000000007.json": json.dumps(subs(
        [("10-K", "2015-03-01", "0000000007-15-000001", "k15.htm"), ("1-U", "2023-04-15", "0000000007-23-000001", "u.htm")])),
    f"{A}/1/000000000123000010/q1.htm": "<html><body><p>In March 2023 we terminated our Phase 2 trial of ACM-111 "
        f"(NCT00000001).</p><p>{filler}</p><p>Our acmeprofen program continues.</p></body></html>",
    f"{A}/1/000000000123000020/p.htm": "<p>Recent developments: we suspended the BRAVO study in May.</p>",
    f"{A}/1/000000000123000025/q2.htm": "<p>In July we decided to stop development of zorbimab.</p><p>We closed enrollment in "
        "ZEPHYR-2 in July.</p><p>In February the FDA lifted the clinical hold on ACM-111.</p><p>The FDA placed a partial "
        "clinical hold on ACM111 in January.</p><p>After ACM-111 results we discontinued the BRAVO study.</p>",
    f"{A}/1/000000000123000030/q3.htm": "<p>The acmeprofen program is ongoing.</p><p>Trials of acmeprofen may be delayed or "
        "halted by regulators.</p><p>The license agreement covering acmeprofen was terminated by our partner.</p><p>Acmeprofen "
        "met its endpoint. Development of other programs was halted.</p><p>Acmeprofen oral suspension is being studied.</p>",
    f"{A}/5/000000000523000003/0000000005-23-000003.txt": "<SEC-DOCUMENT><DOCUMENT>\n<TYPE>6-K\n<TEXT><html>Form 6-K cover"
        "</html></TEXT>\n</DOCUMENT>\n<DOCUMENT>\n<TYPE>EX-99.1\n<TEXT><html><p>Euro Bio today announced it will discontinue "
        "development of EUR-55 and has terminated the trial.</p></html></TEXT>\n</DOCUMENT>\n<DOCUMENT>\n<TYPE>GRAPHIC\n<TEXT>"
        "begin 644 logo.jpg\nM8V%T EUR-55 terminated junk\nend</TEXT>\n</DOCUMENT></SEC-DOCUMENT>",
    f"{A}/5/000000000524000001/20f.htm": "<p>In April 2023 we terminated the EUR-55 trial.</p>",
    f"{A}/5/000000000523000009/0000000005-23-000009.txt": "<DOCUMENT>\n<TYPE>EX-99.1\n<TEXT><p>EUR-55 data presented.</p></TEXT>"
        "\n</DOCUMENT>",
    f"{A}/7/000000000723000001/0000000007-23-000001.txt": "<DOCUMENT>\n<TYPE>EX-99.1\n<TEXT><p>We discontinued OTH-66.</p></TEXT>"
        "\n</DOCUMENT>",
}
CT_STUDIES = {  # registry acronym and other names, as ClinicalTrials.gov's API returns them
    "NCT00000007": {"acronym": "ZEPHYR-2", "interventions": [{"name": "ACM-777", "otherNames": ["zorbimab", "Placebo"]}]},
    "NCT00000005": {"interventions": [{"name": "pembrolizumab", "otherNames": ["Keytruda"]}]},
    "NCT00000001": {"acronym": "TERM", "interventions": [{"name": "ACM-111 dose A", "otherNames": []}]},
}

class R:
    def __init__(self, body):
        self.status_code = 200 if body is not None else 404
        self.text = body or ""
        self.content = self.text.encode()
    def json(self):
        return json.loads(self.text)

class FakeSEC:
    def __init__(self):
        self.urls, self.n_requests = [], 0
    def get(self, url, **kw):
        self.urls.append(url); self.n_requests += 1
        return R(FAKE.get(url))

class FakeCTG:
    def __init__(self):
        self.calls = []
    def get(self, url, params=None, **kw):
        assert url == "https://clinicaltrials.gov/api/v2/studies"
        self.calls.append(params)
        ids = params["filter.ids"].split(",")
        studies = []
        for n in ids:
            d = CT_STUDIES.get(n, {})
            idm = {"nctId": n, **({"acronym": d["acronym"]} if "acronym" in d else {})}
            studies.append({"protocolSection": {"identificationModule": idm,
                                                "armsInterventionsModule": {"interventions": d.get("interventions", [])}}})
        return R(json.dumps({"studies": studies}))

CELLS = [C.CONFIG, C.SETUP, C.LOAD, C.FIRMS, C.CHECK_8K, C.EDGAR, C.COMBINE, C.SUMMARY]
def run(overrides):
    ns, buf, fake, ctg = {}, io.StringIO(), FakeSEC(), FakeCTG()
    try:
        with contextlib.redirect_stdout(buf):
            for src in CELLS:
                exec(compile(src, "<cell>", "exec"), ns)
                if src is C.CONFIG:
                    ns.update(PROJECT_DIR=P, USER_AGENT="Test User test@example.com"); ns.update(overrides)
                if src is C.SETUP:
                    ns["SEC"], ns["CTG"] = fake, ctg
    except Exception:
        print(buf.getvalue()); raise
    print(buf.getvalue())
    return ns, fake, ctg

# ---- unit checks of the name and wording rules ----
ns = {}
for src in (C.CONFIG,):
    exec(compile(src, "<cell>", "exec"), ns)
nv = C.LOAD[C.LOAD.index("STOP_NAMES ="):C.LOAD.index("# other drug names")]
exec("import re\n" + nv, ns)
v = ns["name_variants"]
assert v("Drug: VTX958 Dose A") == {"vtx958"}, v("Drug: VTX958 Dose A")
assert v("nanatinostat in combination with valganciclovir") == {"nanatinostat", "valganciclovir"}
assert v("PRS-343 in combination with atezolizumab") == {"prs-343", "atezolizumab"}
assert v("losmapimod oral tablet") == {"losmapimod"} and v("Arbaclofen ER Tablets") == {"arbaclofen"}
assert v("diacerein 1% ointment") == {"diacerein"}, v("diacerein 1% ointment")
assert v("analogue inhibitors, nucleoside, nucleotide") == set(), v("analogue inhibitors, nucleoside, nucleotide")
assert v("enriched hematopoietic stem cell infusion") == set() and v("superficial hyperthermia") == set()
assert v("High-dose ACM-111") == {"acm-111"}
assert ns["acronyms"]("Losmapimod Safety and Efficacy in COVID-19 (LOSVID STUDY)") == ["LOSVID"]
assert ns["acronyms"]("2-Year Extension Study (STEADFAST Extension)") == ["STEADFAST"]
assert not ns["ok_acronym"]("TERM") and ns["ok_acronym"]("NEON-1") and ns["ok_acronym"]("CleaR-MAC")
print("TEST NAME RULES OK")

# ---- dry run: first firm only (ACME) ----
ns, fake, ctg = run({"DRY_RUN": True, "DRY_RUN_FIRMS": 1})
d = ns["discloses"]
def t(sent, name):
    i = sent.index(name)
    return d(sent, i, len(name))
assert t("The FDA placed a full clinical hold on BMF-219 in June.", "BMF-219")
assert not t("In February the FDA lifted the clinical hold on PRS-343.", "PRS-343")
assert t("As of February 24, we closed enrollment of the SL-172154 trial to focus on IV dosing.", "SL-172154")
assert t("During 2021, we closed our Phase II clinical trial of INB03 for COVID-19.", "INB03")
assert t("Sunesis will not advance vecabrutinib into the Phase 2 portion.", "vecabrutinib")
assert t("Repare does not intend to continue to develop lunresertib in any other trials.", "lunresertib")
assert t("GTB-3650 will supplant the ongoing GTB-3550 program.", "GTB-3550")
assert not t("Acmeprofen oral suspension is being studied.", "Acmeprofen")
assert not t("The FDA may place a clinical hold on ACM-111.", "ACM-111")
sd = pd.read_parquet(f"{F}/stop_disclosure_dryrun.parquet").set_index("nct_id")
assert set(sd.index) == {f"NCT0000000{i}" for i in (1, 2, 5, 7, 8, 9)} | {"NCT00000010"}, sorted(sd.index)
assert not any("CIK0000000005" in u for u in fake.urls)
assert sd.loc["NCT00000008", "exclude_reason"] == "administrative close"
assert sd.loc["NCT00000009", "exclude_reason"] == "collaborator only"
assert sd.loc["NCT00000010", "exclude_reason"] == "sponsor matched by close spelling"
assert sd.loc["NCT00000001", "in_sample"] and sd.loc["NCT00000007", "in_sample"]
print("TEST DRY RUN OK\n" + "=" * 70)

# ---- full run: reuses ACME's files, downloads EURO BIO and OTHERCO; BigPharma out of scope ----
ns, fake, ctg = run({"DRY_RUN": False})
assert not any("CIK0000000001" in u for u in fake.urls), "ACME was done in the dry run: no new requests"
assert not any("CIK0000000009" in u for u in fake.urls), "BigPharma (500 trials) is out of scope"
assert any("CIK0000000005-submissions-001.json" in u for u in fake.urls)
assert sum(len(c["filter.ids"].split(",")) for c in ctg.calls) == 5, ctg.calls       # only EURO BIO's and OTHERCO's trials
sd = pd.read_parquet(f"{F}/stop_disclosure.parquet").set_index("nct_id")
assert "NCT00000004" not in sd.index
s1 = sd.loc["NCT00000001"]
assert s1["days_8_k"] == 5 and s1["days_10_q_10_k_20_f_40_f"] == 56 and s1["first_channel"] == "8-K" and s1["timing"] == "0-30 days", s1
assert "TERM" not in s1["names_searched"] and "acm-111" in s1["names_searched"], s1["names_searched"]
s2 = sd.loc["NCT00000002"]
assert s2["days_8_k"] == -12 and s2["days_prospectus"] == 30 and s2["timing"] == "before", s2
assert s2["timing_detail"] == "1-30 days before" and s2["days_first_v2window"] == -12
s3 = sd.loc["NCT00000003"]
assert s3["days_6_k"] == 19 and s3["days_10_q_10_k_20_f_40_f"] == 364 and s3["first_channel"] == "6-K", s3
assert s3["group"].startswith("files no 8-Ks") and s3["in_sample"]
s5 = sd.loc["NCT00000005"]
assert s5["timing"] == "none" and s5["named_any"] and pd.isna(s5["days_10_q_10_k_20_f_40_f"]), s5   # other sentence / risk factor / license
assert "keytruda" not in s5["names_searched"] and "pembrolizumab" not in s5["names_searched"], s5["names_searched"]
s6 = sd.loc["NCT00000006"]
assert s6["exclude_reason"].startswith("firm not filing reports") and s6["days_reg_a"] == 45, s6
s7 = sd.loc["NCT00000007"]
assert s7["days_10_q_10_k_20_f_40_f"] == 9 and s7["in_sample"], s7
assert sd.loc["NCT00000011", "exclude_reason"] == "former name not in use then"
hits = pd.read_parquet(f"{F}/stop_disclosure_hits.parquet")
q2 = hits[hits["accession"] == "0000000001-23-000025"]
z = q2[q2["matched"].str.lower() == "zorbimab"]
assert len(z) == 1 and z["stop_hit"].all() and z["match_kind"].iloc[0] == "drug_name", q2[["matched", "stop_hit"]]
zz = q2[q2["matched"] == "ZEPHYR-2"]
assert len(zz) == 1 and zz["stop_hit"].all() and zz["match_kind"].iloc[0] == "acronym"
h1 = q2[q2["matched"].str.upper().str.replace("-", "") == "ACM111"]
assert h1["other_trial"].sum() == 1 and h1["stop_hit"].sum() == 1 and "lifted" not in h1.loc[h1["stop_hit"], "snippet"].iloc[0], h1[["snippet", "stop_hit"]]
q3 = hits[hits["accession"] == "0000000001-23-000030"]
assert len(q3[q3["stop_id"] == sd.loc["NCT00000005", "stop_id"]]) >= 5 and not q3["stop_hit"].any(), q3[["snippet", "stop_hit"]]
assert not (hits["accession"] == "K6").any() and not (hits["accession"] == "K4").any()              # outside window / out of scope
k1 = hits[(hits["accession"] == "K1")].iloc[0]
assert k1["match_kind"] == "acronym" and "STRIDE" in k1["snippet"]                                   # no NCT or drug name in K1
assert hits[(hits["accession"] == "K5") & hits["stop_hit"]]["match_kind"].iloc[0] == "nct"          # partner's 8-K, NCT link
audit = pd.read_csv(f"{F}/stop_disclosure_audit_v3.csv")
assert audit["nct_id"].tolist() == ["NCT00000005"] and "announced_elsewhere (Y/N)" in audit.columns
assert audit["ctgov_url"].iloc[0].endswith("NCT00000005") and "acmeprofen" in audit["names_searched"].iloc[0]
assert os.path.exists(f"{F}/figures/fig5_stop_disclosure_all_filings.png")
assert "NASH" not in str(ns["acronyms"]("Acmeprofen in Migraine (NASH)"))
print("TEST FULL OK\n" + "=" * 70)

# ---- rerun: everything cached, no requests ----
ns, fake, ctg = run({"DRY_RUN": False})
assert fake.n_requests == 0 and not ctg.calls, (fake.urls, ctg.calls)
print("TEST RESUME OK\nALL TESTS PASSED")

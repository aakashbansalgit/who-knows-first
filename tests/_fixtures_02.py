"""Shared fixtures for test_02_registry_snapshots.py: a firm list and a mocked ClinicalTrials.gov API."""
import io, json, os, shutil, sys, contextlib, time as _time, types
from urllib.parse import urlparse, parse_qs
import pandas as pd


# ---------------- fixtures ----------------
def make_project(P):
    shutil.rmtree(P, ignore_errors=True)
    os.makedirs(os.path.join(P, "edgar"))
    uni = pd.DataFrame([
        dict(cik=1, name="ACME BIO INC /DE/", former_names="ACME PHARMACEUTICALS CORP"),
        dict(cik=3, name="BIGBIO CORP", former_names=""),
        dict(cik=7, name="PFIZER INC", former_names=""),
        dict(cik=8, name="NOVARTIS AG", former_names=""),
        dict(cik=9, name="TWIN PHARMA INC", former_names=""),
        dict(cik=10, name="TWIN PHARMACEUTICALS CORP", former_names=""),
    ])
    for c in ["sic", "sic_description", "entity_type", "tickers", "exchanges", "state_of_incorporation"]:
        uni[c] = ""
    uni.to_parquet(os.path.join(P, "edgar", "universe_sic2834-2835-2836.parquet"), index=False)
    pd.DataFrame({"cik": [1, 3, 7], "accession": ["a", "b", "c"]}).to_parquet(
        os.path.join(P, "edgar", "filing_index_sic2834-2835-2836_8-K_2005-01-01_2025-12-31.parquet"), index=False)

def study(nct, lead, lead_cls="INDUSTRY", phases=("PHASE3",), collabs=(), stype="INTERVENTIONAL", last="2020-05-01"):
    return {"protocolSection": {
        "identificationModule": {"nctId": nct, "briefTitle": f"Trial {nct}", "orgStudyIdInfo": {"id": "X-" + nct[-2:]},
                                 "secondaryIdInfos": [{"id": "EUDRACT-1"}]},
        "statusModule": {"overallStatus": "COMPLETED", "startDateStruct": {"date": "2010-01", "type": "ACTUAL"},
                         "primaryCompletionDateStruct": {"date": "2013-12-15", "type": "ACTUAL"},
                         "studyFirstSubmitDate": "2009-12-20", "studyFirstPostDateStruct": {"date": "2010-01-05"},
                         "lastUpdatePostDateStruct": {"date": last}},
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": lead, "class": lead_cls},
                                       "collaborators": [{"name": n, "class": c} for n, c in collabs]},
        "designModule": {"studyType": stype, "phases": list(phases), "enrollmentInfo": {"count": 300, "type": "ACTUAL"}},
        "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": "ACM-101"}]},
        "conditionsModule": {"conditions": ["Cancer"]},
    }}

STUDIES = [
    study("NCT00000001", "Acme Bio, Inc."),
    study("NCT00000002", "Acme Pharmaceuticals", phases=("PHASE2",)),
    study("NCT00000003", "Pfizer", phases=("PHASE1",)),
    study("NCT00000004", "University X", lead_cls="OTHER", phases=("PHASE2",),
          collabs=[("BigBio Corporation", "INDUSTRY"), ("National Cancer Institute (NCI)", "NIH")]),
    study("NCT00000005", "Novartis Pharmaceuticals"),
    study("NCT00000006", "Twin Pharma"),
    study("NCT00000007", "AcmeBio", phases=("PHASE1",)),
    study("NCT00000008", "Pfizer", last="2001-03-01"),
    study("NCT00000009", "Pfizer", stype="OBSERVATIONAL"),
]

def history_json(nct):
    ch = [{"version": 0, "date": "2010-01-05", "status": "NOT_YET_RECRUITING", "moduleLabels": ["Study Identification", "Study Status"]},
          {"version": 1, "date": "2011-02-01", "status": "RECRUITING", "moduleLabels": ["Study Status", "Contacts/Locations"]},
          {"version": 2, "date": "2012-03-01", "status": "RECRUITING", "moduleLabels": ["Eligibility"], "extraField": 5},
          {"version": 3, "date": "2014-06-01", "status": "COMPLETED", "moduleLabels": ["Study Status", "Study Design", "Results"]}]
    return {"history": {"changes": ch}, "study": STUDIES[0]}

def version_json(nct, v):
    pcd = {0: ("2012-06", "ANTICIPATED"), 1: ("2012-12", "ANTICIPATED"), 3: ("2013-12-15", "ACTUAL")}[v]
    lead = "Acme Pharmaceuticals" if (nct == "NCT00000002" and v < 3) else "Acme Bio, Inc."
    return {"study": {"protocolSection": {
        "statusModule": {"overallStatus": ["NOT_YET_RECRUITING", "RECRUITING", None, "COMPLETED"][v],
                         "primaryCompletionDateStruct": {"date": pcd[0], "type": pcd[1]}, "lastUpdateSubmitDate": "2014-05-30"},
        "designModule": {"phases": ["PHASE3"], "enrollmentInfo": {"count": 200 + 50 * v, "type": "ESTIMATED"}},
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": lead}},
        "outcomesModule": {"primaryOutcomes": [{"measure": "Overall survival"}], "secondaryOutcomes": []}}}}

class Resp:
    def __init__(self, status, obj=None, headers=None):
        self.status_code = status
        self._obj = obj
        self.headers = headers or {}
        self.text = json.dumps(obj) if obj is not None else "error"
    def json(self):
        return self._obj

class MockCT:
    def __init__(self, strict_fields=False, block=False):
        self.strict, self.block, self.calls, self.seen429 = strict_fields, block, [], set()
    def __call__(self, url, params, timeout):
        self.calls.append((url, dict(params or {})))
        if self.block and "/api/int/" in url:
            return Resp(403)
        if url.endswith("/api/v2/studies"):
            p = params or {}
            if self.strict and p.get("fields", "").startswith("NCTId"):
                return Resp(400, {"message": "bad field"})
            if p.get("pageToken") == "T2":
                return Resp(200, {"studies": STUDIES[5:]})
            return Resp(200, {"studies": STUDIES[:5], "nextPageToken": "T2", "totalCount": len(STUDIES)})
        path = urlparse(url).path
        parts = path.strip("/").split("/")          # api int studies NCT [history v]
        nct = parts[3]
        if len(parts) == 4:
            if nct == "NCT00000003":
                return Resp(404)
            if nct == "NCT00000001" and nct not in self.seen429:
                self.seen429.add(nct)
                return Resp(429, headers={"Retry-After": "0"})
            return Resp(200, history_json(nct))
        return Resp(200, version_json(nct, int(parts[5])))

fast_time = types.SimpleNamespace(sleep=lambda s: None, monotonic=_time.monotonic, time=_time.time)

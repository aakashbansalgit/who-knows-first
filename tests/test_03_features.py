"""Offline test of Notebook 3 on a small made-up project folder."""
import io, os, sys, shutil, contextlib
import pandas as pd
import numpy as np

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "tools"))
from cells import load_stage
C = load_stage("03_features")

P = "/tmp/nb3_test"
shutil.rmtree(P, ignore_errors=True)
E, S, G = f"{P}/edgar", f"{P}/edgar/8k_text/shards", f"{P}/ctgov"
for d in (E, S, f"{G}/aact_monthly"):
    os.makedirs(d)

pd.DataFrame({"cik": [1, 2, 3, 4], "name": ["ACME BIO INC /DE/", "BETA THERAPEUTICS INC", "GAMMA BIO INC", "BIGPHARMA INC"],
              "sic": ["2834"] * 4, "tickers": ["ACME", "BETA", "", "BIGP"], "exchanges": ["Nasdaq", "Nasdaq", "", "NYSE"],
              "former_names": ["", "", "", ""]}).to_parquet(f"{E}/universe_sic2834-2835-2836.parquet")

F = [  # accession, cik, acceptance (ET), items, sic, docs [(type, description, text)]
    ("A1", 1, "20250108173000", "8.01,9.01", "2834", [
        ("8-K", "", "Item 8.01 Other Events. See Exhibit 99.1."),
        ("EX-99.1", "PRESS RELEASE", "ACME Bio Announces Positive Topline Results from Phase 3 ACCEL Trial of acmeciclib\n"
         "BOSTON, Jan. 8, 2025 (GLOBE NEWSWIRE) -- ACME Bio today announced the Phase 3 trial (NCT01234567) met its primary "
         "endpoint with a statistically significant improvement in progression-free survival (p<0.001).")]),
    ("A2", 1, "20250110070000", "7.01,9.01", "2834", [
        ("8-K", "", "Item 7.01 Regulation FD Disclosure."),
        ("EX-99.1", "CORPORATE PRESENTATION", "\n".join(["ACM-101 program update"] + [f"Slide {i}" for i in range(60)]))]),
    ("A3", 2, "20250110110000", "8.01", "2834", [
        ("8-K", "", "Item 8.01 Other Events."),
        ("EX-99.1", "EX-99.1", "Beta Therapeutics Reports Topline Data from Phase 2b Trial of BTX-200\nSAN DIEGO, Jan. 10, 2025 "
         "(BUSINESS WIRE) -- The randomized trial did not meet its primary endpoint. The company will discontinue the program.")]),
    ("A4", 2, "20250111100000", "8.01", "2834", [
        ("EX-99.1", "PRESS RELEASE", "Beta Therapeutics Receives Complete Response Letter from FDA for BTX-100 (PR Newswire)")]),
    ("A5", 1, "20251128140000", "2.02,9.01", "2834", [
        ("EX-99.1", "PRESS RELEASE", "ACME Bio Reports Third Quarter 2025 Financial Results (GLOBE NEWSWIRE) revenue grew.")]),
    ("A6", 3, "20250220163000", "8.01", "2834", [
        ("EX-99.1", "PRESS RELEASE", "Gamma Bio provides update (GLOBE NEWSWIRE): the company has terminated study NCT07654321 "
         "of GMB-77 for business reasons.")]),
    ("A7", 4, "20180305090000", "8.01", "8731", [
        ("8-K", "", "Item 8.01. BigPharma announced that an interim analysis of its Phase 1 study showed a statistically "
         "significant improvement in response.")]),
    # ---- shard 3: cases the first dry run got wrong ----
    ("B1", 1, "20250303120000", "8.01,9.01", "2834", [          # forward-looking: not a readout, not an approval
        ("8-K", "", "Item 8.01 Other Events. ACME Bio issued a press release."),
        ("EX-99.1", "EX-99.1", "ACME Bio Expects Topline Results from Phase 3 ACCEL-2 Trial in the First Half of 2026\nBOSTON, "
         "Mar. 3, 2025 (GLOBE NEWSWIRE) -- ACME Bio today announced that enrollment in the Phase 3 ACCEL-2 trial is complete. "
         "Topline data from the trial are expected in the first half of 2026. The company continues to expect FDA approval of "
         "acmeciclib by the end of 2025.")]),
    ("B2", 1, "20250305160500", "2.02,9.01", "2834", [          # earnings release recapping an old readout
        ("8-K", "", "Item 2.02 Results of Operations and Financial Condition."),
        ("EX-99.1", "EX-99.1", "ACME Bio Reports Fourth Quarter and Full Year 2024 Financial Results and Provides Business "
         "Update\nBOSTON, Mar. 5, 2025 -- ACME Bio (Nasdaq: ACME) today reported financial results. Recent highlights: in "
         "January, the Phase 3 ACCEL trial met its primary endpoint with a statistically significant improvement.")]),
    ("B3", 2, "20250410083000", "7.01,9.01", "2834", [          # "will present data"
        ("EX-99.1", "PRESS RELEASE", "Beta Therapeutics to Present New Data from Phase 2 Trial of BTX-200 at ASCO Annual "
         "Meeting\nSAN DIEGO, Apr. 10, 2025 (BUSINESS WIRE) -- Beta Therapeutics today announced that it will present data "
         "from the Phase 2 trial of BTX-200 at the ASCO Annual Meeting.")]),
    ("B4", 3, "20250415163000", "8.01", "2834", [                # business update with results inside
        ("EX-99.1", "PRESS RELEASE", "Gamma Bio Provides Business Update\nGamma Bio today announced several updates. Data from "
         "the Phase 2 study of GMB-77 showed a statistically significant reduction in symptoms.")]),
    ("B5", 1, "20250402070000", "8.01,9.01", "2834", [          # cover page of short lines; press release with no wire text
        ("8-K", "", "\n".join([f"Cover line {i}" for i in range(70)]) + "\nItem 8.01 Other Events.\nOn April 2, 2025, ACME "
         "Bio issued a press release, attached as Exhibit 99.1."),
        ("EX-99.1", "EX-99.1", "Contact: Jane Doe, Investor Relations, ACME Bio\nACME Bio Announces Positive Results from "
         "Phase 2 Study of ACM-202\nCAMBRIDGE, Mass., April 2, 2025 -- ACME Bio (Nasdaq: ACME) today announced that the "
         "Phase 2 study met its primary endpoint.")]),
    ("B6", 1, "20250505070000", "8.01,9.01", "2834", [          # regulatory news citing an old readout
        ("EX-99.1", "PRESS RELEASE", "FDA Grants Priority Review for ACME Bio's New Drug Application for acmeciclib\nBOSTON, "
         "May 5, 2025 (GLOBE NEWSWIRE) -- ACME Bio today announced the FDA accepted its NDA. The application is based on the "
         "Phase 3 ACCEL trial, which met its primary endpoint.")]),
    ("B7", 1, "20251001070000", "8.01,9.01", "2834", [          # an approval
        ("EX-99.1", "PRESS RELEASE", "ACME Bio Announces FDA Approval of acmeciclib\nBOSTON, Oct. 1, 2025 (GLOBE NEWSWIRE) -- "
         "ACME Bio today announced that the U.S. Food and Drug Administration has approved acmeciclib based on the Phase 3 "
         "ACCEL trial, which met its primary endpoint.")]),
    ("B8", 1, "20250220110000", "8.01", "2834", [                # a drug name shared by six trials: too vague for Step 5
        ("EX-99.1", "PRESS RELEASE", "ACME Bio Provides Update on acmezumab Program (GLOBE NEWSWIRE)")]),
    # ---- second dry run ----
    ("B9", 1, "20251103120000", "8.01,9.01", "2834", [          # results only in the "About ACME" boilerplate
        ("EX-99.1", "PRESS RELEASE", "ACME Bio Announces Relocation of Headquarters\nBOSTON, Nov. 3, 2025 (GLOBE NEWSWIRE) -- ACME "
         "Bio today announced it will move to a new building in Cambridge. The new site adds laboratory space. The new site adds laboratory space. The new site adds laboratory space. The new site adds laboratory space. The new site adds laboratory space. The new site adds laboratory space. The new site adds laboratory space. The new site adds laboratory space. \nAbout ACME Bio\nACME's lead "
         "candidate acmeciclib met its primary endpoint in the Phase 3 ACCEL trial.")]),
    ("B10", 2, "20250915070000", "2.05,8.01,9.01", "2834", [    # restructuring release first, readout in Exhibit 99.2
        ("EX-99.1", "PRESS RELEASE", "Exhibit 99.1 Beta Therapeutics Announces Strategic Actions to Focus on Oncology\n- Implements "
         "cost reduction program\nBeta today announced a restructuring after its Phase 3 study results."),
        ("EX-99.2", "PRESS RELEASE", "Exhibit 99.2 Beta Therapeutics Announces Top-line Results from a Phase 3 Study of BTX-300\n"
         "The randomized, placebo-controlled study did not meet the primary endpoint.")]),
    ("B11", 1, "20250610120000", "8.01,9.01", "2834", [         # preclinical
        ("EX-99.1", "PRESS RELEASE", "ACME Bio Announces Positive Pre-Clinical Data for ACM-303\nIn the study, ACM-303 showed a "
         "statistically significant reduction in tumor growth in mouse models.")]),
    ("B12", 2, "20251208120000", "8.01,9.01", "2834", [         # conference data: a data update, tone unclear
        ("EX-99.1", "PRESS RELEASE", "Beta Therapeutics Presents Updated Data from Ongoing Phase 1 Study of BTX-400 at ASH\nBeta "
         "today announced updated results from the Phase 1 dose-escalation study showing responses in 8 of 12 patients.")]),
    ("B13", 1, "20250714070000", "8.01,9.01", "2834", [         # tone wording seen in the dry run
        ("EX-99.1", "PRESS RELEASE", "ACME Reports Topline Positive Clinical Data from Phase 2 Study of ACM-404\nPrimary endpoint "
         "of induction of response met for both doses compared to placebo, p<0.001.")]),
    # ---- third dry run ----
    ("B14", 1, "20250113120000", "8.01,9.01", "2834", [         # program update; results only in a later CEO quote
        ("EX-99.1", "PRESS RELEASE", "ACME PROVIDES PROGRAM UPDATE AND 2025 MILESTONES\nBOSTON, Jan. 13, 2025 -- ACME Bio "
         "(Nasdaq: ACME) today outlined its plans. The company also reviewed its plans for the year ahead and its cash position. The company also reviewed its plans for the year ahead and its cash position. The company also reviewed its plans for the year ahead and its cash position. The company also reviewed its plans for the year ahead and its cash position. The company also reviewed its plans for the year ahead and its cash position. The company also reviewed its plans for the year ahead and its cash position. The company also reviewed its plans for the year ahead and its cash position. The company also reviewed its plans for the year ahead and its cash position. The company also reviewed its plans for the year ahead and its cash position. The company also reviewed its plans for the year ahead and its cash position. The company also reviewed its plans for the year ahead and its cash position. The company also reviewed its plans for the year ahead and its cash position. The company also reviewed its plans for the year ahead and its cash position. \n“Today we reported preliminary, top-line results from the "
         "Phase 1b study of ACM-505,” said the CEO. The 2.8 mg dose was well tolerated.")]),
    ("B15", 2, "20250114120000", "8.01,9.01", "2834", [         # first patient enrolled; old data in a quote
        ("EX-99.1", "PRESS RELEASE", "BETA ANNOUNCES ENROLLMENT OF FIRST PATIENT IN PHASE 2 STUDY OF BTX-600\nSAN DIEGO, Jan. 14, "
         "2025 -- Beta (Nasdaq: BETA) today announced enrollment of the first patient. “We presented promising data from the "
         "phase 1 portion of the study at ASH,” said the CEO.")]),
    ("B16", 1, "20250116070000", "8.01,9.01", "2834", [         # long contact block before the headline: still a readout
        ("EX-99.1", "EX-99.1", "CONTACT: ACME Bio, Inc. Investor Relations: The Group Timothy Morris, Chief Financial Officer 646-555-2923 CONTACT: ACME Bio, Inc. Investor Relations: The Group Timothy Morris, Chief Financial Officer 646-555-2923 CONTACT: ACME Bio, Inc. Investor Relations: The Group Timothy Morris, Chief Financial Officer 646-555-2923 ACME ANNOUNCES POSITIVE RESULTS FROM PHASE 2 STUDY OF ACM-606\nCAMBRIDGE, "
         "Mass., January 16, 2025 — ACME Bio (NASDAQ: ACME) today announced the study met its primary endpoint.")]),
    # ---- fourth dry run ----
    ("B17", 1, "20250424070000", "7.01,8.01,9.01", "2834", [    # enrollment milestone; results "expected" far after the phrase
        ("EX-99.2", "EX-99.2", "ACME BIO HOLDING CORP. 8-K Exhibit 99.02 ACME Achieves 50 Percent Enrollment in Phase 3 RELIEF Study "
         "of ACM-707\nInterim Results of the First 50 Percent of Participants Expected in September 2025\nTopline Results of "
         "Approximately 470 Participants with Fibromyalgia Expected in the First Quarter of 2026\nNEW YORK, April 24, 2025 (GLOBE "
         "NEWSWIRE) -- ACME Bio today announced enrollment progress.")]),
    ("B18", 1, "20250425070000", "8.01,9.01", "2834", [         # same late "expected", no enrollment headline
        ("EX-99.1", "PRESS RELEASE", "ACME Bio Provides Update on Phase 3 ACCEL-3 Trial\nTopline Results of Approximately 470 "
         "Participants with Fibromyalgia Expected in the First Quarter of 2026\nBOSTON, April 25, 2025 (GLOBE NEWSWIRE) -- ACME "
         "Bio today gave an update.")]),
    # ---- fifth dry run ----
    ("B19", 2, "20250917070000", "8.01,9.01", "2834", [         # headline previews data; first paragraph has no future word
        ("EX-99.1", "PRESS RELEASE", "Beta Therapeutics to Report Data from Randomized Phase 2 Trial of BTX-800 in Fourth Quarter "
         "2025\nSAN DIEGO, Sept. 17, 2025 (GLOBE NEWSWIRE) -- Beta today announced that it is expediting analyses of data from "
         "its randomized Phase 2 trial of BTX-800.")]),
]
fi, metas, docs = [], {1: [], 2: [], 3: []}, {1: [], 2: [], 3: []}
for k, (acc, cik, ts, items, sic, dl) in enumerate(F):
    shard = 1 if k < 4 else 2 if k < 7 else 3
    fi.append({"cik": cik, "accession": acc, "filing_date": pd.Timestamp(ts[:8]), "items_json": items, "form": "8-K"})
    metas[shard].append({"accession": acc, "cik": cik, "acceptance_datetime": ts, "company_header": f"CO{cik}",
                         "sic_header": sic, "items_header": "x", "n_ex99": sum(d[0].startswith("EX-99") for d in dl)})
    for j, (t, desc, txt) in enumerate(dl):
        docs[shard].append({"accession": acc, "doc_type": t, "sequence": str(j + 1), "filename": f"d{j}.htm",
                            "description": desc, "is_html": True, "n_chars": len(txt), "text": txt})
pd.DataFrame(fi).to_parquet(f"{E}/filing_index_sic2834-2835-2836_8-K_2010-01-01_2025-12-31.parquet")
for s in (1, 2, 3):
    pd.DataFrame(metas[s]).to_parquet(f"{S}/meta_{s:05d}.parquet")
    pd.DataFrame(docs[s]).to_parquet(f"{S}/docs_{s:05d}.parquet")

ZUMAB = [f"NCT0600000{i}" for i in range(1, 7)]
pd.DataFrame({"nct_id": ["NCT01234567", "NCT02222222", "NCT07654321"] + ZUMAB,
              "lead_sponsor": ["Acme Bio, Inc.", "Beta Therapeutics", "BigPharma Inc."] + ["Acme Bio, Inc."] * 6,
              "lead_class": ["INDUSTRY"] * 9,
              "interventions": ["ACM101 (acmeciclib) 100 mg;Placebo;ACM101 (2.8 mg)", "BTX-200;Placebo", "GMB-77"] + ["acmezumab"] * 6}).to_parquet(
    f"{G}/industry_trials_current_withcollab.parquet")
pd.DataFrame({"nct_id": ["NCT01234567", "NCT02222222", "NCT07654321", "NCT03333333"] + ZUMAB,
              "sponsor": ["Acme Bio, Inc.", "Beta Therapeutics", "BigPharma Inc.", "Beta Therapeutics"] + ["Acme Bio, Inc."] * 6,
              "role": ["lead"] * 3 + ["collaborator"] + ["lead"] * 6, "cik": [1, 2, 4, 2] + [1] * 6,
              "method": ["exact"] * 10}).to_parquet(f"{G}/trial_firm.parquet")
pd.DataFrame({"sponsor": ["Acme Bio, Inc.", "Beta Therapeutics", "BigPharma Inc.", "Gamma Bio, a BigPharma company"],
              "n_trials": [1, 1, 1, 1], "norm": ["", "", "", ""], "cik": [1, 2, 4, 4],
              "method": ["exact", "exact", "exact", "alias"], "score": [100] * 4, "firm_name": ["", "", "", ""]}).to_csv(
    f"{G}/sponsor_matches.csv", index=False)

EXTRA = {}
def snap(date, rows):
    df = pd.DataFrame(rows + EXTRA.get(date, []))
    df.insert(1, "snapshot_date", date)
    df.to_parquet(f"{G}/aact_monthly/aact_{date[:4]}{date[5:7]}.parquet", index=False)
base = dict(phase="PHASE3", primary_completion_date_type="ESTIMATED", last_update_submitted_date=None,
            results_first_posted_date=None, primary_outcomes_hash="h1")
acme, beta = dict(base, lead_sponsor="Acme Bio, Inc."), dict(base, lead_sponsor="Beta Therapeutics")
EXTRA = {   # stale UNKNOWN record closed; old results that only appear in a later snapshot; a stop on a vaguely-named trial
    "2025-01-01": [dict(beta, nct_id="NCT04444444", overall_status="UNKNOWN", primary_completion_date="2020-01-01",
                        enrollment="50", last_update_posted_date="2019-01-01"),
                   dict(acme, nct_id="NCT05555555", overall_status="COMPLETED", primary_completion_date="2018-01-01",
                        enrollment="50", last_update_posted_date="2019-06-01")],
    "2025-02-01": [dict(beta, nct_id="NCT04444444", overall_status="TERMINATED", primary_completion_date="2020-01-01",
                        enrollment="50", last_update_posted_date="2025-01-25"),
                   dict(acme, nct_id="NCT05555555", overall_status="COMPLETED", primary_completion_date="2018-01-01",
                        enrollment="50", last_update_posted_date="2019-06-01", results_first_posted_date="2019-06-01"),
                   dict(acme, nct_id="NCT06000001", overall_status="RECRUITING", primary_completion_date="2027-01-01",
                        enrollment="100", last_update_posted_date="2024-12-01")],
    "2025-03-01": [dict(acme, nct_id="NCT06000001", overall_status="TERMINATED", primary_completion_date="2027-01-01",
                        enrollment="100", last_update_posted_date="2025-02-15")],
}
big = dict(base, lead_sponsor="BigPharma Inc.")
EXTRA["2025-01-01"] += [dict(acme, nct_id="NCT05555556", overall_status="COMPLETED", primary_completion_date="2024-06-01",
                             enrollment="60", last_update_posted_date="2024-07-01"),
                        dict(big, nct_id="NCT07777777", overall_status="ACTIVE_NOT_RECRUITING", primary_completion_date="2015-06-01",
                             enrollment="500", last_update_posted_date="2016-01-01")]
EXTRA["2025-02-01"] += [dict(acme, nct_id="NCT05555556", overall_status="COMPLETED", primary_completion_date="2024-06-01",
                             enrollment="60", last_update_posted_date="2024-12-15", results_first_posted_date="2024-12-15"),
                        dict(big, nct_id="NCT07777777", overall_status="TERMINATED", primary_completion_date="2015-06-01",
                             enrollment="500", last_update_posted_date="2025-01-20")]
EXTRA["2025-03-01"] = EXTRA.get("2025-03-01", []) + [   # results field missing for a month, then back: still one event
    dict(acme, nct_id="NCT05555556", overall_status="COMPLETED", primary_completion_date="2024-06-01", enrollment="60",
         last_update_posted_date="2024-12-15")]
snap("2024-12-01", [dict(base, nct_id="NCT01234567", overall_status="RECRUITING", primary_completion_date="2024-06-30", enrollment="300",
                         lead_sponsor="Acme Bio, Inc.", last_update_posted_date="2024-11-15"),
                    dict(base, nct_id="NCT07654321", overall_status="RECRUITING", primary_completion_date="2026-01-01", enrollment="80",
                         lead_sponsor="Gamma Bio, Inc.", last_update_posted_date="2024-10-01")])
snap("2025-01-01", [dict(base, nct_id="NCT01234567", overall_status="ACTIVE_NOT_RECRUITING", primary_completion_date="2025-03-31",
                         enrollment="300", lead_sponsor="Acme Bio, Inc.", last_update_posted_date="2024-12-20"),
                    dict(base, nct_id="NCT07654321", overall_status="RECRUITING", primary_completion_date="2026-01-01", enrollment="80",
                         lead_sponsor="Gamma Bio, Inc.", last_update_posted_date="2024-10-01")])
snap("2025-02-01", [dict(base, nct_id="NCT03333333", overall_status="ACTIVE_NOT_RECRUITING", primary_completion_date="2025-01-01",
                         enrollment="40", lead_sponsor="M.D. Anderson Cancer Center", last_update_posted_date="2025-01-20"),
                    dict(base, nct_id="NCT01234567", overall_status="COMPLETED", primary_completion_date="2025-03-31",
                         enrollment="240", lead_sponsor="Acme Bio, Inc.", last_update_posted_date="2025-01-15",
                         primary_outcomes_hash="h2"),
                    dict(base, nct_id="NCT07654321", overall_status="RECRUITING", primary_completion_date="2026-01-01", enrollment="80",
                         lead_sponsor="Gamma Bio, Inc.", last_update_posted_date="2024-10-01")])
snap("2025-03-01", [dict(base, nct_id="NCT01234567", overall_status="COMPLETED", primary_completion_date="2025-03-31",
                         enrollment="240", lead_sponsor="Acme Bio, Inc.", last_update_posted_date="2025-02-26",
                         primary_outcomes_hash="h2", results_first_posted_date="2025-02-25"),
                    dict(base, nct_id="NCT07654321", overall_status="TERMINATED", primary_completion_date="2026-01-01", enrollment="80",
                         lead_sponsor="Gamma Bio, a BigPharma company", last_update_posted_date="2025-02-03"),
                    dict(base, nct_id="NCT03333333", overall_status="COMPLETED", primary_completion_date="2025-01-01", enrollment="40",
                         lead_sponsor="M.D. Anderson Cancer Center", last_update_posted_date="2025-02-10")])
snap("2025-04-01", [dict(acme, nct_id="NCT05555556", overall_status="COMPLETED", primary_completion_date="2024-06-01",
                         enrollment="60", last_update_posted_date="2024-12-15", results_first_posted_date="2024-12-15")])

ALL = [C.CONFIG, C.SETUP, C.LOAD, C.TIMING, C.TEXT, C.REGISTRY, C.DISCLOSE, C.READOUTS, C.SUMMARY]
def run(overrides, cells=ALL):
    ns, buf = {}, io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            for src in cells:
                exec(compile(src, "<cell>", "exec"), ns)
                if src is C.CONFIG:
                    ns.update(PROJECT_DIR=P); ns.update(overrides)
    except Exception:
        print(buf.getvalue()); raise
    print(buf.getvalue())
    return ns

ns = run({"DRY_RUN": False})
fl = pd.read_parquet(f"{P}/features/filings.parquet").set_index("accession")
exp = {"A1": ("2025-01-10", "after_close"),      # Jan 9 2025: markets closed (national day of mourning)
       "A2": ("2025-01-10", "pre_open"), "A3": ("2025-01-10", "market_hours"),
       "A4": ("2025-01-13", "after_close"),      # Saturday
       "A5": ("2025-12-01", "after_close"),      # day after Thanksgiving closes at 1pm
       "A6": ("2025-02-21", "after_close"), "A7": ("2018-03-05", "pre_open")}
for a, (d, t) in exp.items():
    assert str(fl.loc[a, "event_date"])[:10] == d and fl.loc[a, "timing"] == t, (a, fl.loc[a, ["event_date", "timing"]].tolist())
assert fl.loc["A1", "items"] == "8.01;9.01" and not fl.loc["A7", "sic_ok"] and fl.loc["A1", "sic_ok"]

tf = pd.read_parquet(f"{P}/features/filing_text_features.parquet").set_index("accession")
assert tf.loc["A1", "readout"] and tf.loc["A1", "readout_tone"] == "positive" and tf.loc["A1", "nct_ids"] == "NCT01234567"
assert tf.loc["A1", "text_source"] == "press_release" and tf.loc["A1", "has_press_release"]
assert tf.loc["A2", "has_slides"] and not tf.loc["A2", "readout"]
assert tf.loc["A3", "readout"] and tf.loc["A3", "readout_tone"] == "negative" and tf.loc["A3", "text_source"] == "press_release"
assert tf.loc["A4", "fda_crl"] and not tf.loc["A4", "readout"]
assert not tf.loc["A5", "readout"]
assert tf.loc["A7", "readout"] and tf.loc["A7", "readout_tone"] == "positive" and tf.loc["A7", "text_source"] == "form_8k"
for a in ["B1", "B2", "B3", "B4", "B6", "B7", "B8"]:
    assert not tf.loc[a, "readout"], (a, tf.loc[a, ["readout_rule", "readout_snippet"]].tolist())
assert tf.loc["B1", "text_source"] == "press_release" and not tf.loc["B1", "fda_approval"]          # "expect FDA approval"
assert tf.loc["B2", "text_source"] == "press_release"                                              # "today reported", no wire
assert tf.loc["B5", "kinds"] == "form_8k;press_release" and not tf.loc["B5", "has_slides"]         # cover page is not slides
assert tf.loc["B5", "readout"] and tf.loc["B5", "readout_tone"] == "positive" and tf.loc["B5", "readout_rule"] == "results_from"
assert "met its primary endpoint" in tf.loc["B5", "readout_snippet"] and tf.loc["B5", "headline"].startswith("Contact")
assert not tf.loc["B6", "fda_approval"] and tf.loc["B7", "fda_approval"]
assert tf.loc["A1", "readout_rule"] in ("topline", "results_from") and tf.loc["A3", "readout_rule"] == "topline"
assert tf.loc["A1", "readout_kind"] == "topline_or_endpoint" and tf.loc["A7", "readout_kind"] == "topline_or_endpoint"
assert not tf.loc["B9", "readout"], tf.loc["B9", "readout_snippet"]                                # boilerplate is skipped
assert tf.loc["B10", "readout"] and tf.loc["B10", "readout_tone"] == "negative", tf.loc["B10"].to_dict()
assert tf.loc["B10", "headline"].startswith("Beta Therapeutics Announces Top-line"), tf.loc["B10", "headline"]
assert tf.loc["B1", "headline"].startswith("ACME Bio Expects")                                    # no readout: first document
assert not tf.loc["B11", "readout"]                                                                # preclinical
assert tf.loc["B12", "readout"] and tf.loc["B12", "readout_kind"] == "data_update" and tf.loc["B12", "readout_tone"] == "unclear"
assert tf.loc["B13", "readout_tone"] == "positive" and tf.loc["B13", "pos_hits"] >= 2, tf.loc["B13"].to_dict()
assert not tf.loc["B14", "readout"] and not tf.loc["B15", "readout"], tf.loc[["B14", "B15"], "readout_snippet"].tolist()
assert not tf.loc["B17", "readout"] and not tf.loc["B18", "readout"], tf.loc[["B17", "B18"], "readout_snippet"].tolist()
assert not tf.loc["B19", "readout"], tf.loc["B19", "readout_snippet"]
assert tf.loc["B17", "headline"].startswith("ACME Achieves 50 Percent"), tf.loc["B17", "headline"]
assert tf.loc["B16", "readout"] and tf.loc["B16", "readout_tone"] == "positive" and tf.loc["B16", "text_source"] == "press_release"

lk = pd.read_parquet(f"{P}/features/filing_trial_links.parquet")
got = set(zip(lk["accession"], lk["nct_id"], lk["link_type"]))
assert ("A1", "NCT01234567", "nct") in got and ("A1", "NCT01234567", "drug_name") in got     # "acmeciclib"
assert ("A2", "NCT01234567", "drug_name") in got                                           # "ACM-101" vs registry "ACM101"
assert ("A6", "NCT07654321", "nct") in got
assert ("A3", "NCT02222222", "drug_name") in got                                           # BTX-200 is in BETA's own trial
assert not any(a == "A4" for a, _, _ in got)                                                # BTX-100 is not in the registry
print("links:", sorted(got))
assert not (lk["matched"] == "2.8 mg").any()                                                       # doses are not drug names
zl = lk[lk["accession"] == "B8"]
assert len(zl) == 6 and (zl["n_name_trials"] == 6).all() and lk.loc[lk["link_type"] == "nct", "n_name_trials"].isna().all()

re_ = pd.read_parquet(f"{P}/features/registry_events.parquet")
def one(nct, kind):
    r = re_[(re_["nct_id"] == nct) & (re_["event_type"] == kind)]
    assert len(r) == 1, (nct, kind, re_[["nct_id", "event_type", "detail"]])
    return r.iloc[0]
r = one("NCT01234567", "pcd_later")
assert r["detail"] == "9 months" and str(r["registry_date"])[:10] == "2024-12-20" and r["date_source"] == "posted"
sts = re_[(re_["nct_id"] == "NCT01234567") & (re_["event_type"] == "status")].sort_values("registry_date")
assert list(sts["detail"]) == ["RECRUITING -> ACTIVE_NOT_RECRUITING", "ACTIVE_NOT_RECRUITING -> COMPLETED"], sts["detail"]
assert not sts["is_stop"].any()
e = one("NCT01234567", "enrollment_down"); assert e["detail"] == "300 -> 240"
assert one("NCT01234567", "outcome_edit")["registry_date"].strftime("%Y-%m-%d") == "2025-01-15"
rp = one("NCT01234567", "results_posted")
assert rp["date_source"] == "results_posted" and str(rp["registry_date"])[:10] == "2025-02-25", rp
assert re_[(re_["nct_id"] == "NCT05555555") & (re_["event_type"] == "results_posted")].empty   # 2019 results: not a new event
u = one("NCT04444444", "status")
assert u["detail"] == "UNKNOWN -> TERMINATED" and not u["is_stop"]                                # stale record, not a stop
n6 = re_[(re_["nct_id"] == "NCT05555556") & (re_["event_type"] == "results_posted")]
assert len(n6) == 1 and str(n6["registry_date"].iloc[0])[:10] == "2024-12-15", n6               # late field, counted once
s7 = one("NCT07777777", "status")
assert s7["stop_any"] and s7["stale_record"] and not s7["is_stop"]                                 # 2015 trial closed in 2025
z = one("NCT06000001", "status")
assert z["is_stop"] and not z["stale_record"] and pd.isna(z["days_to_8k"]) and z["firm_lead_trials"] == 7     # B8 names acmezumab (6 trials) 5 days later: too vague to count
st = one("NCT07654321", "status")
assert st["firm_8ks_year"] == 0 and z["firm_8ks_year"] > 0   # BigPharma (cik 4) filed only in 2018; ACME files often
assert st["is_stop"] and str(st["registry_date"])[:10] == "2025-02-03" and int(st["cik"]) == 4 and st["owner_source"] == "sponsor_at_time"
assert st["days_to_8k"] == 18 and not st["disclosed_before"]
p1 = one("NCT01234567", "pcd_later")
assert int(p1["cik"]) == 1 and p1["days_to_8k"] == 21   # first 8-K mentioning the trial after 2024-12-20 is A1 (market date 2025-01-10)

co = one("NCT03333333", "status")
assert int(co["cik"]) == 2 and co["owner_source"] == "collaborator", co
ro = pd.read_parquet(f"{P}/features/readouts.parquet")
assert set(ro["accession"]) == {"A1", "A3", "A7", "B5", "B10", "B12", "B13", "B16"}, ro["accession"].tolist()
occ = pd.read_csv(f"{P}/features/readouts_occ_window.csv")
assert set(occ["ticker"]) == {"ACME", "BETA"} and set(occ["accession"]) == {"A1", "A3", "B5", "B10", "B12", "B13", "B16"}, occ
print("TEST FULL OK\n" + "=" * 70)

# dry run: 2 shards spread over the list (first and last); registry and summary still work; separate output files
ns = run({"DRY_RUN": True, "DRY_RUN_SHARDS": 2})
assert os.path.exists(f"{P}/features/filing_text_features_dryrun.parquet")
assert os.path.isdir(f"{P}/features/text_shards_v6_dryrun")
d = pd.read_parquet(f"{P}/features/filing_text_features_dryrun.parquet")
assert set(d["accession"]) == {"A1", "A2", "A3", "A4"} | {f"B{i}" for i in range(1, 20)}, sorted(d["accession"])
print("TEST DRY RUN OK")

# no AACT yet: steps 4-5 skipped cleanly
shutil.rmtree(f"{G}/aact_monthly")
ns = run({"DRY_RUN": True, "DRY_RUN_SHARDS": 1})
assert ns["registry_events"] is None
print("TEST NO-AACT OK\nALL TESTS PASSED")

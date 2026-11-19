"""Offline test of Notebook 5: synthetic 8-K shards, fake LLM (rule-based JSON, a malformed reply, one out-of-memory), fake embedder."""
import io, os, re, sys, json, shutil, contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "tools"))
from cells import load_stage
C = load_stage("05_release_text")

P = "/tmp/nb5_test"
shutil.rmtree(P, ignore_errors=True)
F, SH, CT = f"{P}/features", f"{P}/edgar/8k_text/shards", f"{P}/ctgov"
for d in (F, SH, CT):
    os.makedirs(d)
rng = np.random.default_rng(3)

KINDS = {  # outcome -> (sentence, typical |move|, sign)
    "fail": ("The Phase 3 [X] trial did not meet its primary endpoint. The difference was not statistically significant (p=0.41).", 0.35, -1),
    "win": ("The Phase 3 trial met its primary endpoint with a statistically significant improvement (p<0.001).", 0.14, 1),
    "mixed": ("The trial showed a numerical trend that may suggest benefit in a subgroup; the primary endpoint was not met.", 0.15, -1),
    "update": ("Updated data from the ongoing Phase 1 study were presented at a medical conference.", 0.05, 1),
}
rows, docs, tf, cur = [], [], [], []
for i in range(600):
    cik = 100 + i % 40
    co, tk, drug, code = f"Zentra{cik} Therapeutics, Inc.", f"ZT{cik}", f"zentramab{cik}", f"ZTX-{cik}1"
    kind = ["fail", "win", "win", "mixed", "update", "update"][i % 6]
    sent, size, sign = KINDS[kind]
    year = 2012 + i % 14
    day = pd.Timestamp(f"{year}-03-01") + pd.Timedelta(days=int(i // 14) * 5)
    car = sign * size * float(rng.lognormal(0, 0.35))
    tone_true = {"fail": "negative", "win": "positive", "mixed": "mixed", "update": "unclear"}[kind]
    tone = tone_true if rng.random() > 0.35 else rng.choice(["positive", "negative", "mixed", "unclear"])   # noisy rule labels
    acc, acc2 = f"0000{i:06d}-A", f"0000{i:06d}-B"
    marker = " MALFORMED" if i % 97 == 0 else ""
    pr = (f"EX-99.1\nBOSTON, Mar. 1, {year} (GLOBE NEWSWIRE) -- {co} (Nasdaq: {tk}) today announced topline results for "
          f"{drug} ({code}) in the STELLAR{i} trial (NCT0{i:07d}). {sent.replace('[X]', 'STELLAR' + str(i))} No new safety signals. "
          f"COVID-19, IL-6 and BRAF V600E are mentioned for context.{marker} " + "Further details. " * 20 +
          f"\nAbout {co}\n{co} is a biotech. Forward-looking statements: blah blah " + "legal " * 200)
    docs += [{"accession": acc, "doc_type": "8-K", "description": "8-K", "filename": "f.htm",
              "text": "UNITED STATES SECURITIES AND EXCHANGE COMMISSION cover page ... Item 8.01 Other Events. See Exhibit 99.1."},
             {"accession": acc, "doc_type": "EX-99.1", "description": "Press Release", "filename": "ex991.htm", "text": pr},
             {"accession": acc, "doc_type": "EX-99.2", "description": "Corporate presentation", "filename": "ex992.htm",
              "text": "slide\n" * 100}]
    same = acc
    if i % 10 == 0:                                         # a second 8-K that day
        docs.append({"accession": acc2, "doc_type": "8-K", "description": "8-K", "filename": "g.htm",
                     "text": "cover ... Item 7.01 Regulation FD. The company will host a call to discuss the results."})
        same = f"{acc};{acc2}"
    rows.append({"accession": acc, "cik": cik, "company": co, "ticker": tk, "event_date": day, "day0": day, "timing": "pre_open",
                 "readout_kind": "data_update" if kind == "update" else "topline_or_endpoint", "readout_tone": tone,
                 "earnings_day": i % 25 == 0, "has_prices": i % 50 != 7, "car_1_1": car, "abs_car": abs(car),
                 "pre_vol": 0.04 * float(rng.lognormal(0, 0.3)), "post_vol": 0.04 * float(rng.lognormal(0, 0.3)),
                 "log_dollar_vol": float(rng.normal(15, 1.5)), "car_m10_m2": float(rng.normal(0, 0.05)), "car_2_20": float(rng.normal(0, 0.06)),
                 "same_day_accessions": same if i % 10 else f"{acc};{acc2}", "phase": [3.0, 2.0, np.nan][i % 3], "year": year})
    tf.append({"nct_id": f"NCT0{i:07d}", "cik": cik, "role": "lead", "sponsor": co})
    cur.append({"nct_id": f"NCT0{i:07d}", "interventions": f"Drug: {drug} 10 mg;Drug: Placebo", "phases": "PHASE3"})
rows[5]["same_day_accessions"] = np.nan                     # a missing list
ev = pd.DataFrame(rows)
ev.to_parquet(f"{F}/readout_events.parquet", index=False)
dd = pd.DataFrame(docs)
for k, part in enumerate(np.array_split(dd.index, 4)):
    dd.loc[part].to_parquet(f"{SH}/docs_{k:04d}.parquet", index=False)
open(f"{SH}/docs_9999.parquet", "w").write("not a parquet file")      # an unreadable shard
pd.DataFrame(tf).drop_duplicates(["nct_id", "cik", "role"]).to_parquet(f"{CT}/trial_firm.parquet", index=False)
pd.DataFrame(cur).to_parquet(f"{CT}/industry_trials_current.parquet", index=False)

class Fakes:
    def __init__(self):
        self.calls, self.prompts, self.oom_once, self.gen_calls = 0, 0, True, 0
        self.embed_calls = 0
    def generate(self, prompts, max_new_tokens=160):
        self.calls += 1
        self.gen_calls += 1
        self.prompts += len(prompts)
        if self.oom_once and len(prompts) > 1:
            self.oom_once = False
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        out = []
        for sysm, user in prompts:
            t = user.lower()
            strict = user.rstrip().endswith("no other text.")
            if "malformed" in t and not strict:
                out.append("Sure! Here is my analysis: the endpoint was probably met.")
                continue
            if "did not meet" in t and "trend" not in t:
                a = {"Endpoint": "Not met", "stat_sig": False, "numbers": True, "safety": "none", "hedging": 0, "uncertainty": 2, "impact": -3, "move_pct": "40-50%"}
            elif "trend" in t:
                a = dict(endpoint="partially met", stat_sig="no", numbers="no", safety="mild", hedging=3, uncertainty=8, impact=-1, move_pct=15)
            elif "met its primary" in t:
                a = dict(endpoint="met", stat_sig="yes", numbers="yes", safety="none", hedging=0, uncertainty=2, impact="+2", move_pct="about 20")
            else:
                a = dict(endpoint="not stated", stat_sig="not reported", numbers="no", safety="not mentioned", hedging=1, uncertainty=5, impact=0, move_pct=3)
            if "zentra" in t and "[company]" not in t:             # unmasked text: the 'model' recognizes the case
                a["impact"] = a["impact"]
            out.append("```json\n" + json.dumps(a) + "\n```")
        return out
    def score(self, prompts, letters):
        self.calls += 1
        self.prompts += len(prompts)
        P = []
        for sysm, user in prompts:
            u = user.lower()
            big = "did not meet" in u and "trend" not in u
            if len(letters) == 5:      # move bins
                w = [0.05, 0.1, 0.15, 0.3, 0.4] if big else [0.1, 0.2, 0.4, 0.2, 0.1] if "met its primary" in u or "trend" in u \
                    else [0.7, 0.2, 0.05, 0.03, 0.02]
            else:                      # outcome
                w = [0.05, 0.9, 0.03, 0.02] if big else [0.1, 0.2, 0.6, 0.1] if "trend" in u else \
                    [0.85, 0.05, 0.05, 0.05] if "met its primary" in u else [0.1, 0.1, 0.1, 0.7]
            P.append(w)
        P = np.array(P, dtype=float)
        return P / P.sum(1, keepdims=True), np.full(len(P), 0.93)
    def embed(self, texts):
        self.embed_calls += 1
        V = np.zeros((len(texts), 64))
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                V[i, hash(w) % 64] += 1
        return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)

CELLS = [C.CONFIG, C.SETUP, C.TEXTS, C.DICT, C.LLM, C.EMBED, C.TESTS]

def run(overrides, fk):
    ns, buf = {}, io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            for src in CELLS:
                exec(compile(src, "<cell>", "exec"), ns)
                if src is C.CONFIG:
                    ns.update(PROJECT_DIR=P, **overrides)
                if src is C.SETUP:
                    ns.update(GENERATE=fk.generate, SCORE=fk.score, EMBED=fk.embed, MODEL_USED="fake/model-8b")
    except Exception:
        print(buf.getvalue())
        raise
    print(buf.getvalue())
    return ns

# 1) dry run
fk = Fakes()
ns = run({"DRY_RUN": True, "DRY_RUN_N": 40, "BATCH": 8, "LLM_MODE": "full", "LLM_SCOPE": "all", "UNMASKED_CHECK_N": 300}, fk)
assert os.path.exists(f"{F}/nb5/texts_dryrun.parquet") and fk.prompts > 0
print("TEST DRY RUN OK\n" + "=" * 70)

# 2) full run
fk = Fakes()
FULL = {"DRY_RUN": False, "BATCH": 8, "LLM_MODE": "full", "LLM_SCOPE": "all", "UNMASKED_CHECK_N": 300}
ns = run(FULL, fk)
ev5 = ns["ev"]
assert len(ev5) == (ev["has_prices"]).sum(), (len(ev5), ev["has_prices"].sum())
txt = pd.read_parquet(f"{F}/nb5/texts.parquet").set_index("accession")
t0 = txt.loc["0000000000-A"]
assert t0["source"] == "press_release" and t0["n_filings"] == 2 and "host a call" in t0["text"], t0[["source", "n_filings"]]
m = t0["text"]
for bad in ["Zentra100", "zentramab100", "ZTX-1001", "ZT100", "STELLAR0", "NCT00000000"]:
    assert bad.lower() not in m.lower(), (bad, m[:400])
for keep in ["COVID-19", "IL-6", "V600E", "[COMPANY]", "[DRUG]", "[TRIAL]", "[TRIAL ID]", "[TICKER]"]:
    assert keep in m, (keep, m[:600])
assert "legal legal" not in m and "Forward-looking" not in m, "boilerplate cut"
assert "Zentra100" in t0["text_raw"]
llm = pd.read_parquet(f"{F}/nb5/llm_fake-model-8b_p3_full.parquet")
assert llm["parsed"].all() and llm["retried"].sum() >= 5, llm["retried"].sum()
assert set(llm["endpoint"]) >= {"met", "not_met", "mixed", "not_stated"}
assert llm["exp_move"].notna().all() and llm["exp_move"].between(2.5, 55).all() and llm["p_met"].between(0, 1).all() and llm["exp_move"].round(1).nunique() >= 3
assert (llm.filter(like="p_move_").sum(axis=1).round(6) == 1).all()
assert set(llm["safety"]) >= {"none", "minor"} and set(llm["move_pct"]) >= {45.0, 15.0, 20.0, 3.0} and llm["impact"].notna().all()
raw_parts = os.path.join(F, "nb5", "llm_fake-model-8b_p3_full_unmasked_parts")
assert os.path.isdir(raw_parts) and len(ns["llm_raw"]) > 50
feat = pd.read_parquet(f"{F}/nb5/readout_text_features.parquet")
assert {"pred_controls", "pred-labels", "pred-llm", "pred-all"} & set(feat.columns) or any(c.startswith("pred_") for c in feat.columns)
res = ns["res_y"]
r2 = {k: v[0]["test, all"][0] for k, v in res.items()}
assert r2["+ LLM"] > r2["+ labels"] + 0.05 and r2["+ dictionary"] > r2["+ labels"], r2
assert os.path.exists(f"{F}/figures/fig9_oos_r2.png") and os.path.exists(f"{F}/figures/fig10_llm_move.png")
assert fk.embed_calls == 1
print("TEST FULL OK\n" + "=" * 70)

# 3) resume: no new LLM or embedding calls, same answers
fk = Fakes()
ns = run(FULL, fk)
assert fk.calls == 0 and fk.embed_calls == 0, (fk.calls, fk.embed_calls)
print("TEST RESUME OK\n" + "=" * 70)

# 4) a prompt change asks again; MAX_CHARS change rebuilds texts
fk = Fakes()
ns = run({"DRY_RUN": True, "DRY_RUN_N": 40, "PROMPT_VERSION": "p4", "MAX_CHARS": 3000, "UNMASKED_CHECK_N": 0}, fk)
t = pd.read_parquet(f"{F}/nb5/texts_dryrun.parquet")
assert (t["version"] == "t1_3000").sum() == 40 and fk.prompts >= 40
print("TEST VERSIONS OK\n" + "=" * 70)

# 5) lean mode on topline readouts: no generation, scored questions only, LLM rows in the tests
fk = Fakes()
ns = run({"DRY_RUN": False, "BATCH": 8}, fk)
lean = pd.read_parquet(f"{F}/nb5/llm_fake-model-8b_p3_lean.parquet")
assert fk.gen_calls == 0 and lean["exp_move"].notna().all() and lean["endpoint"].isna().all()
assert len(lean) == (ns["ev"]["readout_kind"] == "topline_or_endpoint").sum()
r2 = {k: v[0]["test, all"][0] for k, v in ns["res_y"].items()}
assert "+ LLM" in r2 and r2["+ LLM"] > r2["+ labels"], r2
print("TEST LEAN OK\n" + "=" * 70)

# 6) no LLM, no embeddings: the rest still runs
fk = Fakes()
ns = run({"DRY_RUN": False, "RUN_LLM": False, "RUN_EMBED": False}, fk)
assert fk.calls == 0 and "+ LLM" not in ns["res_y"] and "+ dictionary" in ns["res_y"]
print("TEST NO LLM OK\nALL TESTS PASSED")

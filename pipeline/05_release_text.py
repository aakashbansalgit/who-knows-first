"""Notebook 5: Text-implied volatility of trial readouts

Pipeline stage of the who-knows-first replication code. Cells are separated by '# %%' markers, so the file
runs top to bottom as a script and opens as a notebook in Jupyter, VS Code or (after tools/make_notebooks.py) Colab.
Needs a GPU runtime for the language-model step.
See README.md for inputs, outputs and the run order.
"""


# %% [markdown]
# # Notebook 5: Text-implied volatility of trial readouts
#
# **Question.** Does the wording of a readout press release predict how big the stock move will be (|abnormal return|, days -1 to +1), beyond the readout labels and the stock's own volatility?
#
# **What it does**
# 1. **Texts.** For each readout with prices (Notebook 4), the first part of the press release (headline and first paragraphs, before the "About the company" and forward-looking-statement sections), from Notebook 1's 8-K files. All 8-Ks the firm filed that market day are used. Company names, tickers, the firm's drug names and code names (e.g. ABC-123), NCT numbers and trial acronyms are replaced by `[COMPANY]`, `[DRUG]`, `[TRIAL]`, so the model can't simply recognize the case.
# 2. **Dictionary measures.** Rates of hedging words, success and failure phrases, safety words, p-values and numbers.
# 3. **LLM reading** with an open-weights model on Colab's GPU (default **Llama 3.1 8B Instruct**, training data up to December 2023). **Lean mode (default, `LLM_MODE = "lean"`)** reads only the two probability-scored questions described below, on topline readouts only (`LLM_SCOPE = "topline"`): about 30-60 minutes on a free Colab T4, so it fits free GPU time. **Full mode** also asks for written answers: for each release, primary endpoint met / not met / mixed / not stated, statistical significance, whether numbers are given, safety, hedging (0-3), uncertainty (1-10), likely impact on the company's value (-3 to +3) and its best guess of the size of the move in percent. Greedy decoding, so the same text gives the same answer. Two questions are also scored from the model's probabilities for each answer letter (one forward pass each): the size of the move (5 bins), giving a probability-weighted **expected move** and its **spread**, and whether the primary endpoint was met, giving **P(met)** for Notebook 6. These give a continuous score even when a small model's written answers bunch on a few values.
# 4. **Embeddings** (`bge-base`) with a ridge regression, as a middle ground.
# 5. **Tests.** Fit on readouts up to `TRAIN_END` (2020), test on later ones. Out-of-sample R² and rank correlation for: controls only; + the Notebook 3 labels; + dictionary; + LLM; + embeddings; + all. The test years are split at the model's training cutoff: **2024-2025 readouts are ones the model cannot have seen**. A subsample is also read without masking: if the model does much better on unmasked pre-cutoff readouts, it is recognizing them.
#
# **How to run**
# - Runtime > Change runtime type > **GPU** (T4 works; L4 or A100 is about 3-5x faster).
# - **Llama needs two things once:** accept the license at huggingface.co/meta-llama/Llama-3.1-8B-Instruct (Hugging Face account, approval is usually quick), and add a Hugging Face read token to Colab Secrets as `HF_TOKEN`. Without them it falls back to Qwen 2.5 7B (no license step, but its training cutoff isn't published, so the cutoff test is weaker).
# - **First run:** `DRY_RUN = True` (40 readouts, about 5-10 minutes including the model download).
# - **Full run:** `DRY_RUN = False`. Lean mode on topline readouts (about 1,400): roughly 30-60 minutes on a free T4. Full mode on all readouts: 5-7 hours on a T4. Answers are saved every batch, so if the free GPU time runs out, rerun from the top another day: it continues where it stopped.
# - Needs Notebook 4 **v2** output (`readout_events.parquet` with `same_day_accessions`).
#
# **Output** (in `MyDrive/biopharma_tiv/features/nb5/`): `texts.parquet`, `dictionary.parquet`, `llm_<model>_<prompt>.parquet` (one row per readout, with the raw answer), `embeddings_<model>.npy`, `readout_text_features.parquet` (everything merged), figures `fig9_oos_r2.png`, `fig10_llm_move.png`.
#
# Llama 3.1 is under Meta's community license; Qwen 2.5 under Apache 2.0. Both are fine for academic research.


# %% CONFIG
# ==== Settings ====
import os
PROJECT_DIR = os.environ.get("WKF_PROJECT_DIR", "/content/drive/MyDrive/biopharma_tiv")
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"   # license accepted on Hugging Face + HF_TOKEN in Colab Secrets
MODEL_CUTOFF = "2023-12-31"                      # training-data cutoff of MODEL_ID (Llama 3.1: December 2023)
FALLBACK_MODEL = "Qwen/Qwen2.5-7B-Instruct"      # used if MODEL_ID can't be loaded ("" = no fallback)
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
PROMPT_VERSION = "p3"                            # change when the prompt changes: answers are cached per prompt version
DRY_RUN = True                                   # True: DRY_RUN_N readouts, outputs end in _dryrun
DRY_RUN_N = 40
RUN_LLM = True
LLM_MODE = "lean"                                # "lean": only the two probability-scored questions (about 4x faster, fits
                                                 #   free GPU time); "full": also the written answers (endpoint, hedging...)
LLM_SCOPE = "topline"                            # "topline": topline / primary-endpoint readouts only; "all": also data updates
RUN_EMBED = True
UNMASKED_CHECK_N = 200                           # test-period readouts also read without masking (0 = skip)
MAX_CHARS = 5000                                 # characters of release text given to the models
BATCH = 8                                        # releases per GPU batch (halved automatically on out-of-memory)
TRAIN_END = 2020                                 # fit on readouts up to this year, test on later years
WINSOR = 0.01


# %% [markdown]
# ## Step 0: Setup


# %% SETUP
import os, re, io, json, glob, time, random, warnings, datetime as dt
import subprocess, sys
import numpy as np
import pandas as pd

def need_pkg(mod, pip_name=None):
    try:
        __import__(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or mod], check=True)

for mod, pipn in [("statsmodels", None), ("sklearn", "scikit-learn"), ("scipy", None), ("pyarrow", None)]:
    need_pkg(mod, pipn)
import pyarrow.parquet as pq
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

if PROJECT_DIR.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")

FEAT_DIR = os.path.join(PROJECT_DIR, "features")
SHARD_DIR = os.path.join(PROJECT_DIR, "edgar", "8k_text", "shards")
CTG_DIR = os.path.join(PROJECT_DIR, "ctgov")
NB5_DIR = os.path.join(FEAT_DIR, "nb5")
FIG_DIR = os.path.join(FEAT_DIR, "figures")
for d in (NB5_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)
SUFFIX = "_dryrun" if DRY_RUN else ""
CUTOFF = pd.Timestamp(MODEL_CUTOFF)
MODEL_USED = None

def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")

def hf_token():
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:
        return os.environ.get("HF_TOKEN")

def load_llm():
    """Loads MODEL_ID (or FALLBACK_MODEL) on the GPU; returns generate(prompts) -> replies, prompts being (system, user) pairs.
    Full precision on GPUs with 20+ GB, 4-bit otherwise (T4)."""
    global MODEL_USED
    import torch
    assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > GPU, then rerun from the top (or set RUN_LLM = False)."
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gb < 20:
        need_pkg("bitsandbytes")
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    if gb >= 20:
        kw = {"torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    else:
        kw = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                        bnb_4bit_compute_dtype=torch.float16)}
    token, last = hf_token(), None
    for mid in [MODEL_ID] + ([FALLBACK_MODEL] if FALLBACK_MODEL else []):
        try:
            tok = AutoTokenizer.from_pretrained(mid, token=token)
            model = AutoModelForCausalLM.from_pretrained(mid, token=token, device_map="cuda", **kw)
            break
        except Exception as e:
            last = e
            print(f"Could not load {mid}: {type(e).__name__}: {str(e)[:300]}")
    else:
        raise RuntimeError("No model loaded. For Llama: accept the license at huggingface.co/meta-llama/Llama-3.1-8B-Instruct "
                           "and add a Hugging Face token to Colab Secrets as HF_TOKEN.") from last
    MODEL_USED = mid
    if mid != MODEL_ID:
        print(f"Using {mid}: its training cutoff isn't published, so MODEL_CUTOFF ({MODEL_CUTOFF}) is only a guess for it.")
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    print(f"Model: {mid} | GPU memory {gb:.0f} GB | {'4-bit' if gb < 20 else 'full precision'}")

    def letter_ids(L):
        return sorted({tok.encode(v, add_special_tokens=False)[0] for v in (L, " " + L)})

    def score(prompts, letters):
        """Probability of each answer letter as the first token of the reply (one forward pass, no generation).
        Returns (probabilities renormalized over the letters, the raw probability mass on the letters)."""
        chats = [tok.apply_chat_template([{"role": "system", "content": s_}, {"role": "user", "content": u}],
                                         tokenize=False, add_generation_prompt=True) for s_, u in prompts]
        enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)     # positions as in generate(), with left padding
            h = model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                            position_ids=pos).last_hidden_state[:, -1, :]      # last position only: no full-vocabulary logits
            probs = torch.softmax(model.lm_head(h).float(), dim=-1)
        P = torch.stack([probs[:, letter_ids(L)].sum(dim=1) for L in letters], dim=1)
        mass = P.sum(dim=1, keepdim=True)
        out = (P / mass).cpu().numpy(), mass[:, 0].cpu().numpy()
        del enc, h, probs
        torch.cuda.empty_cache()
        return out

    def generate(prompts, max_new_tokens=120):
        chats = [tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": u}],
                                         tokenize=False, add_generation_prompt=True) for s, u in prompts]
        enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        res = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        del enc, out
        torch.cuda.empty_cache()
        return res
    return generate, score

def load_embedder():
    need_pkg("sentence_transformers", "sentence-transformers")
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    return lambda texts: np.asarray(m.encode(list(texts), batch_size=32, normalize_embeddings=True, show_progress_bar=False))

def as_bool(s):
    return pd.Series(s).astype("boolean").fillna(False).astype(bool)

def winsor_fit(s, q=WINSOR):
    """Clipping bounds from s (the training years)."""
    s = pd.Series(s, dtype=float).dropna()
    return (s.quantile(q), s.quantile(1 - q)) if len(s) >= 20 and q else (-np.inf, np.inf)

def cluster_groups(firm, date=None):
    f = pd.factorize(pd.Series(firm).astype(str))[0]
    if len(set(f)) < 10:
        return None
    if date is None:
        return f
    d = pd.factorize(pd.Series(date).astype(str))[0]
    return np.column_stack([f, d]) if len(set(d)) >= 10 else f
print("Saving to:", NB5_DIR)


# %% [markdown]
# ## Step 1: Press-release texts, masked
# One text per readout: the press release of the readout 8-K (else another Exhibit 99, else the 8-K's items), then the other 8-Ks the firm filed that market day, up to `MAX_CHARS`. Same document choice as Notebook 3. Saved, so later runs skip this step.


# %% TEXTS
TEXT_PATH = os.path.join(NB5_DIR, f"texts{SUFFIX}.parquet")
ev = pd.read_parquet(os.path.join(FEAT_DIR, "readout_events.parquet"))
missing_cols = {"same_day_accessions", "car_m10_m2", "log_dollar_vol", "abs_car"} - set(ev.columns)
assert not missing_cols, f"readout_events.parquet is from an older Notebook 4 (no {sorted(missing_cols)}): run Notebook 4 v2 first."
ev = ev[ev["has_prices"].astype(bool)].copy()
ev["event_date"] = pd.to_datetime(ev["event_date"])
ev["year"] = ev["event_date"].dt.year
if DRY_RUN:
    ev = ev.sample(min(DRY_RUN_N, len(ev)), random_state=1)
ev = ev.reset_index(drop=True)
print(f"Readouts with prices: {len(ev):,} | years {ev['year'].min()}-{ev['year'].max()}")

# ---- Notebook 3's document choice ----
WIRE = re.compile(r"GLOBE\s?NEWSWIRE|BUSINESS\s?WIRE|PR\s?NEWSWIRE|ACCESS\s?(?:NEWS)?WIRE|MARKETWIRED?|PRWEB|EQS\s?NEWSWIRE", re.I)
PR_STRONG = re.compile(r"for\s+immediate\s+release|\bpress\s+release\b|\bnews\s+release\b|"
                       r"\btoday\s+(?:announced|reported|provided|issued|unveiled|presented)\b", re.I)
PR_WEAK = [re.compile(r"\((?:nasdaq|nyse|amex|tsx|otc)[a-z\s]*:\s*[a-z.]{1,6}\)", re.I),
           re.compile(r"\b[A-Z][A-Z .'-]{2,},\s+(?:[A-Z][a-z]+\.?,\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                      r"[a-z]*\.?\s+\d{1,2},?\s+(?:19|20)\d{2}"),
           re.compile(r"\b(?:investor|media)\s+(?:relations|contacts?)\b|\bcontacts?\s*:", re.I)]
PR_HINT = re.compile(r"press|release|\bnews\b", re.I)
DECK_HINT = re.compile(r"presentation|slides?\b|deck|corporate overview|investor day|poster|r&d day", re.I)
TRANSCRIPT = re.compile(r"transcript", re.I)
OPERATOR = re.compile(r"^\s*Operator\s*:?\s*$|\bOperator\s*:", re.M)
ITEM_HEAD = re.compile(r"\bItem\s+\d{1,2}\.\d{2}\b", re.I)
BOILER = re.compile(r"(?:^|\n|[.!?]\s+)\s*(?:About|ABOUT)\s+(?!\d)(?:the\s+|THE\s+)?[A-Z]|"
                    r"(?i:forward[- ]looking\s+statements|safe\s+harbou?r\b|cautionary\s+(?:note|statement))")

def doc_kind(doc_type, desc, fname, text):
    if not str(doc_type or "").upper().startswith("EX-99"):
        return "form_8k"
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

def lead(text, kind, n):
    """Headline and first paragraphs: the 8-K from its first Item; cut at the boilerplate when it starts after 300 characters."""
    t = text or ""
    if kind == "form_8k":
        m = ITEM_HEAD.search(t)
        t = t[m.start():] if m else t
    t = re.sub(r"^\s*EX(?:HIBIT)?[\s.-]*99(?:\.\d+)?\s*", "", t, flags=re.I)
    x = t[:n]
    m = BOILER.search(x, 300)
    x = x[:m.start()] if m else x
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n", x)).strip()

def filing_lead(docs, n):
    """(kind, text) of one 8-K: press release first, then another Exhibit 99, then the 8-K's items."""
    docs = docs.sort_values(["doc_type", "filename"], na_position="last")
    kinds = [doc_kind(r.doc_type, r.description, r.filename, r.text or "") for r in docs.itertuples()]
    for want in ("press_release", "other_ex99", "form_8k"):
        for k, r in zip(kinds, docs.itertuples()):
            if k == want and (r.text or "").strip():
                return want, lead(r.text, want, n)
    return None, ""

# ---- names to mask ----
SUFFIX_WORDS = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "llc", "lp", "llp", "plc", "sa",
                "ag", "nv", "se", "ab", "as", "asa", "bv", "gmbh", "holdings", "holding", "group", "the", "and", "&"}
GENERIC_CO = {"therapeutics", "pharmaceuticals", "pharmaceutical", "pharma", "bio", "biosciences", "bioscience", "biotherapeutics",
              "biopharma", "biopharmaceuticals", "biotech", "biotechnology", "medical", "medicines", "health", "healthcare",
              "sciences", "science", "international", "global", "technologies", "technology", "labs", "laboratories",
              "oncology", "genetics", "genomics", "therapeutic", "life", "clinical", "research", "american", "national", "new",
              "first", "united", "general", "advanced", "applied", "molecular", "cell", "gene", "immune", "vaccines", "systems"}
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
SPLIT = r"[;()\[\]/,]+|\s+(?:in\s+combination\s+with|combined\s+with|plus|with|and)\s+|\s*\+\s*"
DOSE_TAIL = re.compile(r"(?:\s+\d+(?:\.\d+)?\s*(?:mg|mcg|µg|ug|g|ml|%|iu|units?|mg/kg|mg/m2)(?![a-z]).*"
                       r"|\s+(?:dose|cohort|arm|part|level|group|regimen|schedule)\s*[a-z0-9]{0,3}"
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
        if len(s) >= 4 and re.match(r"[a-z]", s) and s not in STOP_NAMES and not set(re.split(r"[\s-]+", s)) <= GENERIC_WORDS:
            out.add(s)
    return out

firm_drugs = {}
tf_path, cur_path = os.path.join(CTG_DIR, "trial_firm.parquet"), sorted(glob.glob(os.path.join(CTG_DIR, "industry_trials_current*.parquet")))
if os.path.exists(tf_path) and cur_path:
    tf = pd.read_parquet(tf_path)
    cur = pd.read_parquet(cur_path[-1])
    if "interventions" in cur:
        names = cur.set_index("nct_id")["interventions"].to_dict()
        oth = os.path.join(CTG_DIR, "trial_other_names.parquet")
        if os.path.exists(oth):                                   # Notebook 3b's other names (generic name of a code, etc.)
            on = pd.read_parquet(oth)
            for c in [c for c in on.columns if c != "nct_id" and on[c].dtype == object]:
                for n, v in zip(on["nct_id"], on[c]):
                    names[n] = f"{names.get(n, '')};{v if isinstance(v, str) else ';'.join(map(str, v)) if v is not None else ''}"
        for c, n in zip(tf.loc[tf["role"] == "lead", "cik"], tf.loc[tf["role"] == "lead", "nct_id"]):
            firm_drugs.setdefault(int(c), set()).update(name_variants(names.get(n)))
print(f"Firms with drug names to mask: {len(firm_drugs):,}" + ("" if firm_drugs else " (Notebook 2 trial files not found: only "
                                                                                       "code names are masked)"))

CODE = re.compile(r"\b[A-Z]{2,5}-?\d{3,6}[A-Za-z]?\b")                 # ABC-123, XL495; not V600E, IL-6, CD20, COVID-19
NCT_RX = re.compile(r"\bNCT\s?0?\d{7,8}\b", re.I)
ACRO_TRIAL = re.compile(r"(?<![\[\w-])[A-Z][A-Z0-9-]{3,}(?![\]\w-])(?=(?:\s+(?:Phase\s+\S+|pivotal|registrational|clinical))*"
                        r"\s+(?:trial|study)\b)")
ACRO_KEEP = {"PHASE", "TRIAL", "STUDY", "COVID", "COVID-19", "FDA", "NASDAQ", "NYSE", "TOPLINE", "PIVOTAL", "GLOBAL", "OPEN"}

def company_terms(name, ticker):
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'.-]*", str(name or ""))
    core = [w for w in words if w.lower().strip(".,") not in SUFFIX_WORDS]
    terms = set()
    if len(core) >= 2:
        terms.add(" ".join(core))
    terms |= {w for w in core if len(w) >= 4 and w.lower() not in GENERIC_CO}
    return terms, (ticker if isinstance(ticker, str) and len(ticker) >= 3 else None)

_drug_rx = {}

def mask(text, name, ticker, drugs, key=None):
    co, tk = company_terms(name, ticker)
    t = NCT_RX.sub("[TRIAL ID]", text)
    if tk:
        t = re.sub(rf"(?<![\w-]){re.escape(tk)}(?![\w-])", "[TICKER]", t)
    if drugs:
        if key not in _drug_rx or key is None:
            _drug_rx[key] = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(d) for d in sorted(drugs, key=len, reverse=True))
                                       + r")(?![\w-])", re.I)
        t = _drug_rx[key].sub("[DRUG]", t)
    t = CODE.sub("[DRUG]", t)
    if co:
        t = re.sub(r"(?<![\w-])(?:" + "|".join(re.escape(c) for c in sorted(co, key=len, reverse=True)) + r")(?:'s)?(?![\w-])",
                   "[COMPANY]", t, flags=re.I)
    t = ACRO_TRIAL.sub(lambda m: m.group() if m.group().upper() in ACRO_KEEP else "[TRIAL]", t)
    return t

# ---- read the texts (cached) ----
TEXT_VERSION = f"t1_{MAX_CHARS}"                                       # texts are rebuilt when MAX_CHARS or this changes
ev["accs"] = [[a] + [x for x in (s if isinstance(s, str) else "").split(";") if x and x != a]
              for a, s in zip(ev["accession"], ev["same_day_accessions"])]
have = pd.read_parquet(TEXT_PATH) if os.path.exists(TEXT_PATH) else pd.DataFrame(columns=["accession", "version"])
have = have[have["version"] == TEXT_VERSION] if "version" in have else have.iloc[0:0]
todo = ev[~ev["accession"].isin(set(have["accession"]))]
if len(todo):
    doc_paths = sorted(glob.glob(os.path.join(SHARD_DIR, "docs_*.parquet")))
    assert doc_paths, f"No 8-K text shards in {SHARD_DIR}: Notebook 1's output is needed."
    want = sorted({a for accs in todo["accs"] for a in accs})
    parts, t0 = [], time.time()
    for k, p in enumerate(doc_paths, 1):
        try:
            t = pq.read_table(p, columns=["accession", "doc_type", "description", "filename", "text"],
                              filters=[("accession", "in", want)])
        except Exception as e:
            print(f"  skipped unreadable shard {os.path.basename(p)}: {type(e).__name__}")
            continue
        if t.num_rows:
            x = t.to_pandas()
            x["text"] = x["text"].fillna("").str.slice(0, 60000)       # enough for the document type and the lead; caps memory
            parts.append(x)
        if k % 50 == 0 or k == len(doc_paths):
            print(f"  shards {k}/{len(doc_paths)} | {time.time() - t0:.0f}s")
    docs = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["accession", "doc_type", "description", "filename", "text"])
    by_acc = {a: g for a, g in docs.groupby("accession", sort=False)}
    rows = []
    for r in todo.itertuples():
        pieces, kinds = [], []
        for a in r.accs:
            if a not in by_acc:
                continue
            k_, x = filing_lead(by_acc[a], MAX_CHARS)
            if x:
                pieces.append(x)
                kinds.append(k_)
            if sum(map(len, pieces)) >= MAX_CHARS:
                break
        raw = "\n\n---\n\n".join(pieces)[:MAX_CHARS]
        name = getattr(r, "company", "")
        rows.append({"accession": r.accession, "version": TEXT_VERSION, "source": kinds[0] if kinds else None,
                     "n_filings": len(pieces), "n_chars": len(raw), "text_raw": raw,
                     "text": mask(raw, name if isinstance(name, str) else "", r.ticker, firm_drugs.get(int(r.cik), set()), key=int(r.cik))
                     if raw else ""})
    have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True) if len(have) else pd.DataFrame(rows)
    old_all = pd.read_parquet(TEXT_PATH) if os.path.exists(TEXT_PATH) else None
    if old_all is not None and "version" in old_all:                      # keep other readouts' texts (e.g. the dry run's)
        have = pd.concat([old_all[~old_all["accession"].isin(set(have["accession"]))], have], ignore_index=True)
    have.to_parquet(TEXT_PATH + ".tmp", index=False)
    os.replace(TEXT_PATH + ".tmp", TEXT_PATH)
texts = have[have["accession"].isin(set(ev["accession"])) & (have["version"] == TEXT_VERSION)].drop_duplicates("accession", keep="last")
ok = texts["n_chars"].fillna(0) >= 200
print(f"Texts: {ok.sum():,} of {len(ev):,} readouts have 200+ characters | source: "
      + ", ".join(f"{k} {v:.0%}" for k, v in texts.loc[ok, "source"].value_counts(normalize=True).items())
      + f" | median length {texts.loc[ok, 'n_chars'].median():,.0f} characters")
m = texts.loc[ok, "text"]
print("Masked per text (median): " + " | ".join(f"{tag} {m.str.count(re.escape(tag)).median():.0f}"
                                               for tag in ["[COMPANY]", "[DRUG]", "[TRIAL]", "[TRIAL ID]", "[TICKER]"]))
ev = ev.merge(texts.loc[ok, ["accession", "source", "n_chars", "text", "text_raw"]], on="accession")
print("\nExample (masked):\n" + ev["text"].iloc[0][:700])


# %% [markdown]
# ## Step 2: Dictionary measures
# Rates per 1,000 words in the masked text. Hand-built for trial readouts, so every count can be traced to a phrase.


# %% DICT
LEX = {
    "hedge": r"\b(?:may|might|could|potential(?:ly)?|trend(?:s|ed|ing)?|numerical(?:ly)?|appear(?:s|ed)?|suggest(?:s|ed|ive)?|"
             r"encouraging|preliminary|exploratory|post[- ]hoc|subgroups?|signals?|nominal(?:ly)?)\b",
    "success": r"\bmet\s+(?:its|the|both|all|each)\s+(?:\w+\s+){0,3}?(?:co-?)?primary\b|\bachieved\s+(?:its\s+|the\s+)?"
               r"(?:primary|statistical(?:ly)?\s+significan)|\bstatistically\s+significant\s+(?:\w+\s+){0,2}?"
               r"(?:improvement|reduction|increase|benefit|difference|effect)|\bhighly\s+significant\b",
    "failure": r"\bdid\s+not\s+(?:meet|achieve|reach|demonstrate|show)\b|\bfailed\s+to\b|\bnot\s+statistically\s+significant\b|"
               r"\bno\s+(?:statistically\s+)?significant\b|\bmissed\b|\bdiscontinu\w+\s+(?:the\s+)?(?:program|development|trial|study)",
    "safety": r"\bserious\s+adverse\b|\bdeaths?\b|\bdied\b|\bdose[- ]limiting\b|\bdiscontinu\w+\s+due\s+to\b|\bclinical\s+hold\b|"
              r"\bhepatotox\w*|\btoxicit\w*|\bgrade\s+[345]\b",
    "pvalue": r"\bp\s*(?:-?value)?\s*[<=≤]\s*0?\.\d+",
    "percent": r"\d+(?:\.\d+)?\s*%",
}
LEX_RX = {k: re.compile(v, re.I) for k, v in LEX.items()}

def dict_feats(t):
    t = t or ""
    words = max(len(re.findall(r"[A-Za-z]+", t)), 1)
    row = {f"d_{k}": len(rx.findall(t)) * 1000 / words for k, rx in LEX_RX.items()}
    ps = [float(x) for x in re.findall(r"\bp\s*(?:-?value)?\s*[<=≤]\s*(0?\.\d+)", t, re.I)]
    row.update(d_log_words=np.log(words), d_has_p=float(bool(ps)), d_min_p=min(ps) if ps else np.nan)
    return row

dic = pd.DataFrame([dict_feats(t) for t in ev["text"]])
dic.insert(0, "accession", ev["accession"].values)
dic.to_parquet(os.path.join(NB5_DIR, f"dictionary{SUFFIX}.parquet"), index=False)
DCOLS = [c for c in dic.columns if c.startswith("d_") and c != "d_min_p"]
print("Dictionary rates (median per 1,000 words, share > 0):")
for c in DCOLS:
    print(f"  {c:12s} median {dic[c].median():.2f} | > 0 in {(dic[c] > 0).mean():.0%}")


# %% [markdown]
# ## Step 3: LLM reading (GPU)
# Each masked release goes to the model once (greedy decoding). Answers are saved after every batch to `nb5/llm_<model>_<prompt>_parts/`, so a rerun only reads what's missing. A reply that isn't valid JSON is retried once with a stricter instruction; anything still unreadable is left missing and counted.


# %% LLM
SCALE = ("These are mostly small and mid-size biotech companies whose value rests on a few drugs: routine data updates "
         "usually move the stock less than 5%, while clear trial successes or failures often move it 20% to 70% in a day.")
SYSTEM = ("You are an experienced biotech equity analyst. You read company press releases about clinical trial results. "
          "Company names, drug names and trial names are replaced by [COMPANY], [DRUG] and [TRIAL]. " + SCALE +
          " Reply with one JSON object only.")
SYSTEM_MC = ("You are an experienced biotech equity analyst. You read company press releases about clinical trial results. "
             "Company names, drug names and trial names are replaced by [COMPANY], [DRUG] and [TRIAL]. " + SCALE +
             " Answer with a single letter.")
MC = {"move": ("How much will the stock price change over the next trading day, in either direction?\n"
               "A: less than 5%\nB: 5% to 10%\nC: 10% to 20%\nD: 20% to 40%\nE: more than 40%", list("ABCDE"),
               [2.5, 7.5, 15.0, 30.0, 55.0]),
      "outcome": ("Did the trial meet its primary endpoint?\nA: yes\nB: no\nC: partly (mixed results)\nD: the release does not say",
                  list("ABCD"), None)}

def mc_prompt(text, q):
    return SYSTEM_MC, f"Press release:\n<<<\n{text}\n>>>\n\n{MC[q][0]}\nAnswer with one letter."
QUESTIONS = """Answer about the trial result announced above, using only the release:
- "endpoint": did the trial meet its primary endpoint? one of "met", "not_met", "mixed", "not_stated"
- "stat_sig": does it report a statistically significant result on a main measure? "yes", "no" or "not_stated"
- "numbers": does it give effect sizes or p-values? "yes" or "no"
- "safety": safety problems described: "none", "minor" or "serious"
- "hedging": how much the wording softens, qualifies or spins the result, 0 (plain) to 3 (heavy)
- "uncertainty": how hard it is to tell from this release whether the result is good or bad for the drug, 1 (clear) to 10 (very unclear)
- "impact": likely effect on the company's value, -3 (very negative) to 3 (very positive)
- "move_pct": your best guess of how much the stock price will change over the next trading day, in percent, ignoring direction: a number from 0 to 100
Reply with one JSON object with exactly these keys: endpoint, stat_sig, numbers, safety, hedging, uncertainty, impact, move_pct."""
STRICT = "\nReply with the JSON object only, no other text."
CAT = {"endpoint": {"met", "not_met", "mixed", "not_stated"}, "stat_sig": {"yes", "no", "not_stated"}, "numbers": {"yes", "no"},
       "safety": {"none", "minor", "serious"}}
NUM = {"hedging": (0, 3), "uncertainty": (1, 10), "impact": (-3, 3), "move_pct": (0, 100)}
SYN = {"endpoint": {"yes": "met", "true": "met", "achieved": "met", "positive": "met", "success": "met", "successful": "met",
                    "no": "not_met", "false": "not_met", "failed": "not_met", "missed": "not_met", "negative": "not_met",
                    "partially_met": "mixed", "partial": "mixed", "partially": "mixed", "met_partially": "mixed",
                    "not_reported": "not_stated", "unclear": "not_stated", "unknown": "not_stated", "n/a": "not_stated",
                    "na": "not_stated", "none": "not_stated", "not_applicable": "not_stated", "not_mentioned": "not_stated"},
       "stat_sig": {"true": "yes", "false": "no", "not_reported": "not_stated", "unclear": "not_stated", "unknown": "not_stated",
                    "n/a": "not_stated", "na": "not_stated", "not_mentioned": "not_stated", "none": "not_stated",
                    "partial": "yes", "partially": "yes", "mixed": "yes"},
       "numbers": {"true": "yes", "false": "no", "partial": "yes", "partially": "yes", "some": "yes", "none": "no"},
       "safety": {"no": "none", "not_stated": "none", "not_reported": "none", "none_reported": "none", "not_mentioned": "none",
                  "no_new_signals": "none", "mild": "minor", "moderate": "minor", "low": "minor", "manageable": "minor",
                  "yes": "minor", "severe": "serious", "major": "serious", "significant": "serious", "high": "serious"}}

def as_number(v):
    """7, "7", "+2", "25%", "10-20" (midpoint), "about 15"; None if no number."""
    if isinstance(v, bool) or v is None:
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", str(v))]
    if not nums:
        return np.nan
    if len(nums) >= 2 and re.search(r"\d\s*(?:-|to|–)\s*\d", str(v)) and nums[0] >= 0:     # a range: "40-50%"
        return (nums[0] + abs(nums[1])) / 2
    return nums[0]
KEYS = list(CAT) + list(NUM)

def prompt(text, strict=False):
    return SYSTEM, f"Press release:\n<<<\n{text}\n>>>\n\n{QUESTIONS}" + (STRICT if strict else "")

def parse(reply):
    d = {}
    for m in re.finditer(r"\{[^{}]*\}", reply or "", re.S):
        try:
            d = json.loads(m.group())
            break
        except Exception:
            continue
    if not d:
        for k in KEYS:
            mm = re.search(rf"[\"']?{k}[\"']?\s*:\s*[\"']?([-+\w.]+)", reply or "")
            if mm:
                d[k] = mm.group(1)
    d = {str(k).strip().lower(): v for k, v in d.items()}
    row = {}
    for k, ok in CAT.items():
        v = re.sub(r"[\s-]+", "_", str(d.get(k, "")).strip().strip(".\"'").lower())
        v = SYN.get(k, {}).get(v, v)
        row[k] = v if v in ok else None
    for k, (lo, hi) in NUM.items():
        v = as_number(d.get(k))
        row[k] = float(np.clip(v, lo, hi)) if pd.notna(v) else np.nan
    row["parsed"] = all(row[k] is not None for k in CAT) and all(pd.notna(row[k]) for k in NUM)
    return row

def run_llm(df, col, tag):
    """Reads df[col] for every accession not yet answered; saves a part file per batch; returns all answers for df."""
    parts_dir = os.path.join(NB5_DIR, f"llm_{slug(MODEL_USED)}_{PROMPT_VERSION}_{LLM_MODE}{tag}{SUFFIX}_parts")
    os.makedirs(parts_dir, exist_ok=True)
    old = [pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(parts_dir, "*.parquet")))]
    done = pd.concat(old, ignore_index=True).drop_duplicates("accession", keep="last") if old else pd.DataFrame(columns=["accession"])
    todo = df[~df["accession"].isin(set(done["accession"]))]
    todo = todo.assign(_len=todo[col].str.len()).sort_values("_len")        # similar lengths per batch: less padding
    print(f"LLM ({tag.strip('_') or 'masked'}): {len(done):,} already answered, {len(todo):,} to read")
    t0, n, b, k = time.time(), 0, BATCH, len(old)
    rows = todo[["accession", col]].values.tolist()
    i = 0
    while i < len(rows):
        batch = rows[i:i + b]
        try:
            replies = GENERATE([prompt(t) for _, t in batch]) if LLM_MODE == "full" else [""] * len(batch)
            mc = {q: SCORE([mc_prompt(t, q) for _, t in batch], MC[q][1]) for q in MC}
        except Exception as e:
            if "out of memory" in str(e).lower() and b > 1:
                b = max(1, b // 2)
                print(f"  out of GPU memory: batch size now {b}")
                continue
            raise
        out = []
        for j, ((acc, t), rep) in enumerate(zip(batch, replies)):
            r = parse(rep) if LLM_MODE == "full" else {**{k: None for k in CAT}, **{k: np.nan for k in NUM}, "parsed": False}
            pm, mm = mc["move"][0][j], mc["move"][1][j]
            po, mo = mc["outcome"][0][j], mc["outcome"][1][j]
            mids = np.array(MC["move"][2])
            scored = {f"p_move_{L}": float(v) for L, v in zip(MC["move"][1], pm)}
            scored.update(exp_move=float(pm @ mids), move_spread=float(np.sqrt(pm @ (mids - pm @ mids) ** 2)),
                          p_met=float(po[0]), p_not_met=float(po[1]), p_mixed=float(po[2]), p_not_said=float(po[3]),
                          letter_mass=float(min(mm, mo)))
            r["retried"] = False
            if LLM_MODE == "full" and not r["parsed"]:
                rep2 = GENERATE([prompt(t, strict=True)])[0]
                r2 = parse(rep2)
                if r2["parsed"] or sum(v is not None and v == v for v in r2.values()) > sum(v is not None and v == v for v in r.values()):
                    r, rep = r2, rep2
                r["retried"] = True
            out.append({"accession": acc, **r, **scored, "reply": rep[:2000], "model": MODEL_USED, "prompt": PROMPT_VERSION})
        k += 1
        p = os.path.join(parts_dir, f"part_{k:05d}.parquet")
        pd.DataFrame(out).to_parquet(p + ".tmp", index=False)
        os.replace(p + ".tmp", p)
        i += len(batch)
        n += len(batch)
        if k % 10 == 0 or i >= len(rows):
            rate = n / max(time.time() - t0, 1e-9)
            print(f"  {n:,}/{len(rows):,} | {rate * 60:.0f} releases/min | about {(len(rows) - n) / max(rate, 1e-9) / 60:.0f} min left")
    allp = [pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(parts_dir, "*.parquet")))]
    res = pd.concat(allp, ignore_index=True).drop_duplicates("accession", keep="last") if allp else pd.DataFrame(columns=["accession"])
    return res[res["accession"].isin(set(df["accession"]))]

llm = pd.DataFrame(columns=["accession"])
llm_raw = pd.DataFrame(columns=["accession"])
if RUN_LLM:
    if "GENERATE" not in globals() or GENERATE is None:
        GENERATE, SCORE = load_llm()
    ev_llm = ev[ev["readout_kind"] == "topline_or_endpoint"] if LLM_SCOPE == "topline" else ev
    print(f"Mode: {LLM_MODE} | readouts read: {len(ev_llm):,} ({LLM_SCOPE})")
    llm = run_llm(ev_llm, "text", "")
    llm.to_parquet(os.path.join(NB5_DIR, f"llm_{slug(MODEL_USED)}_{PROMPT_VERSION}_{LLM_MODE}{SUFFIX}.parquet"), index=False)
    if LLM_MODE == "full":
        print(f"Parsed answers: {as_bool(llm['parsed']).mean():.1%} ({(~as_bool(llm['parsed'])).sum():,} left missing) | retried "
              f"{as_bool(llm['retried']).sum():,}")
        print("Usable answers per question: " + " | ".join(f"{k} {llm[k].notna().mean():.0%}" for k in KEYS))
        bad = llm[~as_bool(llm["parsed"])]
        for rep_ in bad["reply"].head(3):
            print("  unreadable answer: " + re.sub(r"\s+", " ", str(rep_))[:300])
        print("endpoint: " + ", ".join(f"{k} {v:.0%}" for k, v in llm["endpoint"].value_counts(normalize=True).items()))
    for c in ("move_pct", "exp_move"):
        q = llm[c].dropna()
        if len(q):
            print(f"{c}: median {q.median():.1f}% | 10th-90th percentile {q.quantile(0.1):.1f}%-{q.quantile(0.9):.1f}% | "
                  f"distinct values {q.round(1).nunique()}")
    print(f"Probability the model put on the answer letters (median): {llm['letter_mass'].median():.0%} "
          "(low values would mean it wanted to answer something else)")
    test_rows = ev_llm[ev_llm["year"] > TRAIN_END]
    n_chk = min(UNMASKED_CHECK_N, 10) if DRY_RUN else UNMASKED_CHECK_N
    if n_chk and len(test_rows):
        pre = test_rows[test_rows["event_date"] <= CUTOFF]
        post = test_rows[test_rows["event_date"] > CUTOFF]
        h = max(n_chk // 2, 1)
        chk = pd.concat([pre.sample(min(h, len(pre)), random_state=2), post.sample(min(h, len(post)), random_state=2)])
        llm_raw = run_llm(chk, "text_raw", "_unmasked")
else:
    print("LLM step skipped (RUN_LLM = False)")


# %% [markdown]
# ## Step 4: Embeddings
# Masked texts embedded with `bge-base` (first 512 tokens). A ridge regression on them predicts |move|: out-of-fold within the training years (5 folds by firm), and fit on all training years to predict the test years.


# %% EMBED
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

E_PATH = os.path.join(NB5_DIR, f"embeddings_{slug(EMBED_MODEL)}{SUFFIX}.npz")
emb_pred = pd.Series(np.nan, index=ev.index)
if RUN_EMBED:
    E, accs = None, None
    if os.path.exists(E_PATH):
        z = np.load(E_PATH, allow_pickle=True)
        if set(ev["accession"]) <= set(z["accession"]):
            pos = {a: i for i, a in enumerate(z["accession"])}
            E = z["E"][[pos[a] for a in ev["accession"]]]
    if E is None:
        if "EMBED" not in globals() or EMBED is None:
            EMBED = load_embedder()
        t0 = time.time()
        E = EMBED(ev["text"].tolist())
        np.savez(E_PATH, E=E, accession=ev["accession"].to_numpy(dtype=object))
        print(f"Embedded {len(E):,} texts in {time.time() - t0:.0f}s")
    lo, hi = winsor_fit(ev.loc[ev["year"] <= TRAIN_END, "abs_car"])
    y = ev["abs_car"].clip(lo, hi).to_numpy()
    tr = (ev["year"] <= TRAIN_END).to_numpy()
    alphas = np.logspace(-1, 4, 16)
    if tr.sum() >= 30 and (~tr).sum() and ev.loc[tr, "cik"].nunique() >= 2:
        groups = ev.loc[tr, "cik"].to_numpy()
        oof = np.full(tr.sum(), np.nan)
        folds = GroupKFold(n_splits=min(5, len(set(groups))))
        for a, b in folds.split(E[tr], y[tr], groups):
            oof[b] = RidgeCV(alphas=alphas).fit(E[tr][a], y[tr][a]).predict(E[tr][b])
        full = RidgeCV(alphas=alphas).fit(E[tr], y[tr])
        emb_pred[tr] = oof
        emb_pred[~tr] = full.predict(E[~tr])
        print(f"Embedding ridge: alpha {full.alpha_:.3g} | out-of-fold correlation with |move| in training years "
              f"{np.corrcoef(oof, y[tr])[0, 1]:.2f}")
    else:
        print("Too few training-year readouts for the embedding ridge (dry run?)")
ev["emb_pred"] = emb_pred.values


# %% [markdown]
# ## Step 5: Tests
# Check this cell's output. Fit on readouts up to `TRAIN_END`, tested on later years. Out-of-sample R² = 1 - (squared errors / squared deviations from the training-years mean); it can be negative. The test years are split at the model's training cutoff. The outcome is |abnormal return| on days -1..+1, clipped at the training years' 1st/99th percentiles. Every model is the same linear model (features standardized on the training years, a small ridge penalty chosen on the training years), so the rows differ only in the features.


# %% TESTS
d = ev.merge(dic, on="accession", how="left")
LP = ["exp_move", "move_spread", "p_met", "p_not_met", "p_mixed", "p_not_said"]
if len(llm):
    d = d.merge(llm[["accession"] + KEYS + ["parsed"] + [c for c in LP if c in llm]], on="accession", how="left")
for k in KEYS + LP:
    if k not in d:
        d[k] = np.nan
d["abs_impact"] = d["impact"].abs()
d["log_move"] = np.log1p(d["move_pct"])
d["log_exp_move"] = np.log1p(d["exp_move"]) if "exp_move" in d else np.nan
d["log_pre_vol"] = np.log(d["pre_vol"].where(d["pre_vol"] > 0))
d["phase_cat"] = d["phase"].map(lambda p: f"phase {int(p)}" if pd.notna(p) else "unknown") if "phase" in d else "unknown"
d["earnings_day"] = d["earnings_day"].astype(float)
tr = d["year"] <= TRAIN_END
te = ~tr
post = d["event_date"] > CUTOFF
lo, hi = winsor_fit(d.loc[tr, "abs_car"])
d["y"] = d["abs_car"].clip(lo, hi)
vr = np.log(d["post_vol"] / d["pre_vol"])
vlo, vhi = winsor_fit(vr[tr])
d["y_vol"] = vr.clip(vlo, vhi)

BASE_C, BASE_N = ["readout_kind", "timing", "phase_cat"], ["log_pre_vol", "log_dollar_vol", "earnings_day"]
LAB_C = ["readout_tone"]
LLM_C = ["endpoint", "stat_sig", "numbers", "safety"]
LLM_N = ["hedging", "uncertainty", "impact", "abs_impact", "log_move", "log_exp_move", "move_spread", "p_not_met", "p_mixed"]
HAVE_LLM, HAVE_EMB = bool(d["exp_move"].notna().any()), bool(d["emb_pred"].notna().any())
SPECS = {"controls": (BASE_C, BASE_N), "+ labels": (BASE_C + LAB_C, BASE_N), "+ dictionary": (BASE_C + LAB_C, BASE_N + DCOLS)}
if HAVE_LLM:
    SPECS["+ LLM"] = (BASE_C + LAB_C + LLM_C, BASE_N + LLM_N)
if HAVE_EMB:
    SPECS["+ embeddings"] = (BASE_C + LAB_C, BASE_N + ["emb_pred"])
SPECS["+ all"] = (BASE_C + LAB_C + (LLM_C if HAVE_LLM else []),
                  BASE_N + DCOLS + (LLM_N if HAVE_LLM else []) + (["emb_pred"] if HAVE_EMB else []))

def design(df, cats, nums, fit_rows):
    """Dummies (categories seen in the fit rows, first dropped) and numbers (fit-row median filled, plus a missing flag)."""
    X = pd.DataFrame(index=df.index)
    for c in cats:
        s = df[c].astype(object).where(df[c].notna(), "missing").astype(str)
        levels = sorted(s[fit_rows].unique())
        for lv in levels[1:]:
            X[f"{c}={lv}"] = (s == lv).astype(float)
    for c in nums:
        v = pd.to_numeric(df[c], errors="coerce").astype(float)
        med = v[fit_rows].median()
        X[c] = v.fillna(med if pd.notna(med) else 0.0)
        if v[fit_rows].isna().any() or v.isna().any():
            X[c + "_missing"] = v.isna().astype(float)
    X = X.loc[:, X[fit_rows].std() > 0]
    X.insert(0, "const", 1.0)
    return X

def oos(ycol, rows_tr, rows_te, subsets):
    out = {}
    for name, (cats, nums) in SPECS.items():
        X = design(d, cats, nums, rows_tr).drop(columns="const")
        ok_tr = rows_tr & d[ycol].notna()
        mu, sd = X[ok_tr].mean(), X[ok_tr].std().replace(0, 1)
        Z = ((X - mu) / sd).to_numpy()
        fit = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Z[ok_tr.to_numpy()], d.loc[ok_tr, ycol].to_numpy())
        pred = fit.predict(Z)
        ybar = d.loc[ok_tr, ycol].mean()
        res = {}
        for sname, srows in subsets.items():
            m = srows & d[ycol].notna()
            if m.sum() < 10:
                res[sname] = (np.nan, np.nan, int(m.sum()))
                continue
            yy, pp = d.loc[m, ycol].to_numpy(), pred[m.to_numpy()]
            r2 = 1 - ((yy - pp) ** 2).sum() / ((yy - ybar) ** 2).sum()
            res[sname] = (r2, spearmanr(yy, pp)[0], int(m.sum()))
        out[name] = (res, pred)
    return out

from sklearn.linear_model import RidgeCV
subsets = {"test, all": te, f"test, up to {CUTOFF:%b %Y}": te & ~post, f"test, after {CUTOFF:%b %Y}": te & post,
           "test, topline only": te & (d["readout_kind"] == "topline_or_endpoint")}
print(f"Run: {'DRY RUN' if DRY_RUN else 'FULL'} | model {MODEL_USED or 'none'} | readouts with text {len(d):,} "
      f"(train up to {TRAIN_END}: {tr.sum():,}, test: {te.sum():,}, of which after the model cutoff: {(te & post).sum():,})")
if tr.sum() < 30 or te.sum() < 10:
    print("Too few readouts for the out-of-sample tests (dry run): check the steps above ran, then do the full run.")
    res_y = {}
else:
    res_y = oos("y", tr, te, subsets)
    print(f"\n=== Out-of-sample: |abnormal return| days -1..+1 (R2 | rank correlation | n) ===")
    print(f"  {'model':14s} " + " ".join(f"{s:>28s}" for s in subsets))
    for name, (res, _) in res_y.items():
        print(f"  {name:14s} " + " ".join(f"{r[0]:+8.3f} | {r[1]:+.2f} | {r[2]:5,}".rjust(28) for r in res.values()))
    res_v = oos("y_vol", tr, te, {"test, all": te, f"test, after {CUTOFF:%b %Y}": te & post})
    print("\n=== Out-of-sample: volatility after vs before (log ratio, days +2..+21 vs -60..-11), R2 ===")
    print("  " + " | ".join(f"{name} {res['test, all'][0]:+.3f}" for name, (res, _) in res_v.items()))

    # ---- drift: does the text predict the move AFTER the readout (days +2..+20), beyond the immediate reaction? ----
    if "car_2_20" in d and d["car_2_20"].notna().sum() > 50:
        d["y_drift"] = d["car_2_20"].clip(*winsor_fit(d.loc[tr, "car_2_20"]))
        d["car_1_1_c"] = d["car_1_1"].clip(*winsor_fit(d.loc[tr, "car_1_1"]))
        SPECS_KEEP = dict(SPECS)
        for k_, (c_, n_) in SPECS_KEEP.items():
            SPECS[k_] = (c_, n_ + ["car_1_1_c"])
        res_d = oos("y_drift", tr, te, {"test, all": te, f"test, after {CUTOFF:%b %Y}": te & post,
                                         "test, topline only": te & (d["readout_kind"] == "topline_or_endpoint")})
        SPECS.clear()
        SPECS.update(SPECS_KEEP)
        print("\n=== Out-of-sample: drift after the readout, days +2..+20 (signed; every model also knows the days -1..+1 reaction) ===")
        print(f"  {'model':14s} " + " ".join(f"{s_:>28s}" for s_ in res_d[next(iter(res_d))][0]))
        for name, (res, _) in res_d.items():
            print(f"  {name:14s} " + " ".join(f"{r[0]:+8.3f} | {r[1]:+.2f} | {r[2]:5,}".rjust(28) for r in res.values()))
        print("  (a positive R2 for '+ LLM' or '+ all' above '+ labels' means the text predicts drift the price has not yet absorbed)")
        if HAVE_LLM:
            zt = d[(d["readout_kind"] == "topline_or_endpoint")].dropna(
                subset=["y_drift", "car_1_1_c", "log_pre_vol", "log_dollar_vol", "p_not_met", "p_mixed", "log_exp_move", "move_spread"])
            if len(zt) > 100 and zt["cik"].nunique() >= 10:
                g = cluster_groups(zt["cik"], zt["day0"])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    md_ = smf.ols("y_drift ~ car_1_1_c + C(readout_tone) + C(timing) + log_pre_vol + log_dollar_vol + C(year)"
                                  " + p_not_met + p_mixed + log_exp_move + move_spread", zt).fit(
                        **({"cov_type": "cluster", "cov_kwds": {"groups": g}} if g is not None else {}))
                kk = ["car_1_1_c", "p_not_met", "p_mixed", "log_exp_move", "move_spread"]
                print(f"\n=== Topline readouts, all years: drift days +2..+20 on the LLM's read, given the immediate reaction (clustered) ===")
                print(pd.DataFrame({"coef": md_.params[kk], "se": md_.bse[kk], "p": md_.pvalues[kk]}).round(4).to_string())
                print(f"  n = {int(md_.nobs):,}; tone, timing, volatility, size and year controls not shown")
                zt = zt.assign(react=np.where(zt["car_1_1"] < 0, "stock fell (days -1..+1)", "stock rose (days -1..+1)"),
                               read=pd.cut(zt["p_met"], [-0.01, 0.33, 0.67, 1.0], labels=["P(met) < 1/3", "1/3 to 2/3", "> 2/3"]))
                print("\nMean drift days +2..+20 by the LLM's P(met) and the day's reaction (topline readouts):")
                print(zt.pivot_table(index="read", columns="react", values="y_drift", aggfunc=["mean", "size"], observed=True)
                      .round(3).to_string())
                print("  (text says 'met' but the stock fell, then drifts up = the market overreacted; drifts down = the text missed something)")

    if "+ LLM" in res_y:
        rng = np.random.default_rng(7)
        firms = d.loc[te, "cik"].unique()
        yv = d["y"].to_numpy()
        ybar = d.loc[tr, "y"].mean()
        pl, pb = res_y["+ LLM"][1], res_y["+ labels"][1]
        idx_by_firm = {f: np.flatnonzero((d["cik"] == f).to_numpy() & te.to_numpy() & d["y"].notna().to_numpy()) for f in firms}
        diffs = []
        for _ in range(500):
            ii = np.concatenate([idx_by_firm[f] for f in rng.choice(firms, len(firms))])
            sst = ((yv[ii] - ybar) ** 2).sum()
            diffs.append(((yv[ii] - pb[ii]) ** 2).sum() / sst - ((yv[ii] - pl[ii]) ** 2).sum() / sst)
        lo_, hi_ = np.percentile(diffs, [2.5, 97.5])
        print(f"\nLLM over labels, test years: R2 gain {res_y['+ LLM'][0]['test, all'][0] - res_y['+ labels'][0]['test, all'][0]:+.3f} "
              f"(95% interval by resampling firms {lo_:+.3f} to {hi_:+.3f})")

def sp(a, b):
    m = pd.notna(a) & pd.notna(b)
    return f"{spearmanr(a[m], b[m])[0]:+.2f}" if m.sum() >= 10 and a[m].nunique() > 1 else "  n/a"

if len(llm):
    x = d[d[KEYS].notna().any(axis=1)]
    print("\n=== LLM answers vs outcomes (all years; rank correlations) ===")
    for lab, rows in [("all", x), ("topline only", x[x["readout_kind"] == "topline_or_endpoint"]),
                      ("Notebook 3 tone = mixed", x[x["readout_tone"] == "mixed"]), ("after the model cutoff", x[x["event_date"] > CUTOFF])]:
        if len(rows) >= 10:
            print(f"  {lab:24s} n = {len(rows):5,} | expected move vs |return| {sp(rows['exp_move'], rows['abs_car'])} | "
                  f"move_pct {sp(rows['move_pct'], rows['abs_car'])} | spread {sp(rows['move_spread'], rows['abs_car'])} | "
                  f"uncertainty {sp(rows['uncertainty'], rows['abs_car'])} | impact vs return {sp(rows['impact'], rows['car_1_1'])} | "
                  f"P(met) vs return {sp(rows['p_met'], rows['car_1_1'])}")
    t = x[x["readout_kind"] == "topline_or_endpoint"]
    if len(t) >= 10 and t["p_met"].notna().sum() >= 10:
        print("\nTopline readouts by the LLM's P(met): mean return days -1..+1")
        print(t.groupby(pd.cut(t["p_met"], [-0.01, 0.33, 0.67, 1.0], labels=["P(met) < 1/3", "1/3 to 2/3", "> 2/3"]),
                        observed=True)["car_1_1"].agg(["size", "mean", "median"]).round(3).to_string())
    if len(t) and t["endpoint"].notna().any():
        print("\nLLM 'endpoint' vs Notebook 3 tone (topline readouts, share of each tone):")
        print(pd.crosstab(t["readout_tone"], t["endpoint"], normalize="index").round(2).to_string())
        print("\nMean return (days -1..+1) by LLM endpoint answer, topline readouts:")
        print(t.groupby("endpoint")["car_1_1"].agg(["size", "mean", "median"]).round(3).to_string())

    full = d["endpoint"].notna().any()
    llm_terms = (["C(endpoint)", "C(safety)", "hedging", "uncertainty", "abs_impact"] if full else []) + \
                ["log_exp_move", "move_spread", "p_not_met", "p_mixed"]
    f = "y ~ C(readout_kind) + C(readout_tone) + C(timing) + log_pre_vol + log_dollar_vol + earnings_day + C(year) + " + " + ".join(llm_terms)
    z = d.dropna(subset=["y", "log_pre_vol", "log_dollar_vol", "log_exp_move", "move_spread", "p_not_met", "p_mixed"]
                 + (["endpoint", "safety", "hedging", "uncertainty", "abs_impact"] if full else []))
    if len(z) > 100 and z["cik"].nunique() >= 10:
        g = cluster_groups(z["cik"], z["day0"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = smf.ols(f, z).fit(**({"cov_type": "cluster", "cov_kwds": {"groups": g}} if g is not None else {}))
        tab = pd.DataFrame({"coef": m.params, "se": m.bse, "p": m.pvalues})
        keep = [i for i in tab.index if i.startswith(("C(endpoint)", "C(safety)")) or i in
                ("hedging", "uncertainty", "abs_impact", "log_exp_move", "move_spread", "p_not_met", "p_mixed")]
        print(f"\n=== All years, in sample: |return| on LLM answers with Notebook 4's controls (clustered by firm and date) ===")
        print(tab.loc[keep].round(4).to_string())
        print(f"R2 = {m.rsquared:.3f}, n = {int(m.nobs):,} (controls and year effects not shown)")

if len(llm_raw):
    print("\n=== Does the model recognize the readouts? Same readouts, masked vs unmasked (rank correlations) ===")
if len(llm_raw) and llm["impact"].notna().any():
    k = d[["accession", "car_1_1", "abs_car", "event_date"]].merge(llm[["accession", "impact", "move_pct"]], on="accession") \
        .merge(llm_raw[["accession", "impact", "move_pct"]], on="accession", suffixes=("", "_raw"))
    for lab, rows in [(f"up to {CUTOFF:%b %Y}", k[k["event_date"] <= CUTOFF]), (f"after {CUTOFF:%b %Y}", k[k["event_date"] > CUTOFF])]:
        if len(rows) >= 10:
            print(f"  {lab:16s} n = {len(rows):4,} | impact vs return: masked {sp(rows['impact'], rows['car_1_1'])}, "
                  f"unmasked {sp(rows['impact_raw'], rows['car_1_1'])} | move_pct vs |return|: masked "
                  f"{sp(rows['move_pct'], rows['abs_car'])}, unmasked {sp(rows['move_pct_raw'], rows['abs_car'])}")
if len(llm_raw):
    print("  (unmasked much better than masked before the cutoff, and not after, would mean the model knows these cases)")
    k2 = d[["accession", "car_1_1", "abs_car", "event_date"]].merge(llm[["accession", "exp_move", "p_met"]], on="accession") \
        .merge(llm_raw[["accession", "exp_move", "p_met"]], on="accession", suffixes=("", "_raw"))
    for lab, rows in [(f"up to {CUTOFF:%b %Y}", k2[k2["event_date"] <= CUTOFF]), (f"after {CUTOFF:%b %Y}", k2[k2["event_date"] > CUTOFF])]:
        if len(rows) >= 10:
            print(f"  {lab:16s} P(met) vs return: masked {sp(rows['p_met'], rows['car_1_1'])}, unmasked {sp(rows['p_met_raw'], rows['car_1_1'])}"
                  f" | expected move vs |return|: masked {sp(rows['exp_move'], rows['abs_car'])}, unmasked "
                  f"{sp(rows['exp_move_raw'], rows['abs_car'])}")

keep_cols = ["accession", "cik", "ticker", "event_date", "year", "readout_kind", "readout_tone", "timing", "car_1_1", "abs_car", "y",
             "y_vol", "y_drift", "car_2_20", "source", "n_chars"] + DCOLS + ["d_min_p"] + [c for c in KEYS + ["parsed"] + LP if c in d] + ["emb_pred"]
for name, (res, pred) in res_y.items():
    d[f"pred_{slug(name)}"] = pred
    keep_cols.append(f"pred_{slug(name)}")
d[[c for c in keep_cols if c in d]].to_parquet(os.path.join(NB5_DIR, f"readout_text_features{SUFFIX}.parquet"), index=False)

# ---- figures ----
import matplotlib.pyplot as plt
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "legend.frameon": False, "axes.axisbelow": True})
if res_y:
    names = list(res_y)
    show = ["test, all", f"test, up to {CUTOFF:%b %Y}", f"test, after {CUTOFF:%b %Y}"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    w = 0.8 / len(show)
    for j, (s, col) in enumerate(zip(show, COL)):
        vals = [res_y[n][0][s][0] for n in names]
        ax.bar(np.arange(len(names)) + (j - (len(show) - 1) / 2) * w, vals, width=w * 0.92, color=col, label=s)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Out-of-sample R²")
    ax.set_title(f"Predicting the size of the move from the release (fit up to {TRAIN_END})", loc="left")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(FIG_DIR, f"fig9_oos_r2{SUFFIX}.png"), dpi=200, bbox_inches="tight")
    plt.show()
if len(llm) and d["exp_move"].notna().sum() >= 20:
    BINS, LABS = [-0.1, 5, 10, 20, 40, 100], ["0-5%", "5-10%", "10-20%", "20-40%", "40%+"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for j, (lab, rows) in enumerate([("up to the model cutoff", d[d["event_date"] <= CUTOFF]), ("after the model cutoff", d[d["event_date"] > CUTOFF])]):
        r2 = rows.dropna(subset=["exp_move"])
        g = r2.groupby(pd.cut(r2["exp_move"], BINS, labels=LABS), observed=True)["abs_car"].agg(["mean", "size"])
        g = g[g["size"] >= 5]
        if len(g):
            ax.plot([LABS.index(i) for i in g.index], g["mean"] * 100, color=COL[j], linewidth=2, marker="o", markersize=8,
                    label=f"{lab} (n = {int(g['size'].sum()):,})")
    ax.set_xticks(range(len(LABS)))
    ax.set_xticklabels(LABS)
    ax.set_xlabel("LLM: expected size of the move (probability-weighted)")
    ax.set_ylabel("Mean |return|, days -1..+1 (%)")
    ax.set_title("Readouts the model expects to move the stock more do move it more?", loc="left")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(FIG_DIR, f"fig10_llm_move{SUFFIX}.png"), dpi=200, bbox_inches="tight")
    plt.show()


# %% [markdown]
# ## What next
# - Check Step 5's output.
# - If the LLM adds out-of-sample R² after the cutoff too, the text measure holds up without memorization. Next would be a second model (e.g. Qwen) as a robustness check, and the text measure against options volume for 2024-2025 readouts.
# - `readout_text_features.parquet` has every feature and prediction per readout for the paper's tables.

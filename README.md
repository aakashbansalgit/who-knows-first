# Who Knows First? Replication code

Code for **"Who Knows First? Rational Learning, Price Drift and Insider Trading Around Clinical Trial Results"** by Aakash Bansal (working paper, 2026).

The paper asks what price drift before clinical trial readouts can reveal about insider trading. It measures the drift in 1,359 topline readouts announced on Form 8-K by US-listed biotech firms from 2010 to 2025. It then tests the channels the disclosure and insider-trading rules monitor: Form 4 trades, options volume, the timing of filings, and the ClinicalTrials.gov registry, using a separate sample of 1,083 trial stops. The pipeline here builds every dataset from public sources and produces the paper's tables, figures and reported numbers.

## Repository layout

```
pipeline/    the 11 stages, run in order (Python scripts with '# %%' cell markers)
notebooks/   the same stages as Colab notebooks, generated from pipeline/ by tools/make_notebooks.py
tests/       offline tests of every stage on synthetic data with planted effects
tools/       make_notebooks.py, check_paper_numbers.py, cells.py (shared cell reader), stages.json
archive/     superseded and exploratory code, not run (see archive/README.md)
```

`pipeline/` is the source of truth. Each stage opens as a notebook in Jupyter or VS Code, runs top to bottom as a script, and becomes a Colab notebook through `python tools/make_notebooks.py`. Edit the stage, never the generated notebook.

## Stages

| Stage | What it does | Main outputs (under `features/`) |
| --- | --- | --- |
| `01_edgar_8k` | Biopharma firm universe from SIC codes; downloads and parses 8-K press releases from EDGAR | 8-K texts and filing index |
| `02_registry_snapshots` | Industry trials on ClinicalTrials.gov, sponsors matched to firms, and each trial's month-by-month history from AACT snapshots | trial-firm links, registry histories |
| `03_features` | Filing times, readout classification and tone, registry changes (stops, delays, enrollment cuts, outcome edits) | `readout_events`, `registry_events` |
| `03b_stop_disclosure` | Searches every SEC filing of each sponsor for the first disclosure of each trial stop, with a hand-coding sample | `stop_disclosure` |
| `04_prices_events` | Daily prices, abnormal returns against XBI, event windows, OCC options volume, first tests | event-level returns |
| `05_release_text` | Masked release texts, dictionary tone, a language model's reading (GPU), embeddings, out-of-sample tests | `nb5/readout_text_features` |
| `06_failure_risk` | Market-implied probability of success; registry warning flags before readouts | `nb6/` |
| `07_disclosure_channels` | Long-window drift after stops, channel choice and timing, hand-coding samples | `nb7/` |
| `08_exits` | How each sponsor without a ticker exited (EDGAR filing histories); outcomes after each stop | `nb8/` |
| `09_insiders` | Form 4 trades from the SEC Insider Transactions Data Sets; insider tests and the Friday test | `nb9/` |
| `10_robustness` | Every robustness check, the 20-placebo run-up, the window-selection test, Holm corrections and the paper's figures | `nb10/`, `figures/paper/` |

Run them in this order. Each stage reads the outputs of earlier ones from the project folder, and each starts with a **settings cell**. Several stages default to a dry run (`DRY_RUN = True`) on a small sample. Set it to `False` for the full run.

## Data sources

No data are included. Every input is public, and the pipeline downloads it.

| Source | Used for | Access and terms |
| --- | --- | --- |
| SEC EDGAR (filing indexes, 8-Ks, all filings, submissions API) | Readouts, filing times, stop disclosures, exits | Free. The SEC's fair-access policy requires a User-Agent with your name and email (`SEC_USER_AGENT`) and at most 10 requests a second, which the code respects. |
| SEC Insider Transactions Data Sets | Form 3, 4 and 5 trades | Free, quarterly files from sec.gov. |
| AACT (CTTI) monthly snapshots of ClinicalTrials.gov | Registry histories and trial stops | Free account and API key. |
| Yahoo Finance, through `yfinance` | Daily prices for stocks and XBI | Yahoo's terms limit use to personal, non-commercial research. Prices are not redistributed here. |
| OCC volume reports | Daily options volume around readouts | Public. |
| Hugging Face models | Llama 3.1 8B Instruct (Meta community license, accept on Hugging Face), Qwen 2.5 7B Instruct (Apache 2.0, fallback), BAAI/bge-base-en-v1.5 embeddings (MIT) | Free account; Llama needs a read token (`HF_TOKEN`). |

## Running it

**On Google Colab (how the paper was produced).**
1. Open `notebooks/01_edgar_8k.ipynb` in Colab, for example through File > Open notebook > GitHub.
2. Outputs go to `MyDrive/biopharma_tiv` on Google Drive, which the first cell mounts.
3. Put your AACT key in Colab Secrets as `AACT_API` and your Hugging Face token as `HF_TOKEN`.
4. Set `USER_AGENT` in the settings cell of stages 01, 02, 03b, 04, 08 and 09.
5. Use a GPU runtime for stage 05. On a free T4, the language model takes 30 to 60 minutes on the topline readouts.

**Locally or on a server.** You need Python 3.10 or later.

```bash
pip install -r requirements.txt
cp .env.example .env            # fill it in, then: set -a; source .env; set +a
python pipeline/01_edgar_8k.py  # and so on, in the order above
```

`WKF_PROJECT_DIR` sets the project folder, `SEC_USER_AGENT` the User-Agent, and `AACT_API_KEY` and `HF_TOKEN` the keys. Nothing secret is written into the code or the outputs.

## Where each result comes from

| In the paper | Stage (step) |
| --- | --- |
| Table 1, data sources and samples | 01 to 09 (sample counts printed by each stage) |
| Tables 2 and 3, readout returns | 04 (Step 5) |
| Table 4, implied and realized probability of success | 06 (Step 2); 10 (Steps 2 and 19) |
| Tables 5 and 6, registry posting and first disclosure | 07; 10 (Step 4) |
| Table 7, outcomes after stops | 08 (Step 3); 10 (Step 6) |
| Tables 8 and 9, insider trades and the 2023 reform | 09 (Step 3); 10 (Steps 5 and 12) |
| Table 10, Friday filings | 09 (Step 4); 10 (Step 13) |
| Section 9.2, the text of the release | 05 (Step 5) |
| Table A1, size of the readout move | 04 (Step 5); 10 (Step 7) |
| Table A2, robustness | 10 (Steps 1 to 21) |
| Table A3, sample flow | Counts printed by each stage; 10 (Step 21 (a)) for the run-up and insider samples |
| The run-up (Section 5.2): two-week and six-week gaps, robustness, window-selection test, share before the filing | 10 (Steps 19, 19c, 20 and 21) |
| Holm corrections | 10 (Steps 9 and 19c) |
| Figures 1 and 3 (six-week paths; implied probability) | 10 (Step 19) |
| Figures 2, 4, 5, 6 and 8 (readout paths, stop reasons, drift after stops, exits, insider sales) | 10 (Step 17) |
| Figure 7 (exits as competing risks) | 10 (Step 18) |
| Figure 9 (out-of-sample R-squared of the text measures) | 05 (Step 5); the paper redraws it in the common style |

After a full run, `python tools/check_paper_numbers.py` compares the saved run-up results with the numbers in the paper.

## Random seeds and reproducibility

Every random draw that affects a result is seeded, so running a stage top to bottom on the same inputs gives the same output every time.

| Where | Seed | Used for |
| --- | --- | --- |
| `10_robustness`, settings cell | `SEED = 2026` | One shared stream for the bootstraps and for the placebo dates of Steps 3 to 16 |
| `10_robustness`, Step 18 | `SEED + 1` to `SEED + 5`, `SEED + 20` | Placebo dates for the earlier 5-placebo robustness checks and a first 20-placebo check |
| `10_robustness`, Steps 19 and 21 | `SEED + 100` to `SEED + 104` | The 20 placebo dates per readout (the paper's run-up), five draws |
| `10_robustness`, Step 20 | `SEED + 200`, `SEED + 300` | Placebo dates for mixed and unclear readouts; placebo falsification test |
| `10_robustness`, Step 21 | `SEED + 2100` | Label permutations and placebo pseudo-events in the window-selection test |
| `10_robustness`, Step 8 | `random_state = SEED` | Never-disclosed stops drawn for the recall check |
| `06_failure_risk`, `07_disclosure_channels`, `08_exits` | `2026` | Bootstraps and placebo dates |
| `05_release_text` | `random_state = 1, 2`; `7` | Dry-run sample and spot checks; bootstrap of the R-squared |
| `02`, `03b`, `07` | fixed `random_state` values | Hand-coding and audit samples (so the hand-coded files match) |

The language model decodes greedily (`do_sample=False`), so its answers do not depend on a seed. Cross-validation uses `GroupKFold`, which is deterministic. The only unseeded randomness is the jitter in retry waits when a download fails, which affects timing only.

Two things can still move a number in its last digit:
- **Step order in `10_robustness`.** The bootstraps, and the placebo dates of Steps 3 to 16, draw from one shared stream, so they depend on which steps ran before them in the session. The paper's numbers came from sessions that ran the steps in stages, so a single top-to-bottom run can differ in the last reported digit for those steps. The run-up in Section 5.2 uses its own placebo seeds (Steps 19 to 21), so its point estimates do not depend on step order; only its bootstrap intervals and p-values can move.
- **Data vintage.** The paper uses EDGAR filings through 2025, AACT snapshots from January 2017 to January 2026, and prices downloaded in 2026. Yahoo revises adjusted prices and drops many delisted tickers, so a later download can differ slightly.

## Tests

```bash
bash tests/run_all.sh
```

Each stage has an offline test. It builds synthetic inputs with planted effects, mocks EDGAR, AACT, Yahoo, OCC and the language model, and checks that the stage recovers what was planted. No network or GPU is needed. The tests read the cells straight from `pipeline/`, so they test the code that runs.

## License and citation

The code is released under the MIT License (see `LICENSE`). Data obtained by running it remain under their providers' terms.

Bansal, Aakash. 2026. "Who Knows First? Rational Learning, Price Drift and Insider Trading Around Clinical Trial Results." Working paper, SSRN.

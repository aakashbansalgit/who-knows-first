"""Archived: a check of the language model's answer-order bias (reverse the answer options for 200 readouts).

Written but not run for the paper (Section 10.1 notes the possible position bias). Paste after Step 1 of
pipeline/05_release_text.py on a GPU runtime.
"""

# %% ORDER_CELL
# ==== Option-order check: paste at the end of Notebook 5 (GPU runtime). Needs Step 0 and Step 1 (texts) run first. ====
N_ORDER = 200                                     # topline readouts re-asked with the answer options in reverse order
if "SCORE" not in globals() or SCORE is None:
    GENERATE, SCORE = load_llm()
REV = {"move": ("How much will the stock price change over the next trading day, in either direction?\n"
                "A: more than 40%\nB: 20% to 40%\nC: 10% to 20%\nD: 5% to 10%\nE: less than 5%", list("ABCDE")),
       "outcome": ("Did the trial meet its primary endpoint?\nA: the release does not say\nB: partly (mixed results)\nC: no\nD: yes",
                   list("ABCD"))}
MIDS = np.array(MC["move"][2])                    # 2.5, 7.5, 15, 30, 55 in the original order

def ask(texts, qtext, letters):
    return SCORE([(SYSTEM_MC, f"Press release:\n<<<\n{t}\n>>>\n\n{qtext}\nAnswer with one letter.") for t in texts], letters)

samp = ev[ev["readout_kind"] == "topline_or_endpoint"].sample(min(N_ORDER, (ev["readout_kind"] == "topline_or_endpoint").sum()), random_state=7)
texts_, rows, b, i, t0 = samp["text"].tolist(), [], BATCH, 0, time.time()
while i < len(texts_):
    tb = texts_[i:i + b]
    try:
        o_out, o_mov = ask(tb, MC["outcome"][0], MC["outcome"][1]), ask(tb, MC["move"][0], MC["move"][1])
        r_out, r_mov = ask(tb, *REV["outcome"]), ask(tb, *REV["move"])
    except Exception as e:
        if "out of memory" in str(e).lower() and b > 1:
            b = max(1, b // 2)
            continue
        raise
    for j in range(len(tb)):
        po, pr = o_out[0][j], r_out[0][j][::-1]   # reversed answers back to the original order: met, not met, mixed, not said
        mo, mr = o_mov[0][j], r_mov[0][j][::-1]
        rows.append({"p_met": po[0], "p_met_rev": pr[0], "cat": int(np.argmax(po)), "cat_rev": int(np.argmax(pr)),
                     "exp_move": float(mo @ MIDS), "exp_move_rev": float(mr @ MIDS),
                     "p_first_out": o_out[0][j][0], "p_first_out_rev": r_out[0][j][0],
                     "p_first_move": o_mov[0][j][0], "p_first_move_rev": r_mov[0][j][0]})
    i += len(tb)
OC = pd.concat([samp[["accession", "abs_car"]].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
OC.to_parquet(os.path.join(NB5_DIR, f"option_order_{slug(MODEL_USED)}.parquet"), index=False)
print(f"Model {MODEL_USED} | {len(OC)} topline readouts, each asked in both orders ({time.time() - t0:.0f}s)\n")
print("=== P(met), original against reversed options ===")
print(f"correlation {OC[['p_met', 'p_met_rev']].corr().iloc[0, 1]:.3f} | mean {OC['p_met'].mean():.3f} vs {OC['p_met_rev'].mean():.3f} "
      f"| same top answer {(OC['cat'] == OC['cat_rev']).mean():.0%} | |difference| above 0.2: {((OC['p_met'] - OC['p_met_rev']).abs() > 0.2).mean():.0%}")
print("\n=== Expected move, original against reversed options ===")
print(f"correlation {OC[['exp_move', 'exp_move_rev']].corr().iloc[0, 1]:.3f} | mean {OC['exp_move'].mean():.1f}% vs {OC['exp_move_rev'].mean():.1f}%")
ac = OC["abs_car"].clip(*OC["abs_car"].quantile([0.01, 0.99]))
avg = (OC["exp_move"] + OC["exp_move_rev"]) / 2
print(f"correlation with the actual absolute move: original {OC['exp_move'].corr(ac):.3f} | reversed {OC['exp_move_rev'].corr(ac):.3f} | "
      f"average of both orders {avg.corr(ac):.3f}")
print("\n=== Position bias: probability on whichever answer is listed first ===")
print(f"endpoint question: original order (A = yes) {OC['p_first_out'].mean():.3f} | reversed (A = does not say) {OC['p_first_out_rev'].mean():.3f}")
print(f"move question: original (A = under 5%) {OC['p_first_move'].mean():.3f} | reversed (A = over 40%) {OC['p_first_move_rev'].mean():.3f}")
print("\nReading: correlations near 1 and small mean shifts mean the answers follow the content, not the letter position.")

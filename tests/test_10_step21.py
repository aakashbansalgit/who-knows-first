"""Step 21 of 10_robustness on the synthetic data of test_10_robustness: it must redraw Step 19's placebo dates exactly."""
import os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "test_10_robustness.py")).read().split("# checks on planted effects")[0].replace("print(out)", "")
if os.environ.get("NOEFFECT"):
    src = src.replace("ret[c][j - 10:j - 1] += 0.008", "ret[c][j - 10:j - 1] += 0.0")
src = src.replace('E.to_parquet(f"{F}/nb9/readout_insiders.parquet", index=False)',
                  'E.loc[E.index[:7], "log_dollar_vol"] = np.nan\nE.to_parquet(f"{F}/nb9/readout_insiders.parquet", index=False)')
exec(src)
exec(C.ROBUST30, ns)
D, H, D21, H21 = ns["D"], ns["H"], ns["D21"], ns["H21"]
assert len(D21) == len(D) and len(H21) == len(H), (len(D21), len(D), len(H21), len(H))
m = H.merge(H21[["accession", "x30_s0"]], on="accession")
assert np.allclose(m["xe30"] + m["x10"], m["x30_s0"], equal_nan=True), "placebo draws not replicated"
print("ALL TESTS PASSED")

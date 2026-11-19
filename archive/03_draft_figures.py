"""Archived: draft figures from Notebook 3 outputs. Not used in the paper and not run by the pipeline."""

# ==== Figures from Notebook 3 (run after the full run; reads the saved files) ====
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEAT = os.path.join(os.environ.get("WKF_PROJECT_DIR", "/content/drive/MyDrive/biopharma_tiv"), "features")
FIG_DIR = os.path.join(FEAT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10,
                     "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
                     "axes.axisbelow": True, "xtick.major.size": 3, "ytick.major.size": 0})

def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), dpi=200, bbox_inches="tight")
    plt.show()

reg = pd.read_parquet(os.path.join(FEAT, "registry_events.parquet"))
filings = pd.read_parquet(os.path.join(FEAT, "filings.parquet"))
links = pd.read_parquet(os.path.join(FEAT, "filing_trial_links.parquet"))
ro = pd.read_parquet(os.path.join(FEAT, "readouts.parquet"))

stops = reg[reg["is_stop"] & (reg["firm_8ks_year"] > 0)].copy()
stops["size"] = pd.cut(stops["firm_lead_trials"], [0, 20, 100, np.inf], labels=["1-20", "21-100", "100+"])
stops = stops[stops["size"].notna()]

# ---- Figure 1: when does an 8-K name a stopped trial? ----
stops["when"] = np.select([stops["disclosed_before"], stops["days_to_8k"].le(30), stops["days_to_8k"].notna()],
                          ["8-K first (up to 30 days before)", "8-K 0-30 days after", "8-K 31-365 days after"],
                          "No 8-K names the trial")
order = ["8-K first (up to 30 days before)", "8-K 0-30 days after", "8-K 31-365 days after", "No 8-K names the trial"]
share = pd.crosstab(stops["size"], stops["when"], normalize="index").reindex(columns=order, fill_value=0) * 100
n = stops["size"].value_counts().reindex(share.index)
fig, ax = plt.subplots(figsize=(8, 3.2))
left = np.zeros(len(share))
y = np.arange(len(share))[::-1]
for col, c in zip(order, [BLUE, ORANGE, AQUA, GRAY]):
    ax.barh(y, share[col], left=left, height=0.55, color=c, edgecolor=SURF, linewidth=1.5, label=col)
    for yi, l, v in zip(y, left, share[col]):
        if v >= 6:
            ax.text(l + v / 2, yi, f"{v:.0f}%", ha="center", va="center", color=INK if c in (GRAY, AQUA) else "white", fontsize=9)
    left += share[col].to_numpy()
ax.set_yticks(y, [f"{s} trials\n(n = {k:,})" for s, k in zip(share.index, n)])
ax.set_xlim(0, 100); ax.set_xlabel("Share of stops (%)"); ax.grid(axis="y", visible=False)
ax.set_title("When does a firm's 8-K name a stopped trial?", loc="left", pad=22)
ax.text(0, 1.03, "Terminated, suspended or withdrawn trials at 8-K filers, by how many trials the firm leads",
        transform=ax.transAxes, color=INK2, fontsize=9)
ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0, -0.28), fontsize=9)
save(fig, "fig1_stop_disclosure.png")

# ---- Figure 2: 8-K mentions of a stopped trial, by week around the registry date ----
precise = links[(links["link_type"] == "nct") | (links["n_name_trials"] <= 5)][["accession", "nct_id"]].drop_duplicates()
ment = precise.merge(filings[["accession", "event_date"]], on="accession")
ev = stops.reset_index()[["index", "nct_id", "registry_date", "size"]].merge(ment, on="nct_id")
ev["week"] = np.floor((ev["event_date"] - ev["registry_date"]).dt.days / 7).astype(int)
ev = ev[ev["week"].between(-13, 52)].drop_duplicates(["index", "week"])
weeks = np.arange(-13, 53)
fig, ax = plt.subplots(figsize=(8, 3.6))
for s, c in [("1-20", BLUE), ("21-100", ORANGE)]:
    rate = ev[ev["size"] == s].groupby("week").size().reindex(weeks, fill_value=0) / n[s] * 100
    ax.plot(weeks, rate, color=c, linewidth=2, label=f"Firms leading {s} trials")
    ax.text(weeks[-1] + 0.8, rate.iloc[-8:].mean(), f"{s} trials", color=INK2, va="center", fontsize=9)
ax.axvline(0, color=INK2, linewidth=1, linestyle="--")
ax.text(0.4, ax.get_ylim()[1] * 0.95, "registry shows the stop", color=INK2, fontsize=9, va="top")
ax.set_xlim(-13, 58); ax.set_ylim(bottom=0)
ax.set_xlabel("Weeks from the registry date"); ax.set_ylabel("% of stops with an 8-K naming the trial")
ax.set_title("8-Ks naming a stopped trial, week by week", loc="left")
ax.legend(loc="upper right", fontsize=9)
save(fig, "fig2_mentions_event_time.png")

# ---- Figure 3: readouts per year ----
ro["year"] = pd.to_datetime(ro["event_date"]).dt.year
per = ro[ro["year"] <= 2025].groupby(["year", "readout_kind"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(8, 3.6))
bottom = np.zeros(len(per))
for col, c, lab in [("topline_or_endpoint", BLUE, "Topline or primary endpoint"), ("data_update", ORANGE, "Data update")]:
    vals = per.get(col, pd.Series(0, index=per.index)).to_numpy()
    ax.bar(per.index, vals, bottom=bottom, width=0.7, color=c, edgecolor=SURF, linewidth=1.5, label=lab)
    bottom += vals
ax.set_xticks(per.index, [str(y) if y % 2 == 0 else "" for y in per.index])
ax.grid(axis="x", visible=False)
ax.set_ylabel("Readout 8-Ks"); ax.set_title("Trial readouts announced in 8-Ks, per year", loc="left")
ax.legend(loc="upper left", fontsize=9)
save(fig, "fig3_readouts_per_year.png")

# ---- Figure 4: time of day readouts reach EDGAR ----
t = pd.to_datetime(ro["acceptance_et"])
hrs = t.dt.hour + t.dt.minute / 60
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.axvspan(9.5, 16, color=GRID, alpha=0.6, linewidth=0)
ax.text(12.75, 1, "market open", transform=ax.get_xaxis_transform(), ha="center", va="top", color=INK2, fontsize=9)
ax.hist(hrs, bins=np.arange(0, 24.5, 0.5), color=BLUE, edgecolor=SURF, linewidth=1)
ax.set_xticks(range(0, 25, 3), [f"{h}:00" for h in range(0, 25, 3)])
ax.set_xlim(0, 24); ax.grid(axis="x", visible=False)
ax.set_xlabel("EDGAR acceptance time (Eastern)"); ax.set_ylabel("Readout 8-Ks per half hour")
ax.set_title("When readout 8-Ks are filed", loc="left")
save(fig, "fig4_readout_time_of_day.png")
print("Saved to", FIG_DIR)

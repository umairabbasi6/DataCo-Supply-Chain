"""Figure 1 - DataCo inflation decomposition (reads results/A2_*.csv)."""
import pandas as pd
import matplotlib.pyplot as plt
from common import RES, FIG, C1, C2, C_GRAY, INK, INK2

dec = pd.read_csv(f"{RES}/A2_dataco_inflation_decomposition.csv").iloc[:6]
lab = ["S0 Original\n(as submitted)", "S1 − Order\nStatus", "S2 − shipping\ndate",
       "S3 − order-date\nstring & PII", "S4 + grouped\nCV", "S5 Clean\nfeature set"]
vals = dec["XGB AUC mean"].values
fig, ax = plt.subplots(figsize=(7.2, 3.4))
bars = ax.bar(lab, vals - 0.5, bottom=0.5, color=[C2] + [C_GRAY] * 4 + [C1],
              width=0.6, edgecolor="white", linewidth=2)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center",
            va="bottom", color=INK, fontsize=8)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("XGBoost ROC-AUC (5-fold CV)\nbars start at chance = 0.5")
ax.set_title("DataCo: how much of the reported AUC is leakage")
ax.grid(axis="x", visible=False)
ax.tick_params(axis="x", labelsize=7.5)
fig.savefig(f"{FIG}/Fig1_dataco_inflation_decomposition.png")
print("Fig 1 saved")

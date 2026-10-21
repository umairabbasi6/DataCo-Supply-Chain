"""Figure 2 - IoT label validity (reads raw DS3 and results/B3*, B5*)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import RAW, RES, FIG, SEED, C1, C2, C3, C_GRAY, INK, INK2

rng = np.random.default_rng(SEED)
ds3 = pd.read_csv(f"{RAW}/IoT_Traffic_Dataset.csv")
lvl, spd = ds3["Congestion_Level"].values, ds3["Traffic_Speed_kmph"].values
perm = pd.read_csv(f"{RES}/B3b_ds2_permutation_null.csv")["null_auc"].values
pt = pd.read_csv(f"{RES}/B3_ds2_permutation_test.csv").iloc[0]
abl = pd.read_csv(f"{RES}/B5_fused_corpus_ablation.csv").set_index("Setting")["RF ROC-AUC"]

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
# (a)
ax = axes[0]
jit = rng.uniform(-0.25, 0.25, len(lvl))
for k, c, n in [(0, C3, "0 free flow"), (1, C1, "1 moderate"), (2, C2, "2 heavy = 'anomaly'")]:
    m = lvl == k
    ax.scatter(spd[m], lvl[m] + jit[m], s=8, color=c, alpha=0.6, edgecolors="none", label=n)
for t in (20, 50):
    ax.axvline(t, color=INK, lw=1, ls="--")
    ax.text(t + 1.5, -0.42, f"{t} km/h", fontsize=7, color=INK)
ax.set_yticks([0, 1, 2]); ax.set_ylim(-0.5, 2.4)
ax.set(xlabel="Traffic speed (km/h)", ylabel="Congestion level")
ax.set_title("(a) DS3: label is a speed threshold")
ax.grid(axis="y", visible=False)
# (b)
ax = axes[1]
ax.hist(perm, bins=25, color=C_GRAY, edgecolor="white", label="null: shuffled labels")
ax.axvline(pt["Observed AUC"], color=C2, lw=2,
           label=f"observed AUC = {pt['Observed AUC']:.3f}")
ax.set(xlabel="Random Forest ROC-AUC (5-fold CV)", ylabel="Permutations (n = 200)")
ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
ax.set_title(f"(b) DS2: label not learnable (p = {pt['Permutation p-value (200 perms)']:.2f})")
ax.legend(fontsize=7, loc="upper right")
ax.grid(axis="x", visible=False)
# (c)
ax = axes[2]
items = [("All 17\nfeatures", abl["All 17 features (5-fold CV)"], C2),
         ("Source\nflag only", abl["dataset_source flag ONLY"], C_GRAY),
         ("Source flag\n+ speed", abl["dataset_source + speed (2 features)"], C_GRAY),
         ("Within\nDS2 rows", abl["Fused model, AUC within DS2 rows only"], C1),
         ("Within\nDS3 rows", abl["Fused model, AUC within DS3 rows only"], C1)]
v = np.array([i[1] for i in items])
b = ax.bar([i[0] for i in items], v - 0.5, bottom=0.5, color=[i[2] for i in items],
           width=0.6, edgecolor="white", linewidth=2)
for bar, val in zip(b, v):
    ax.text(bar.get_x() + bar.get_width() / 2, max(val, 0.5) + 0.01, f"{val:.3f}",
            ha="center", fontsize=7.5, color=INK)
ax.set_ylim(0.5, 1.06)
ax.set_ylabel("Random Forest ROC-AUC\n(bars start at chance = 0.5)")
ax.set_title("(c) Fused corpus: source of the AUC")
ax.tick_params(axis="x", labelsize=7); ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(f"{FIG}/Fig2_iot_label_validity.png")
print("Fig 2 saved")

"""Figures 3-4 - corrected DataCo benchmark (reads data/processed OOF + results/C6)."""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from common import PROC, RES, FIG, C1, C2, C3, C4, C_GRAY, INK2

oof = pd.read_csv(f"{PROC}/dataco_oof_predictions.csv.gz")
y = oof["y"].values
show = [("XGBoost", C1, "-"), ("Random Forest", C3, "-"), ("MLP (feed-forward NN)", C4, "-"),
        ("Logistic Regression", C2, "-"), ("Shipping Mode only (LR)", C_GRAY, "--")]
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
for name, c, ls in show:
    p = oof[name].values
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, color=c, ls=ls, lw=1.8, label=f"{name} ({roc_auc_score(y, p):.3f})")
    pr, rc, _ = precision_recall_curve(y, p)
    axes[1].plot(rc, pr, color=c, ls=ls, lw=1.8, label=f"{name} ({average_precision_score(y, p):.3f})")
axes[0].plot([0, 1], [0, 1], color=INK2, lw=1, ls=":")
axes[0].set(xlabel="False positive rate", ylabel="True positive rate",
            title="(a) ROC, pooled out-of-fold")
axes[0].legend(fontsize=6.5, loc="lower right", title="model (ROC-AUC)", title_fontsize=6.5)
axes[1].axhline(y.mean(), color=INK2, lw=1, ls=":")
axes[1].text(0.01, y.mean() - 0.025, f"late-delivery prevalence {y.mean():.3f}", fontsize=6.5, color=INK2)
axes[1].set(xlabel="Recall", ylabel="Precision", ylim=(0.4, 1.02),
            title="(b) Precision-Recall, pooled out-of-fold")
axes[1].legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(0.0, 0.3), title="model (PR-AUC)", title_fontsize=6.5)
fig.tight_layout()
fig.savefig(f"{FIG}/Fig3_dataco_roc_pr_corrected.png")

imp = pd.read_csv(f"{RES}/C6_permutation_importance_xgb.csv").head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.barh(imp["Feature"], imp["AUC drop mean"], xerr=imp["AUC drop SD"], color=C1, height=0.6,
        edgecolor="white", error_kw=dict(ecolor=INK2, lw=1))
ax.set_xlabel("Drop in ROC-AUC when the feature is permuted")
ax.set_title("XGBoost permutation importance (grouped fold 1)")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(f"{FIG}/Fig4_dataco_permutation_importance.png")
print("Figs 3-4 saved")

"""
STEP 3 - Corrected, leakage-free DataCo late-delivery benchmark.

* Pre-shipment features only (see results/C0_feature_set.csv for every column
  kept or excluded, with the reason).
* All preprocessing (encoding, scaling) is fitted inside each training fold
  through scikit-learn Pipelines.
* 5-fold StratifiedGroupKFold grouped by Order Id (items of the same order are
  never split across train and test).
* Every model is evaluated with the same folds; mean ± SD reported.
* ROC-AUC, PR-AUC, Brier, Accuracy, Precision, Recall, F1 (threshold 0.5).
* DeLong tests on pooled out-of-fold predictions.
* Temporal hold-out (train < 2017-07-01, test >= 2017-07-01) as a second check.
* Permutation importance on raw feature names (no off-by-one possible).
* Small XGBoost hyper-parameter sensitivity grid (grouped inner CV).
"""
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_curve, precision_recall_curve, confusion_matrix)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from common import (load_dataco, build_clean_dataco, CLEAN_CAT, CLEAN_NUM, EXCLUDED,
                    SEED, N_FOLDS, PROC, FIG, save, delong_test,
                    C1, C2, C3, C4, C_GRAY, INK, INK2)

t0 = time.time()
d = load_dataco()
X, y, groups, od = build_clean_dataco(d)
print(f"Clean feature matrix: {X.shape}, late rate {y.mean():.3f}")

# Save the exact modelling table used
out = X.copy(); out.insert(0, "Order Id", groups)
out.insert(1, "order_date", od.dt.strftime("%Y-%m-%d %H:%M")); out["Late_delivery_risk"] = y
out.to_csv(f"{PROC}/dataco_clean_modelling_table.csv.gz", index=False, compression="gzip")

# Feature documentation table
fs = [(c, "kept", "categorical", "Known when the order is placed") for c in CLEAN_CAT]
fs += [(c, "kept", "numeric",
        "Derived from order date / order basket" if c in
        ("order_month", "order_weekday", "order_hour", "items_in_order", "order_value")
        else "Known when the order is placed") for c in CLEAN_NUM]
fs += [(c, "excluded", "-", r) for c, r in EXCLUDED.items()]
save(pd.DataFrame(fs, columns=["Column", "Status", "Type", "Reason"]), "C0_feature_set.csv")

# ── Preprocessors ──
def onehot_pre():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                              sparse_output=False), CLEAN_CAT),
        ("num", StandardScaler(), CLEAN_NUM)])

def ordinal_pre():
    return ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CLEAN_CAT),
        ("num", "passthrough", CLEAN_NUM)])

# ═══════════════ MLP configuration (grouped inner 3-fold on a 60k-row sample) ═══════════════
# sklearn's built-in early stopping uses a ROW-level validation split, which leaks
# items of the same order and lets the network over-fit; we therefore select the
# size / L2 penalty / number of epochs with GROUP-aware inner validation instead.
import warnings; warnings.filterwarnings("ignore")
_rs = np.random.default_rng(SEED)
_ss = _rs.choice(len(y), 60000, replace=False)
_inner = list(StratifiedGroupKFold(3, shuffle=True, random_state=SEED)
              .split(X.iloc[_ss], y[_ss], groups[_ss]))
mlp_grid = []
for hidden in [(32,), (64,), (64, 32)]:
    for alpha in (1e-3, 1e-2, 1e-1):
        for epochs in (5, 10, 20):
            aucs = []
            for tr, te in _inner:
                m = Pipeline([("pre", onehot_pre()), ("m", MLPClassifier(
                    hidden_layer_sizes=hidden, alpha=alpha, batch_size=512,
                    max_iter=epochs, early_stopping=False, random_state=SEED))])
                m.fit(X.iloc[_ss[tr]], y[_ss[tr]])
                aucs.append(roc_auc_score(y[_ss[te]], m.predict_proba(X.iloc[_ss[te]])[:, 1]))
            mlp_grid.append((str(hidden), alpha, epochs, round(np.mean(aucs), 4),
                             round(np.std(aucs), 4)))
mlp_grid = pd.DataFrame(mlp_grid, columns=["hidden_layers", "alpha", "epochs",
                                           "AUC mean", "AUC SD"])
save(mlp_grid, "C9_mlp_selection_grid.csv")
_b = mlp_grid.sort_values("AUC mean", ascending=False).iloc[0]
MLP_BEST = {"hidden": eval(_b["hidden_layers"]), "alpha": float(_b["alpha"]),
            "epochs": int(_b["epochs"])}
print(f"  MLP selected: {MLP_BEST} (inner AUC {_b['AUC mean']}) ({time.time()-t0:.0f}s)")

MODELS = {
    "Majority baseline": lambda: Pipeline([("pre", ordinal_pre()),
                                           ("m", DummyClassifier(strategy="prior"))]),
    "Shipping Mode only (LR)": lambda: Pipeline([
        ("pre", ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"),
                                    ["Shipping Mode"])])),
        ("m", LogisticRegression(max_iter=1000))]),
    "Logistic Regression": lambda: Pipeline([("pre", onehot_pre()),
                                             ("m", LogisticRegression(max_iter=3000, C=1.0))]),
    "Random Forest": lambda: Pipeline([("pre", ordinal_pre()),
        ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                     max_features="sqrt", random_state=SEED, n_jobs=2))]),
    "XGBoost": lambda: Pipeline([("pre", ordinal_pre()),
        ("m", XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                            eval_metric="logloss", random_state=SEED, n_jobs=2))]),
    "MLP (feed-forward NN)": lambda: Pipeline([("pre", onehot_pre()),
        ("m", MLPClassifier(hidden_layer_sizes=MLP_BEST["hidden"], alpha=MLP_BEST["alpha"],
                            batch_size=512, max_iter=MLP_BEST["epochs"],
                            early_stopping=False, random_state=SEED))]),
}

def metrics(yt, p):
    yp = (p >= 0.5).astype(int)
    return dict(ROC_AUC=roc_auc_score(yt, p), PR_AUC=average_precision_score(yt, p),
                Brier=brier_score_loss(yt, p), Accuracy=accuracy_score(yt, yp),
                Precision=precision_score(yt, yp, zero_division=0),
                Recall=recall_score(yt, yp, zero_division=0),
                F1=f1_score(yt, yp, zero_division=0))

# ═══════════════ Grouped 5-fold CV ═══════════════
cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
folds = list(cv.split(X, y, groups))
for i, (tr, te) in enumerate(folds):  # integrity check
    assert not set(groups[tr]) & set(groups[te]), "order leaked across folds"
oof = {m: np.zeros(len(y)) for m in MODELS}
per_fold = []
for name, mk in MODELS.items():
    for k, (tr, te) in enumerate(folds):
        m = mk().fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        oof[name][te] = p
        per_fold.append(dict(Model=name, Fold=k + 1, **metrics(y[te], p)))
        if name == "XGBoost" and k == 0:
            xgb_fold1, te_fold1 = m, te
    print(f"  {name:26s} done ({time.time()-t0:.0f}s)")

pf = pd.DataFrame(per_fold)
save(pf.round(4), "C1_cv_per_fold.csv")
agg = pf.drop(columns="Fold").groupby("Model", sort=False).agg(["mean", "std"])
summary = pd.DataFrame({"Model": agg.index})
for met in ["ROC_AUC", "PR_AUC", "Brier", "Accuracy", "Precision", "Recall", "F1"]:
    summary[met] = [f"{a:.4f} ± {b:.4f}" for a, b in zip(agg[(met, "mean")], agg[(met, "std")])]
save(summary, "C2_cv_summary_mean_sd.csv")
print(summary.to_string(index=False))
pd.DataFrame(oof).assign(y=y, order_id=groups).to_csv(
    f"{PROC}/dataco_oof_predictions.csv.gz", index=False, compression="gzip")

# Confusion matrices (pooled OOF, threshold 0.5)
cms = []
for name in MODELS:
    tn, fp, fn, tp = confusion_matrix(y, (oof[name] >= 0.5).astype(int)).ravel()
    cms.append((name, tp, tn, fp, fn))
save(pd.DataFrame(cms, columns=["Model", "TP", "TN", "FP", "FN"]), "C3_confusion_pooled_oof.csv")

# ═══════════════ DeLong tests (pooled OOF) ═══════════════
dl = []
for other in ["Random Forest", "Logistic Regression", "MLP (feed-forward NN)",
              "Shipping Mode only (LR)"]:
    a1, a2, z, p = delong_test(y, oof["XGBoost"], oof[other])
    dl.append(("XGBoost", other, round(a1, 4), round(a2, 4), round(a1 - a2, 4),
               round(z, 2), f"{p:.2e}"))
a1, a2, z, p = delong_test(y, oof["Random Forest"], oof["MLP (feed-forward NN)"])
dl.append(("Random Forest", "MLP (feed-forward NN)", round(a1, 4), round(a2, 4),
           round(a1 - a2, 4), round(z, 2), f"{p:.2e}"))
save(pd.DataFrame(dl, columns=["Model A", "Model B", "AUC A", "AUC B", "Diff",
                               "z", "p-value"]), "C4_delong_tests.csv")
print(pd.DataFrame(dl).to_string(index=False))

# ═══════════════ Temporal hold-out ═══════════════
cut = pd.Timestamp("2017-07-01")
tr_t = np.where(od < cut)[0]; te_t = np.where(od >= cut)[0]
tmp = []
for name, mk in MODELS.items():
    p = mk().fit(X.iloc[tr_t], y[tr_t]).predict_proba(X.iloc[te_t])[:, 1]
    tmp.append(dict(Model=name, **{k: round(v, 4) for k, v in metrics(y[te_t], p).items()}))
tmp = pd.DataFrame(tmp)
tmp.insert(1, "Train rows", len(tr_t)); tmp.insert(2, "Test rows", len(te_t))
save(tmp, "C5_temporal_holdout.csv")
print(tmp.to_string(index=False))
print(f"  temporal done ({time.time()-t0:.0f}s)")

# ═══════════════ Permutation importance (XGBoost, fold 1) ═══════════════
rs = np.random.default_rng(SEED)
sub = rs.choice(te_fold1, size=min(20000, len(te_fold1)), replace=False)
pi = permutation_importance(xgb_fold1, X.iloc[sub], y[sub], scoring="roc_auc",
                            n_repeats=5, random_state=SEED, n_jobs=1)
imp = (pd.DataFrame({"Feature": X.columns, "AUC drop mean": pi.importances_mean,
                     "AUC drop SD": pi.importances_std})
       .sort_values("AUC drop mean", ascending=False))
save(imp.round(5), "C6_permutation_importance_xgb.csv")
print(imp.head(10).to_string(index=False))

# ═══════════════ XGBoost sensitivity grid (grouped inner 3-fold, 60k sample) ═══════════════
ss = rs.choice(len(y), 60000, replace=False)
inner = StratifiedGroupKFold(3, shuffle=True, random_state=SEED)
grid = []
for depth in (3, 6, 9):
    for lr in (0.05, 0.1):
        for n in (200, 600):
            aucs = []
            for tr, te in inner.split(X.iloc[ss], y[ss], groups[ss]):
                m = Pipeline([("pre", ordinal_pre()), ("m", XGBClassifier(
                    n_estimators=n, max_depth=depth, learning_rate=lr, subsample=0.8,
                    colsample_bytree=0.8, tree_method="hist", eval_metric="logloss",
                    random_state=SEED, n_jobs=2))]).fit(X.iloc[ss[tr]], y[ss[tr]])
                aucs.append(roc_auc_score(y[ss[te]], m.predict_proba(X.iloc[ss[te]])[:, 1]))
            grid.append((depth, lr, n, round(np.mean(aucs), 4), round(np.std(aucs), 4)))
save(pd.DataFrame(grid, columns=["max_depth", "learning_rate", "n_estimators",
                                 "AUC mean", "AUC SD"]), "C7_xgb_sensitivity_grid.csv")
print(f"  grid done ({time.time()-t0:.0f}s)")

# ═══════════════ Before vs after table ═══════════════
cvm = agg.xs("mean", axis=1, level=1)
paper = {"Logistic Regression": (0.725, 0.670), "Random Forest": (0.888, 0.700),
         "XGBoost": (0.931, 0.802), "LSTM / MLP": (0.797, 0.685)}
ba = []
for k, (auc_p, f1_p) in paper.items():
    mk = "MLP (feed-forward NN)" if k == "LSTM / MLP" else k
    ba.append((k, auc_p, round(cvm.loc[mk, "ROC_AUC"], 4), f1_p, round(cvm.loc[mk, "F1"], 4)))
save(pd.DataFrame(ba, columns=["Model", "Paper AUC (leaky, row split)",
                               "Corrected AUC (grouped 5-fold)", "Paper F1",
                               "Corrected F1"]), "C8_before_vs_after.csv")

# ═══════════════ What the signal actually is ═══════════════
sig = (pd.DataFrame({"Shipping Mode": X["Shipping Mode"],
                     "Order placed": np.where(X["order_hour"] < 12, "before 12:00", "12:00 or later"),
                     "late": y})
       .groupby(["Shipping Mode", "Order placed"])["late"].agg(["mean", "size"])
       .round(3).reset_index().rename(columns={"mean": "Late rate", "size": "Rows"}))
save(sig, "C10_late_rate_by_mode_and_hour.csv")
print(sig.to_string(index=False))

# Figures 3 and 4 are drawn by 08_figures_3_4.py from the saved OOF predictions.
print(f"STEP 3 done in {time.time()-t0:.0f}s")

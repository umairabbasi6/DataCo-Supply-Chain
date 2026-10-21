"""How far can a simple lookup table go?  Compares the ML models with the late
rate of each Shipping Mode x (order before/after 12:00) x payment Type cell.

Lookup tables are evaluated two ways:
  * in-sample        (cell rate computed on all rows; optimistic, kept for reference)
  * out-of-fold      (cell rates learned on the training folds only, scored on the
                      held-out fold; same order-grouped folds as 03_benchmark_dataco.py)
The out-of-fold numbers are the ones to compare with XGBoost / Random Forest.

Reads data/processed/ files written by 03_benchmark_dataco.py."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from common import PROC, SEED, N_FOLDS, save

N_BOOT = 200

X = pd.read_csv(f"{PROC}/dataco_clean_modelling_table.csv.gz")
oof = pd.read_csv(f"{PROC}/dataco_oof_predictions.csv.gz")
y = X["Late_delivery_risk"].values
groups = X["Order Id"].values
X["pm"] = (X["order_hour"] >= 12).astype(int)

# The saved tables must be row-aligned with each other
assert (oof["y"].values == y).all() and (oof["order_id"].values == groups).all(), \
    "OOF predictions are not aligned with the modelling table"

# Same folds as 03_benchmark_dataco.py (depends only on y, groups, seed)
folds = list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
             .split(X, y, groups))
for tr, te in folds:
    assert not set(groups[tr]) & set(groups[te]), "order leaked across folds"


def cell_key(cols):
    return X[cols].astype(str).agg("|".join, axis=1).values


def in_sample(cols):
    return X.groupby(cols)["Late_delivery_risk"].transform("mean").values


def out_of_fold(cols):
    key = cell_key(cols)
    pred = np.zeros(len(X))
    for tr, te in folds:
        rate = pd.Series(y[tr]).groupby(key[tr]).mean()
        pred[te] = pd.Series(key[te]).map(rate).fillna(y[tr].mean()).values
    return pred


specs = [
    ("Shipping Mode", ["Shipping Mode"]),
    ("Shipping Mode x before/after 12:00", ["Shipping Mode", "pm"]),
    ("Shipping Mode x before/after 12:00 x payment Type", ["Shipping Mode", "pm", "Type"]),
]

rows, preds = [], {}
for name, cols in specs:
    n_cells = X.groupby(cols).ngroups
    p_in = in_sample(cols)
    p_oof = out_of_fold(cols)
    preds[name] = p_oof
    rows.append((f"{name} ({n_cells} cells), in-sample", roc_auc_score(y, p_in)))
    rows.append((f"{name} ({n_cells} cells), grouped 5-fold out-of-fold",
                 roc_auc_score(y, p_oof)))

rows += [
    ("XGBoost, grouped 5-fold out-of-fold (24 features)", roc_auc_score(y, oof["XGBoost"])),
    ("Random Forest, grouped 5-fold out-of-fold (24 features)",
     roc_auc_score(y, oof["Random Forest"])),
]
df = pd.DataFrame(rows, columns=["Predictor", "ROC-AUC"]).round(4)
save(df, "C11_rule_ceiling.csv")

# ── Order-level bootstrap of the AUC difference (ML model - 32-cell lookup) ──
# Resamples whole orders, so items of one order are never treated as independent.
# Conditional on the out-of-fold predictions (models are not refitted per resample).
rng = np.random.default_rng(SEED)
order_rows = pd.Series(np.arange(len(X))).groupby(groups).apply(lambda s: s.values)
order_ids = order_rows.index.values
lookup32 = preds["Shipping Mode x before/after 12:00 x payment Type"]
xgb_p = oof["XGBoost"].values
rf_p = oof["Random Forest"].values

diff_xgb, diff_rf = [], []
for _ in range(N_BOOT):
    pick = rng.choice(len(order_ids), size=len(order_ids), replace=True)
    idx = np.concatenate(order_rows.iloc[pick].values)
    yb = y[idx]
    a_l = roc_auc_score(yb, lookup32[idx])
    diff_xgb.append(roc_auc_score(yb, xgb_p[idx]) - a_l)
    diff_rf.append(roc_auc_score(yb, rf_p[idx]) - a_l)


def ci(d):
    d = np.asarray(d)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


bt = []
for name, d in [("XGBoost minus 32-cell lookup", diff_xgb),
                ("Random Forest minus 32-cell lookup", diff_rf)]:
    m, lo, hi = ci(d)
    bt.append((name, round(m, 4), round(lo, 4), round(hi, 4), N_BOOT))
save(pd.DataFrame(bt, columns=["AUC difference (out-of-fold)", "Mean", "CI 2.5%",
                               "CI 97.5%", "Order-level bootstrap reps"]),
     "C13_lookup_vs_ml_bootstrap.csv")

# ── Late rate by payment Type x shipping mode ──
ct = (X.groupby(["Type", "Shipping Mode"])["Late_delivery_risk"].mean().round(3)
      .unstack().reset_index())
save(ct, "C12_late_rate_by_type_and_mode.csv")

print(df.to_string(index=False))
print()
print(pd.DataFrame(bt, columns=["Difference", "Mean", "CI 2.5%", "CI 97.5%", "Reps"])
      .to_string(index=False))
print()
print(ct.to_string(index=False))
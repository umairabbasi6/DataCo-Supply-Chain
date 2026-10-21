"""
STEP 1 - DataCo leakage audit + inflation decomposition.

(a) Leakage tests: which columns encode the target after the fact.
(b) Inflation decomposition: re-run the ORIGINAL pipeline (STEP1/STEP2 of the
    submitted paper) and remove one source of inflation at a time, using the
    same XGBoost configuration throughout, so each AUC drop is attributable.
(c) Replicate the Random Forest CV figure reported in the paper (0.9608) and
    show why it differs from the single-split figure (0.888).
"""
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                     train_test_split)
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from common import (load_dataco, build_clean_dataco, CLEAN_CAT, CLEAN_NUM, SEED,
                    N_FOLDS, save, is_text, FIG, C1, C2, C_GRAY, INK, INK2)

t0 = time.time()
d = load_dataco()
y = d["Late_delivery_risk"].values
print(f"DataCo: {d.shape}")

# ═══════════════════════ (a) LEAKAGE TESTS ═══════════════════════
od = pd.to_datetime(d["order date (DateOrders)"])
sd = pd.to_datetime(d["shipping date (DateOrders)"])
gap = (sd - od).dt.days
sched = d["Days for shipment (scheduled)"]
real = d["Days for shipping (real)"]
canc = d["Order Status"].isin(["CANCELED", "SUSPECTED_FRAUD"])
items = d.groupby("Order Id").size()

tests = [
    ("Delivery Status", "Late delivery <=> target = 1",
     f"{((d['Delivery Status'] == 'Late delivery').astype(int) == y).mean():.4f}",
     "Identical to target"),
    ("Days for shipping (real)", "rule: real > scheduled reproduces target",
     f"{((real > sched).astype(int) == y).mean():.4f}", "Target is defined from it"),
    ("shipping date (DateOrders)", "shipping date - order date == real shipping days",
     f"{(gap == real).mean():.4f}", "Rebuilds the removed leakage column"),
    ("shipping date (DateOrders)", "rule: (ship - order) > scheduled reproduces target",
     f"{((gap > sched).astype(int) == y).mean():.4f}", "Post-event"),
    ("Order Status", "share of CANCELED / SUSPECTED_FRAUD rows labelled late",
     f"{y[canc].mean():.4f} (n={canc.sum():,})", "Post-event outcome; 0% late by construction"),
    ("Order Id", "orders whose items all share one label",
     f"{(d.groupby('Order Id')['Late_delivery_risk'].nunique() == 1).mean():.4f}",
     "Row-level split puts items of the same order in train and test"),
    ("Order Id", "mean items (rows) per order", f"{items.mean():.2f}",
     "Needs group-aware CV"),
]
lt = pd.DataFrame(tests, columns=["Column", "Test", "Result", "Implication"])
save(lt, "A1_dataco_leakage_tests.csv")
print(lt.to_string(index=False))

# ═══════════════ (b) INFLATION DECOMPOSITION (XGBoost) ═══════════════
ORIG_DROP = ["Delivery Status", "Days for shipping (real)", "Customer Email",
             "Customer Password", "Product Image", "Order Zipcode"]
ORIG_ID_DROP = ["Late_delivery_risk", "Order Item Id", "Order Id", "Customer Id",
                "Product Card Id"]


def original_features(df):
    """Faithful re-implementation of the submitted paper's STEP1 + STEP2."""
    x = df.drop(columns=ORIG_DROP).copy()
    x = x.fillna(x.median(numeric_only=True))
    for c in x.columns:
        if is_text(x[c]):
            x[c] = LabelEncoder().fit_transform(x[c].astype(str))
    x = x.drop(columns=[c for c in ORIG_ID_DROP if c in x.columns])
    x = x.select_dtypes("number").fillna(0)
    return x


X_orig = original_features(d)
print(f"Original feature matrix: {X_orig.shape[1]} features")
X_orig_scaled = StandardScaler().fit_transform(X_orig)  # scaler on ALL rows (as submitted)
groups = d["Order Id"].values


def xgb():
    spw = (y == 0).sum() / (y == 1).sum()
    return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                         scale_pos_weight=spw, eval_metric="logloss",
                         tree_method="hist", random_state=SEED, n_jobs=2)


def cv_auc(model_fn, X, grouped):
    X = np.asarray(X, dtype=float)
    if grouped:
        splitter = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
        it = splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
        it = splitter.split(X, y)
    aucs = []
    for tr, te in it:
        m = model_fn().fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return np.mean(aucs), np.std(aucs)


# Single 80/20 split as in the paper (reproduces the reported 0.931)
Xtr, Xte, ytr, yte = train_test_split(X_orig_scaled, y, test_size=0.2,
                                      random_state=SEED, stratify=y)
single = roc_auc_score(yte, xgb().fit(Xtr, ytr).predict_proba(Xte)[:, 1])
print(f"Replicated single-split XGBoost AUC (paper reported 0.931): {single:.4f}")

steps = []
def add(label, X, grouped, note):
    m, s = cv_auc(xgb, X, grouped)
    steps.append((label, "Grouped by Order Id" if grouped else "Random rows",
                  X.shape[1], round(m, 4), round(s, 4), note))
    print(f"  {label:58s} AUC={m:.4f} ± {s:.4f}  ({time.time()-t0:.0f}s)")

cols = list(X_orig.columns)
def without(*names):
    return X_orig[[c for c in cols if c not in names]]

add("S0 Original pipeline (as submitted)", X_orig, False,
    "Order Status + date strings + names included; row-level CV")
add("S1 - Order Status", without("Order Status"), False, "Remove post-event status")
add("S2 - shipping date", without("Order Status", "shipping date (DateOrders)"), False,
    "Remove post-event shipping timestamp")
add("S3 - order-date string & customer PII",
    without("Order Status", "shipping date (DateOrders)", "order date (DateOrders)",
            "Customer Fname", "Customer Lname", "Customer Street", "Order Customer Id",
            "Customer Zipcode"), False,
    "Remove near-unique strings that let the model memorise individual orders")
X_s3 = without("Order Status", "shipping date (DateOrders)", "order date (DateOrders)",
               "Customer Fname", "Customer Lname", "Customer Street",
               "Order Customer Id", "Customer Zipcode")
add("S4 S3 + grouped CV (same order never in train and test)", X_s3, True,
    "Group-aware evaluation")

# S5: the clean, documented feature set used in the benchmark
Xc, yc, gc, _ = build_clean_dataco(d)
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
Xc_num = np.column_stack([enc.fit_transform(Xc[CLEAN_CAT]), Xc[CLEAN_NUM].values])
add("S5 Clean pre-shipment feature set + grouped CV", pd.DataFrame(Xc_num), True,
    "Final corrected setting (see STEP 3 benchmark)")

# Leak control: what a fully leaking feature does
Xleak = X_s3.copy(); Xleak["ship_minus_order_days"] = gap.values
add("Control: S4 + parsed (shipping - order) days", Xleak, True,
    "Shows the ceiling when the post-event timestamp is parsed")

dec = pd.DataFrame(steps, columns=["Setting", "CV split", "n_features",
                                   "XGB AUC mean", "XGB AUC SD", "What changed"])
dec.loc[len(dec)] = ["Paper single 80/20 split (replicated)", "Random rows",
                     X_orig.shape[1], round(single, 4), np.nan,
                     "Reported in paper as 0.931"]
save(dec, "A2_dataco_inflation_decomposition.csv")

# ═══════════════ (c) RANDOM FOREST: 0.888 vs 0.9608 explained ═══════════════
rf_rows = []
def rf_paper():   # the model in Table 5 of the paper
    return RandomForestClassifier(n_estimators=200, max_depth=15, class_weight="balanced",
                                  random_state=SEED, n_jobs=2)
def rf_cv():      # the model actually used in the paper's cross-validation
    return RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=2)

single_rf = roc_auc_score(yte, rf_paper().fit(Xtr, ytr).predict_proba(Xte)[:, 1])
rf_rows.append(("RF(200, depth 15, balanced) - single 80/20 split", round(single_rf, 4),
                "Paper Table 5 reports 0.888"))
m, s = cv_auc(rf_cv, X_orig_scaled, False)
rf_rows.append(("RF(100, unlimited depth) - 5-fold row CV", f"{m:.4f} ± {s:.4f}",
                "Paper Table 7 reports 0.9608 ± 0.0029 - a DIFFERENT model"))
print(f"  RF cv-config row CV: {m:.4f}  ({time.time()-t0:.0f}s)")
m, s = cv_auc(rf_cv, X_orig_scaled, True)
rf_rows.append(("RF(100, unlimited depth) - 5-fold grouped CV", f"{m:.4f} ± {s:.4f}",
                "Same model, orders kept together"))
print(f"  RF cv-config grouped CV: {m:.4f}  ({time.time()-t0:.0f}s)")
save(pd.DataFrame(rf_rows, columns=["Model / evaluation", "AUC", "Note"]),
     "A3_rf_discrepancy_explained.csv")

# Figure 1 is drawn by 04_figure1.py from results/A2_dataco_inflation_decomposition.csv
print(f"Done in {time.time()-t0:.0f}s")

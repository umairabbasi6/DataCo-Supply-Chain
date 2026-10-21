"""
STEP 2 - Label-validity audit of the two IoT datasets and of the fused corpus.

DS2 Supply_chain_data.csv  : is the 'anomaly' label (Risk Factor) learnable at all?
DS3 IoT_Traffic_Dataset.csv: is Congestion_Level a deterministic function of a feature?
Fused corpus (DS2 + DS3, as built in the submitted paper's STEP 2B):
    how much of the reported AUC comes from knowing the source file?
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold,
                                     cross_val_score, cross_val_predict,
                                     permutation_test_score, train_test_split)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (roc_auc_score, adjusted_rand_score, silhouette_score,
                             precision_recall_curve)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from common import (RAW, PROC, RES, FIG, SEED, save, is_text, C1, C2, C3, C_GRAY, INK, INK2)

rng = np.random.default_rng(SEED)
cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
def rf(n=200): return RandomForestClassifier(n, random_state=SEED, n_jobs=2)

# ═══════════════════════════════ DS2 ═══════════════════════════════
ds2 = pd.read_csv(f"{RAW}/Supply_chain_data.csv")
num2 = ["GPS Latitude", "GPS Longitude", "Temperature", "Humidity", "Speed",
        "Delay Time", "Cost"]
cat2 = ["Stage", "Entity", "Delivery Status", "Disruption Type"]
y2 = ds2["Risk Factor"].isin(["Equipment Failure", "Human Error"]).astype(int).values

rows = []
for c in num2:
    x = ds2[c]; u = (x - x.min()) / (x.max() - x.min())
    p = stats.kstest(u, "uniform").pvalue
    rows.append((c, "KS test vs Uniform(min,max) (p)", round(p, 3),
                 "consistent with uniform random draws" if p > 0.05 else "deviates from uniform"))
rows.append(("Timestamp", "distinct time-of-day values in 1,000 rows",
             ds2["Timestamp"].str[11:].nunique(), "generated timestamps"))
for c in ["Transaction ID", "RFID ID", "Route ID"]:
    rows.append((c, "distinct values", ds2[c].nunique(), "unique per row (identifier)"))
def cramers_v(a, b):
    ct = pd.crosstab(a, b); chi = stats.chi2_contingency(ct)
    return chi.pvalue, np.sqrt(chi.statistic / (ct.values.sum() * (min(ct.shape) - 1)))
for c in num2:
    p = stats.mannwhitneyu(ds2[c][y2 == 1], ds2[c][y2 == 0]).pvalue
    rows.append((c, "Mann-Whitney U, anomaly vs normal (p)", round(p, 3),
                 "no difference" if p > 0.05 else "difference"))
for c in cat2:
    p, v = cramers_v(ds2[c], y2)
    rows.append((c, "Chi-square vs anomaly label (p); Cramer's V", f"{p:.3f}; V={v:.3f}",
                 "no association" if p > 0.05 else f"weak association (V={v:.2f})"))
p, v = cramers_v(ds2["Risk Factor"], ds2["Disruption Type"])
rows.append(("Risk Factor x Disruption Type", "Chi-square (p); Cramer's V", f"{p:.3f}; V={v:.3f}",
             "independent" if p > 0.05 else f"weak association (V={v:.2f})"))
ds2_tests = pd.DataFrame(rows, columns=["Variable", "Test", "Value", "Reading"])
save(ds2_tests, "B1_ds2_randomness_tests.csv")

# Features exactly as mapped in the submitted STEP 2B (10 features)
X2 = pd.DataFrame({
    "temperature": ds2["Temperature"], "humidity": ds2["Humidity"], "speed": ds2["Speed"],
    "gps_lat": ds2["GPS Latitude"], "gps_lon": ds2["GPS Longitude"],
    "delay_time": ds2["Delay Time"], "cost": ds2["Cost"],
    "route_id": LabelEncoder().fit_transform(ds2["Route ID"]),
    "stage": LabelEncoder().fit_transform(ds2["Stage"]),
    "entity": LabelEncoder().fit_transform(ds2["Entity"])})

labels2 = {
    "Equipment Failure + Human Error (paper)": y2,
    "+ Weather (alternative rule)": ds2["Risk Factor"].isin(
        ["Equipment Failure", "Human Error", "Weather"]).astype(int).values,
    "Risk Factor = Equipment Failure": (ds2["Risk Factor"] == "Equipment Failure").astype(int).values,
    "Risk Factor = Human Error": (ds2["Risk Factor"] == "Human Error").astype(int).values,
    "Disruption Type = Equipment Failure": (ds2["Disruption Type"] == "Equipment Failure").astype(int).values,
    "Delivery Status = In Transit": (ds2["Delivery Status"] == "In Transit").astype(int).values,
}
models2 = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Random Forest": rf(),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                             random_state=SEED, n_jobs=2, eval_metric="logloss"),
}
res2 = []
for ln, yy in labels2.items():
    for mn, m in models2.items():
        s = cross_val_score(m, X2, yy, cv=rcv, scoring="roc_auc", n_jobs=1)
        res2.append((ln, round(yy.mean(), 3), mn, round(s.mean(), 4), round(s.std(), 4)))
        print(f"DS2 | {ln:40s} | {mn:20s} AUC {s.mean():.3f} ± {s.std():.3f}")
# Extra check: does any non-sensor column help? (Disruption Type is weakly associated)
X2_all = pd.get_dummies(ds2[num2 + cat2], dtype=float)
s = cross_val_score(rf(), X2_all, y2, cv=rcv, scoring="roc_auc")
res2.append(("Equipment Failure + Human Error (paper)", round(y2.mean(), 3),
             "Random Forest, ALL non-ID columns incl. Disruption Type", round(s.mean(), 4), round(s.std(), 4)))
s = cross_val_score(LogisticRegression(max_iter=1000), pd.get_dummies(ds2["Disruption Type"], dtype=float),
                    y2, cv=rcv, scoring="roc_auc")
res2.append(("Equipment Failure + Human Error (paper)", round(y2.mean(), 3),
             "Logistic Regression, Disruption Type only", round(s.mean(), 4), round(s.std(), 4)))
res2 = pd.DataFrame(res2, columns=["Label definition", "Positive rate", "Model",
                                   "AUC mean (5x10 CV)", "AUC SD"])
save(res2, "B2_ds2_learnability.csv")

score, perm, pval = permutation_test_score(rf(100), X2, y2, cv=cv5, scoring="roc_auc",
                                           n_permutations=200, random_state=SEED, n_jobs=1)
save(pd.DataFrame([("Random Forest", round(score, 4), round(perm.mean(), 4),
                    round(perm.std(), 4), round(pval, 4))],
                  columns=["Model", "Observed AUC", "Null mean AUC", "Null SD",
                           "Permutation p-value (200 perms)"]),
     "B3_ds2_permutation_test.csv")
pd.DataFrame({"null_auc": perm}).to_csv(f"{RES}/B3b_ds2_permutation_null.csv", index=False)
print(f"DS2 permutation test: observed {score:.3f}, null {perm.mean():.3f}, p={pval:.3f}")

# ═══════════════════════════════ DS3 ═══════════════════════════════
ds3 = pd.read_csv(f"{RAW}/IoT_Traffic_Dataset.csv")
feat3 = [c for c in ds3.columns if c != "Congestion_Level"]
lvl = ds3["Congestion_Level"].values
spd = ds3["Traffic_Speed_kmph"].values
tree = DecisionTreeClassifier(max_depth=2, random_state=SEED).fit(ds3[feat3], lvl)
rule = np.where(spd <= 20, 2, np.where(spd <= 50, 1, 0))
r3 = [
    ("3-class depth-2 decision tree, training accuracy", f"{tree.score(ds3[feat3], lvl):.4f}"),
    ("Tree uses only", ", ".join(sorted({feat3[i] for i in tree.tree_.feature if i >= 0}))),
    ("Rule: level = 2 if speed<=20, 1 if 20<speed<=50, else 0 — agreement", f"{(rule == lvl).mean():.4f}"),
    ("Paper label (level==2) == (speed<=20) agreement", f"{((lvl == 2) == (spd <= 20)).mean():.4f}"),
    ("Alt. label (level>=1) == (speed<=50) agreement", f"{((lvl >= 1) == (spd <= 50)).mean():.4f}"),
    ("ROC-AUC of (−speed) alone for level==2", f"{roc_auc_score(lvl == 2, -spd):.4f}"),
    ("ROC-AUC of (−speed) alone for level>=1", f"{roc_auc_score(lvl >= 1, -spd):.4f}"),
]
for c in ["Traffic_Density_vpkm", "Vehicle_Count", "FFT_Feature_1", "Time_of_Day_min"]:
    r3.append((f"Mann-Whitney p, {c}: level 2 vs others",
               f"{stats.mannwhitneyu(ds3[c][lvl == 2], ds3[c][lvl != 2]).pvalue:.3f}"))
for name, yy in [("level==2", (lvl == 2).astype(int)), ("level>=1", (lvl >= 1).astype(int))]:
    s_all = cross_val_score(rf(), ds3[feat3], yy, cv=cv5, scoring="roc_auc")
    s_ns = cross_val_score(rf(), ds3[[c for c in feat3 if c != "Traffic_Speed_kmph"]], yy,
                           cv=cv5, scoring="roc_auc")
    r3.append((f"RF 5-fold AUC, all features ({name})", f"{s_all.mean():.4f}"))
    r3.append((f"RF 5-fold AUC, speed removed ({name})", f"{s_ns.mean():.4f}"))
ds3_tests = pd.DataFrame(r3, columns=["Test", "Result"])
save(ds3_tests, "B4_ds3_label_determinism.csv")
print(ds3_tests.to_string(index=False))
print(export_text(tree, feature_names=feat3))

# ═══════════════════════ FUSED CORPUS (as submitted) ═══════════════════════
COLUMN_MAP = {"Temperature": "temperature", "Humidity": "humidity", "Speed": "speed",
              "GPS Latitude": "gps_lat", "GPS Longitude": "gps_lon",
              "Delay Time": "delay_time", "Cost": "cost", "Route ID": "route_id",
              "Stage": "stage", "Entity": "entity", "Latitude": "gps_lat",
              "Longitude": "gps_lon", "Traffic_Speed_kmph": "speed",
              "Traffic_Density_vpkm": "traffic_density", "FFT_Feature_1": "fft_1",
              "FFT_Feature_2": "fft_2", "FFT_Feature_3": "fft_3",
              "Vehicle_Count": "vehicle_count", "Time_of_Day_min": "time_of_day"}

def extract(df):
    out = {}
    for o, s in COLUMN_MAP.items():
        if o in df.columns and s not in out:
            col = df[o]
            out[s] = (LabelEncoder().fit_transform(col.astype(str)) if is_text(col)
                      else col.values)
    return pd.DataFrame(out)

p2 = extract(ds2); p2["anomaly"] = y2; p2["dataset_source"] = 0
p3 = extract(ds3); p3["anomaly"] = (lvl == 2).astype(int); p3["dataset_source"] = 1
fused = pd.concat([p2, p3], ignore_index=True, sort=False)
for c in fused.columns:
    if c not in ("anomaly", "dataset_source"):
        fused[c] = pd.to_numeric(fused[c], errors="coerce")
        fused[c] = fused[c].fillna(fused[c].median())
fused.to_csv(f"{PROC}/fused_iot_corpus_rebuilt.csv", index=False)
yf = fused["anomaly"].values
src = fused["dataset_source"].values
featf = [c for c in fused.columns if c != "anomaly"]
print(f"Fused corpus rebuilt: {fused.shape}, anomaly rate {yf.mean():.3f}, "
      f"features {len(featf)}")

# (i) replicate the paper's single split (scaler fitted on all rows, as submitted)
Xs = StandardScaler().fit_transform(fused[featf])
Xtr, Xte, ytr, yte = train_test_split(Xs, yf, test_size=0.2, random_state=SEED, stratify=yf)
rfp = RandomForestClassifier(200, max_depth=15, class_weight="balanced",
                             random_state=SEED, n_jobs=2).fit(Xtr, ytr)
single = roc_auc_score(yte, rfp.predict_proba(Xte)[:, 1])
print(f"Replicated fused single-split RF AUC (paper: 0.949): {single:.4f}")

# (ii) ablations, 5-fold CV
def cvauc(cols):
    s = cross_val_score(rf(), fused[cols], yf, cv=cv5, scoring="roc_auc")
    return s.mean(), s.std()
abl = [("Paper setting: all 17 features, single 80/20 split", round(single, 4), np.nan)]
for name, cols in [
    ("All 17 features (5-fold CV)", featf),
    ("dataset_source flag ONLY", ["dataset_source"]),
    ("speed ONLY", ["speed"]),
    ("dataset_source + speed (2 features)", ["dataset_source", "speed"]),
    ("All features EXCEPT dataset_source", [c for c in featf if c != "dataset_source"]),
    ("All features EXCEPT dataset_source and speed",
     [c for c in featf if c not in ("dataset_source", "speed")]),
]:
    m, s = cvauc(cols)
    abl.append((name, round(m, 4), round(s, 4)))
    print(f"Fused | {name:50s} AUC {m:.4f}")

# Within-source performance of the fused model (out-of-fold)
oof = cross_val_predict(rf(), fused[featf], yf, cv=cv5, method="predict_proba")[:, 1]
abl.append(("Fused model, AUC within DS2 rows only", round(roc_auc_score(yf[src == 0], oof[src == 0]), 4), np.nan))
abl.append(("Fused model, AUC within DS3 rows only", round(roc_auc_score(yf[src == 1], oof[src == 1]), 4), np.nan))
abl = pd.DataFrame(abl, columns=["Setting", "RF ROC-AUC", "SD"])
save(abl, "B5_fused_corpus_ablation.csv")
print(abl.to_string(index=False))

# (iii) Recall 0.963 at precision 0.50: reachable by any model by moving the threshold
prec, rec, thr = precision_recall_curve(yf, oof)
ok = prec[:-1] >= 0.50
rec_at = rec[:-1][ok].max()
save(pd.DataFrame([("Random Forest (OOF, fused)", "precision >= 0.50", round(rec_at, 4),
                    "Paper: LSTM recall 0.963 at precision 0.500")],
                  columns=["Model", "Operating point", "Max recall", "Comparison"]),
     "B6_recall_threshold_check.csv")
print(f"RF recall at precision>=0.50: {rec_at:.3f}")

# (iv) Clustering: do the k=3 clusters just recover the source file?
km = KMeans(3, random_state=SEED, n_init=10).fit(Xs)
ct = pd.crosstab(km.labels_, src, rownames=["cluster"], colnames=["source (0=DS2,1=DS3)"])
ari = adjusted_rand_score(src, km.labels_)
sil = silhouette_score(Xs, km.labels_, sample_size=4000, random_state=SEED)
ct_out = ct.reset_index()
ct_out["anomaly_rate"] = pd.Series(yf).groupby(km.labels_).mean().round(3).values
save(ct_out, "B7_clusters_vs_source.csv")
save(pd.DataFrame([("Adjusted Rand Index (clusters vs source)", round(ari, 4)),
                   ("Silhouette (k=3)", round(sil, 4))], columns=["Metric", "Value"]),
     "B8_cluster_metrics.csv")
print(ct, f"\nARI={ari:.3f}, silhouette={sil:.3f}")

# Figure 2 is drawn by 05_figure2.py from the saved results.
print("STEP 2 done")

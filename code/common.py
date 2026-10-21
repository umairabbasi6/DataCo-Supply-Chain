"""
Shared paths, constants, plotting style and statistics helpers.
Project: Leakage and label-validity audit of public supply-chain datasets
         + corrected DataCo late-delivery benchmark.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
for d in (PROC, RES, FIG):
    os.makedirs(d, exist_ok=True)

SEED = 42
N_FOLDS = 5

# ── Colour roles (validated reference categorical palette, light mode) ──
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C_GRAY = "#8a8985"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2,
})


def load_dataco():
    return pd.read_csv(os.path.join(RAW, "DataCoSupplyChainDataset.csv"),
                       encoding="latin-1")


def is_text(s):
    """True for object or pandas string dtype (pandas 2 and 3 safe)."""
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


# ── DeLong test for two correlated ROC AUCs (Sun & Xu, 2014 fast algorithm) ──
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        a = i
        while a < n and z[a] == z[i]:
            a += 1
        t[i:a] = 0.5 * (i + a - 1) + 1
        i = a
    out = np.empty(n)
    out[j] = t
    return out


def _fast_delong(preds_sorted, m):
    n = preds_sorted.shape[1] - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds_sorted[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_test(y, p1, p2):
    """Two-sided DeLong test. Returns (auc1, auc2, z, p)."""
    y = np.asarray(y)
    order = np.argsort(-y, kind="mergesort")
    m = int(y.sum())
    preds = np.vstack([np.asarray(p1), np.asarray(p2)])[:, order]
    aucs, cov = _fast_delong(preds, m)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else np.inf
    p = 2 * stats.norm.sf(abs(z))
    return aucs[0], aucs[1], z, p


# ── Clean, pre-shipment feature set for DataCo (used by the corrected benchmark) ──
CLEAN_CAT = ["Type", "Shipping Mode", "Customer Segment", "Market", "Order Region",
             "Order Country", "Category Name", "Department Name"]
CLEAN_NUM = ["Days for shipment (scheduled)", "Order Item Quantity",
             "Order Item Product Price", "Order Item Discount Rate",
             "Order Item Discount", "Sales", "Order Item Total",
             "Order Item Profit Ratio", "Order Profit Per Order",
             "Latitude", "Longitude",
             "order_month", "order_weekday", "order_hour",
             "items_in_order", "order_value"]

# Columns excluded from the clean set, with the reason (written to results/)
EXCLUDED = {
    "Late_delivery_risk": "Target",
    "Delivery Status": "Post-event: encodes the target (Late delivery = 1 exactly)",
    "Days for shipping (real)": "Post-event: actual shipping days; target = real > scheduled",
    "shipping date (DateOrders)": "Post-event: shipping date - order date reproduces real shipping days",
    "Order Status": "Post-event: CANCELED / SUSPECTED_FRAUD are always labelled not-late",
    "Order Id": "Identifier (used only as the grouping key for CV)",
    "Order Item Id": "Identifier", "Customer Id": "Identifier",
    "Order Customer Id": "Identifier", "Product Card Id": "Identifier",
    "Order Item Cardprod Id": "Identifier", "Category Id": "Duplicate of Category Name",
    "Department Id": "Duplicate of Department Name",
    "Product Category Id": "Duplicate of Category Name",
    "Customer Email": "PII / masked constant", "Customer Password": "PII / masked constant",
    "Customer Fname": "PII", "Customer Lname": "PII", "Customer Street": "PII",
    "Customer Zipcode": "PII / high cardinality", "Order Zipcode": "86% missing",
    "Customer City": "High-cardinality store location (Latitude/Longitude kept)",
    "Customer State": "High-cardinality store location (Latitude/Longitude kept)",
    "Customer Country": "Store country (2 values; redundant with Latitude/Longitude)",
    "Order City": "High cardinality (3,597 levels); Order Country/Region kept",
    "Order State": "High cardinality (1,089 levels); Order Country/Region kept",
    "Product Description": "100% missing", "Product Image": "URL string",
    "Product Name": "High cardinality; Category Name kept",
    "Product Price": "Duplicate of Order Item Product Price",
    "Product Status": "Constant", "Sales per customer": "Duplicate of Order Item Total",
    "Benefit per order": "Duplicate of Order Profit Per Order",
    "order date (DateOrders)": "Raw string; replaced by month / weekday / hour",
}


def build_clean_dataco(d):
    """Return X (DataFrame of clean features), y, groups (Order Id), order_date."""
    od = pd.to_datetime(d["order date (DateOrders)"])
    X = d[CLEAN_CAT + [c for c in CLEAN_NUM if c in d.columns]].copy()
    X["order_month"] = od.dt.month
    X["order_weekday"] = od.dt.weekday
    X["order_hour"] = od.dt.hour
    X["items_in_order"] = d.groupby("Order Id")["Order Item Id"].transform("size")
    X["order_value"] = d.groupby("Order Id")["Sales"].transform("sum")
    X = X[CLEAN_CAT + CLEAN_NUM]
    for c in CLEAN_CAT:
        X[c] = X[c].astype(str)
    return X, d["Late_delivery_risk"].values, d["Order Id"].values, od


def save(df, name):
    path = os.path.join(RES, name)
    df.to_csv(path, index=False)
    return path

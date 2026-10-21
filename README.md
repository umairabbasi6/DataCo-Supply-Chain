# Leakage and Label-Validity Audit of Public Supply-Chain Datasets + Corrected DataCo Benchmark

Reproducible code, data and results for the revised study (Option 2 + 3).

## Folder structure

```
scm_audit/
├── README.md                  <- this file
├── RESULTS_SUMMARY.md         <- every headline number, with the file it comes from
├── requirements.txt
├── code/
│   ├── common.py              <- paths, clean feature definition, DeLong test, plot style
│   ├── 01_audit_dataco.py     <- DataCo leakage tests + inflation decomposition
│   ├── 02_audit_iot.py        <- DS2 / DS3 label validity + fused-corpus confounding
│   ├── 03_benchmark_dataco.py <- corrected grouped-CV benchmark (LR, RF, XGBoost, MLP)
│   ├── 04_figure1.py          <- Fig. 1 (from results/A2)
│   ├── 05_figure2.py          <- Fig. 2 (from DS3 + results/B3, B5)
│   ├── 07_rule_ceiling.py     <- simple lookup-rule vs ML comparison (C11, C12)
│   ├── 08_figures_3_4.py      <- Figs. 3-4 (from OOF predictions + results/C6)
│   ├── 06_results_workbook.py <- bundles all tables into results/ALL_RESULTS.xlsx
│   ├── run_all.py             <- runs everything in the right order
│   └── _original_STEP2B_from_notebook.py  <- original fusion code (reference only)
├── data/
│   ├── raw/                   <- the 3 datasets + DataCo column dictionary
│   └── processed/             <- clean DataCo modelling table, rebuilt fused IoT corpus,
│                                 out-of-fold predictions
├── results/                   <- every table as CSV (A* = DataCo audit, B* = IoT audit,
│                                 C* = corrected benchmark), ALL_RESULTS.xlsx, run logs
└── figures/                   <- Fig1-Fig4, 300 dpi PNG
```

## Datasets (data/raw)

| File | Used as | Source |
|---|---|---|
| DataCoSupplyChainDataset.csv | DS1 - late-delivery prediction | Constante, Silva & Pereira (2019), Mendeley Data V5, doi:10.17632/8gx2fvg2k6.5 |
| DescriptionDataCoSupplyChain.csv | column dictionary | same |
| Supply_chain_data.csv | DS2 - "Supply Chain IoT" | Kaggle, programmer3, *Smart Logistics IoT Dataset* |
| IoT_Traffic_Dataset.csv | DS3 - "IoT Traffic Flow" | Kaggle, programmer3, *IoT-Based Traffic Flow and Congestion Dataset* |

`tokenized_access_logs.csv` (DataCo web clickstream) is **not used**. The raw DataCo
file (96 MB) is not included in the zip because of size; place it in `data/raw/`
before running.

## How to run

```bash
pip install -r requirements.txt
cd code
python run_all.py          # or run 01, 02, 03 separately
```

Google Colab: upload the `scm_audit` folder, then
`!pip install xgboost` and `%cd scm_audit/code` and `!python run_all.py`.
All randomness is seeded (seed = 42).

## What each step does

**01_audit_dataco.py**
* Tests each suspicious column against the target (Delivery Status, real shipping
  days, shipping date, Order Status, order structure).
* Re-implements the submitted paper's preprocessing exactly and removes one
  inflation source at a time with an identical XGBoost model (Fig. 1).
* Explains the reviewer's RF 0.888 vs 0.9608 discrepancy (two different RF models).

**02_audit_iot.py**
* DS2: uniformity tests, label-feature association tests, cross-validated
  learnability under six label definitions, 200-permutation test.
* DS3: shows Congestion_Level is an exact speed threshold (<=20 / <=50 km/h).
* Fused corpus: rebuilds it exactly as the paper did, then ablates the source flag
  and speed, reports within-source AUC, the recall-threshold check, and whether
  K-Means clusters merely recover the source file (Fig. 2).

**03_benchmark_dataco.py**
* Pre-shipment features only; every excluded column is listed with its reason
  (`results/C0_feature_set.csv`).
* Encoding/scaling fitted inside each training fold (Pipelines).
* 5-fold StratifiedGroupKFold by Order Id, asserted leak-free.
* LR, RF, XGBoost, MLP + majority and Shipping-Mode-only baselines.
* ROC-AUC, PR-AUC, Brier, Accuracy, Precision, Recall, F1 (mean +/- SD).
* DeLong tests, temporal hold-out (train < 2017-07-01), permutation importance,
  XGBoost sensitivity grid, before-vs-after table (Figs. 3-4).

## Note on the LSTM

The original LSTM received single-timestep inputs, so it could not use temporal
information. DataCo rows are independent order items, not sensor sequences, so the
benchmark uses a feed-forward neural network (MLP) instead and says so explicitly.

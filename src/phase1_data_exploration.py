"""
phase1_data_exploration.py
Phase 1 — Data Gathering & Exploration
Run: python src/phase1_data_exploration.py
"""
import os, sys, json, joblib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.preprocessing import LabelEncoder

from config import (
    KPI_CSV, CELL_XLSX, CLEAN_CSV, ENC_PKL, CFG_JSON,
    TARGET_COLS, TARGET_LABELS, FEATURE_COLS,
    DATA_DIR, MODELS_DIR, OUTPUTS_DIR, COLORS, LAYER_COLORS, SEED
)

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

# ── STEP 1: Load raw data ────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 1 — Data Gathering & Exploration")
print("=" * 60)

df_kpi  = pd.read_csv(KPI_CSV)
df_cell = pd.read_excel(CELL_XLSX, sheet_name="4G")

print(f"\nKPI dataset  : {df_kpi.shape[0]:,} rows x {df_kpi.shape[1]} columns")
print(f"Cell file    : {df_cell.shape[0]:,} rows x {df_cell.shape[1]} columns")
print(f"Date range   : {df_kpi['Date'].min()}  to  {df_kpi['Date'].max()}")
print(f"Unique cells : {df_kpi['Cell_ID'].nunique()}")
print(f"Unique BTS   : {df_kpi['BTS_ID'].nunique()}")
print(f"Layers       : {sorted(df_kpi['Layer'].unique().tolist())}")

# ── STEP 2: Data types & statistics ─────────────────────────────────────────
print("\n── Column data types ──")
print(df_kpi.dtypes)
print("\n── Target variable statistics ──")
print(df_kpi[TARGET_COLS].describe().round(3))

# ── STEP 3: Missing values ───────────────────────────────────────────────────
missing     = df_kpi.isnull().sum()
missing_pct = (missing / len(df_kpi) * 100).round(2)
missing_df  = pd.DataFrame({"Count": missing, "Pct (%)": missing_pct})
missing_df  = missing_df[missing_df["Count"] > 0]
print(f"\n── Missing values ({len(missing_df)} columns affected) ──")
print(missing_df)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
if len(missing_df):
    axes[0].barh(missing_df.index, missing_df["Pct (%)"],
                 color="#C73E1D", alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("Missing (%)")
    axes[0].set_title("Missing Values by Column", fontweight="bold")
msno.matrix(df_kpi, ax=axes[1], fontsize=8, sparkline=False,
            color=(0.27, 0.52, 0.71))
axes[1].set_title("Data Presence Matrix", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "p1_missing_values.png"))
plt.close()

# ── STEP 4: Outlier detection ────────────────────────────────────────────────
print("\n── Outlier analysis (IQR x 1.5) ──")
print(f"  {'Column':<30}  {'Lower':>10}  {'Upper':>10}  {'Outliers':>9}  {'Pct%':>6}")
for col in TARGET_COLS:
    q1, q3 = df_kpi[col].quantile([0.25, 0.75])
    iqr     = q3 - q1
    lo, hi  = q1 - 1.5*iqr, q3 + 1.5*iqr
    n_out   = ((df_kpi[col] < lo) | (df_kpi[col] > hi)).sum()
    print(f"  {col:<30}  {lo:>10.3f}  {hi:>10.3f}  {n_out:>9}  {n_out/len(df_kpi)*100:>5.1f}%")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
for i, (col, c) in enumerate(zip(TARGET_COLS, COLORS)):
    data = df_kpi[col].dropna()
    axes[i].boxplot(data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor=c, alpha=0.65),
                    medianprops=dict(color="black", linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4, markerfacecolor=c))
    axes[i].set_title(TARGET_LABELS[col], fontweight="bold")
    axes[i].set_xticks([])
plt.suptitle("Outlier Detection — Target Boxplots", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "p1_outliers.png"))
plt.close()

# ── STEP 5: Target distributions ────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
for i, (col, c) in enumerate(zip(TARGET_COLS, COLORS)):
    data = df_kpi[col].dropna()
    axes[i].hist(data, bins=50, color=c, edgecolor="white", alpha=0.85)
    axes[i].axvline(data.mean(),   color="black", ls="--", lw=2, label=f"Mean={data.mean():.2f}")
    axes[i].axvline(data.median(), color="white", ls=":", lw=2, label=f"Median={data.median():.2f}")
    axes[i].set_title(TARGET_LABELS[col], fontweight="bold")
    axes[i].legend(fontsize=8)
plt.suptitle("Target Variable Distributions", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "p1_distributions.png"))
plt.close()

# ── STEP 6: Feature engineering ─────────────────────────────────────────────
df = df_kpi.copy()
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Temporal features
df["DayOfWeek"]  = df["Date"].dt.dayofweek
df["DayOfMonth"] = df["Date"].dt.day
df["Month"]      = df["Date"].dt.month
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
df["IsWeekend"]  = (df["DayOfWeek"] >= 5).astype(int)

# Drop useless columns
df.drop(columns=["Overload_Period", "Region", "Category"], inplace=True)

# Label encoding
le_layer = LabelEncoder()
le_cell  = LabelEncoder()
le_bts   = LabelEncoder()
df["Layer_enc"] = le_layer.fit_transform(df["Layer"])
df["Cell_enc"]  = le_cell.fit_transform(df["Cell_ID"])
df["BTS_enc"]   = le_bts.fit_transform(df["BTS_ID"])
encoders = {"layer": le_layer, "cell": le_cell, "bts": le_bts}

# Merge cell metadata
cell_meta = df_cell[["Cell_ID","Layer","Lat","Lon","Azimuth","PCI","EARFCN","TAC"]]
df = df.merge(cell_meta, on=["Cell_ID","Layer"], how="left")

# Derived features
df["DL_UL_Vol_Ratio"] = df["DataVol_DL_DRB_GB"] / (df["DataVol_UL_DRB_GB"] + 1e-6)
df["PRB_Total"]       = df["LTE_PrbUtil_DL_pct"] + df["LTE_PrbUtil_UL_pct"]

# Build model dataset
df_model = df[FEATURE_COLS + TARGET_COLS].dropna().reset_index(drop=True)

print(f"\n── Final modelling dataset ──")
print(f"  Shape    : {df_model.shape}")
print(f"  Features : {len(FEATURE_COLS)}")
print(f"  Targets  : {len(TARGET_COLS)}")
print(f"  Missing  : {df_model.isnull().sum().sum()}")

# ── STEP 7: Save outputs ─────────────────────────────────────────────────────
df_model.to_csv(CLEAN_CSV, index=False)
joblib.dump(encoders, ENC_PKL)
col_config = {"FEATURE_COLS": FEATURE_COLS, "TARGET_COLS": TARGET_COLS}
with open(CFG_JSON, "w") as f:
    json.dump(col_config, f, indent=2)

print(f"\n  Saved: {CLEAN_CSV}")
print(f"  Saved: {ENC_PKL}")
print(f"  Saved: {CFG_JSON}")
print("\n✅ Phase 1 complete. Run phase2_visualisation.py next.")

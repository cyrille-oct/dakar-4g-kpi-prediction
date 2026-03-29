"""
phase4_deployment.py
Phase 4 — Model Deployment & Monitoring
Run: python src/phase4_deployment.py
"""
import os, sys, json, joblib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from config import (
    CLEAN_CSV, CFG_JSON, MODELS_DIR, OUTPUTS_DIR,
    TARGET_COLS, TARGET_LABELS, FEATURE_COLS, COLORS
)

os.makedirs(OUTPUTS_DIR, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

print("=" * 60)
print("PHASE 4 — Model Deployment & Monitoring")
print("=" * 60)

# ── Load artefacts ────────────────────────────────────────────────────────────
with open(CFG_JSON) as f:
    cfg = json.load(f)
FEATURE_COLS = cfg["FEATURE_COLS"]
TARGET_COLS  = cfg["TARGET_COLS"]

scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
models = {}
for target in TARGET_COLS:
    safe = target.replace("/","_")
    models[target] = joblib.load(os.path.join(MODELS_DIR, f"model_{safe}.pkl"))
    print(f"  Loaded model_{safe}.pkl  [{type(models[target]).__name__}]")

with open(os.path.join(OUTPUTS_DIR, "test_metrics.json")) as f:
    test_metrics = json.load(f)

# ── Load and reconstruct test set ────────────────────────────────────────────
df = pd.read_csv(CLEAN_CSV)
for enc_col, src_col in [("Layer_enc","Layer"),("Cell_enc","Cell_ID"),("BTS_enc","BTS_ID")]:
    if enc_col not in df.columns and src_col in df.columns:
        df[enc_col] = LabelEncoder().fit_transform(df[src_col])
if "DayOfWeek" not in df.columns and "Date" in df.columns:
    df["Date"]       = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["DayOfWeek"]  = df["Date"].dt.dayofweek
    df["DayOfMonth"] = df["Date"].dt.day
    df["Month"]      = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"]  = (df["DayOfWeek"] >= 5).astype(int)
if "DL_UL_Vol_Ratio" not in df.columns:
    df["DL_UL_Vol_Ratio"] = df["DataVol_DL_DRB_GB"] / (df["DataVol_UL_DRB_GB"] + 1e-6)
    df["PRB_Total"]       = df["LTE_PrbUtil_DL_pct"] + df["LTE_PrbUtil_UL_pct"]

available = [f for f in FEATURE_COLS if f in df.columns]
if len(available) < len(FEATURE_COLS):
    FEATURE_COLS = available
df = df[FEATURE_COLS + TARGET_COLS].dropna().reset_index(drop=True)

split_idx = int(0.80 * len(df))
df_test   = df.iloc[split_idx:].reset_index(drop=True)
X_test_sc = scaler.transform(df_test[FEATURE_COLS].values)

print(f"Test set: {len(df_test):,} rows")

# ── STEP 2: Inference function ────────────────────────────────────────────────
def predict_kpi(raw_features: dict) -> dict:
    """Predict all 4 KPIs for a single cell observation."""
    x    = np.array([[raw_features[f] for f in FEATURE_COLS]])
    x_sc = scaler.transform(x)
    return {t: float(models[t].predict(x_sc)[0]) for t in TARGET_COLS}

sample = dict(zip(FEATURE_COLS, df_test[FEATURE_COLS].values[0]))
result = predict_kpi(sample)
print("\nSingle-sample prediction (first test row):")
print(f"  {'Target':<30}  {'Predicted':>10}  {'Actual':>10}  {'Error':>8}")
for t, pred in result.items():
    actual = df_test[t].iloc[0]
    print(f"  {TARGET_LABELS[t]:<30}  {pred:>10.4f}  {actual:>10.4f}  {abs(actual-pred):>8.4f}")

# ── STEP 3: Batch prediction ──────────────────────────────────────────────────
df_preds = df_test[FEATURE_COLS].copy()
for target in TARGET_COLS:
    df_preds[f"Actual_{target}"]    = df_test[target].values
    df_preds[f"Predicted_{target}"] = models[target].predict(X_test_sc)
    df_preds[f"Error_{target}"]     = df_preds[f"Actual_{target}"] - df_preds[f"Predicted_{target}"]
    df_preds[f"AbsError_{target}"]  = df_preds[f"Error_{target}"].abs()

out_path = os.path.join(OUTPUTS_DIR, "batch_predictions.csv")
df_preds.to_csv(out_path, index=False)
print(f"\nBatch predictions saved -> {out_path}  ({len(df_preds):,} rows)")

# ── STEP 4: Rolling RMSE monitoring ──────────────────────────────────────────
WINDOW    = 50
ALERT_PCT = 0.20
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
axes = axes.flatten()
for i, (target, c) in enumerate(zip(TARGET_COLS, COLORS)):
    err       = df_preds[f"Error_{target}"].values
    r_rmse    = pd.Series(err**2).rolling(WINDOW, min_periods=5).mean().apply(np.sqrt)
    glob_rmse = float(np.sqrt(np.mean(err**2)))
    alert_lvl = glob_rmse * (1 + ALERT_PCT)
    axes[i].plot(r_rmse.index, r_rmse.values, color=c, lw=1.8, label=f"Rolling RMSE (w={WINDOW})")
    axes[i].axhline(glob_rmse, color="black", ls="--", lw=1.2, label=f"Global RMSE={glob_rmse:.3f}")
    axes[i].axhline(alert_lvl, color="red", ls=":", lw=1.5, label=f"Alert (+{int(ALERT_PCT*100)}%)")
    axes[i].fill_between(r_rmse.index, glob_rmse, alert_lvl, alpha=0.08, color="orange")
    alert_pts = r_rmse[r_rmse > alert_lvl]
    if len(alert_pts):
        axes[i].scatter(alert_pts.index, alert_pts.values, color="red", s=20, zorder=5,
                        label=f"{len(alert_pts)} alert(s)")
    axes[i].set_title(TARGET_LABELS[target], fontweight="bold", fontsize=10)
    axes[i].set_xlabel("Test sample index"); axes[i].set_ylabel("RMSE")
    axes[i].legend(fontsize=7, ncol=2); axes[i].grid(True, alpha=0.3)
plt.suptitle("Deployment Monitoring — Rolling RMSE with Alert Thresholds",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "p4_rolling_rmse.png"))
plt.close()

# ── STEP 5: Error distribution ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for i, (target, c) in enumerate(zip(TARGET_COLS, COLORS)):
    err    = df_preds[f"Error_{target}"].values
    abserr = df_preds[f"AbsError_{target}"].values
    axes[0,i].hist(err, bins=45, color=c, edgecolor="white", alpha=0.85)
    axes[0,i].axvline(0, color="black", ls="--", lw=1.5)
    axes[0,i].axvline(err.mean(), color="red", ls=":", lw=1.5, label=f"Mean={err.mean():.3f}")
    axes[0,i].set_title(TARGET_LABELS[target], fontweight="bold", fontsize=9)
    axes[0,i].legend(fontsize=8)
    if i==0: axes[0,i].set_ylabel("Frequency")
    axes[1,i].boxplot(abserr, patch_artist=True, boxprops=dict(facecolor=c, alpha=0.65),
                      medianprops=dict(color="black", linewidth=2.5))
    p95 = np.percentile(abserr, 95)
    axes[1,i].axhline(p95, color="red", ls=":", lw=1.5, label=f"P95={p95:.3f}")
    axes[1,i].legend(fontsize=8)
    if i==0: axes[1,i].set_ylabel("|Error|")
plt.suptitle("Error Distribution — Signed (top) and Absolute (bottom)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "p4_error_distribution.png"))
plt.close()

# ── STEP 6: Concept drift detection ──────────────────────────────────────────
N_BLOCKS   = 4
block_size = len(df_test) // N_BLOCKS
fig, axes  = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
for i, (target, c) in enumerate(zip(TARGET_COLS, COLORS)):
    block_rmses, block_labels = [], []
    for b in range(N_BLOCKS):
        start = b * block_size
        end   = (b+1)*block_size if b < N_BLOCKS-1 else len(df_test)
        err_b = df_preds[f"Error_{target}"].iloc[start:end].values
        block_rmses.append(float(np.sqrt(np.mean(err_b**2))))
        block_labels.append(f"Block {b+1}")
    baseline = block_rmses[0]
    drifts   = [100*(r-baseline)/baseline for r in block_rmses]
    bar_cols = ["#44BBA4" if d<=10 else "#F4720B" if d<=20 else "#C73E1D" for d in drifts]
    bars = axes[i].bar(block_labels, block_rmses, color=bar_cols, edgecolor="white", alpha=0.88)
    axes[i].axhline(baseline, color="black", ls="--", lw=1.3, label=f"Baseline={baseline:.3f}")
    axes[i].set_title(TARGET_LABELS[target], fontweight="bold", fontsize=10)
    axes[i].set_ylabel("RMSE"); axes[i].legend(fontsize=8)
    for bar, drift in zip(bars, drifts):
        color = "green" if abs(drift)<=10 else "red"
        axes[i].text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.015,
                     f"{drift:+.1f}%", ha="center", fontsize=9, color=color, fontweight="bold")
plt.suptitle("Concept Drift — RMSE per Time Block  (Green ≤10%  Orange ≤20%  Red >20%)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "p4_concept_drift.png"))
plt.close()

# ── STEP 7: Monitoring dashboard ─────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)
for i, (target, c) in enumerate(zip(TARGET_COLS, COLORS)):
    y_t   = df_test[target].values
    y_p   = models[target].predict(X_test_sc)
    mae_  = mean_absolute_error(y_t, y_p)
    rmse_ = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2_   = float(r2_score(y_t, y_p))
    ax1   = fig.add_subplot(gs[0,i])
    ax1.scatter(y_t, y_p, alpha=0.25, s=8, color=c)
    lims = [min(y_t.min(),y_p.min()), max(y_t.max(),y_p.max())]
    ax1.plot(lims, lims, "k--", lw=1.2)
    ax1.set_title(f"{TARGET_LABELS[target]}\nR²={r2_:.3f}", fontweight="bold", fontsize=9)
    ax1.set_xlabel("Actual", fontsize=8); ax1.set_ylabel("Predicted", fontsize=8)
    err    = y_t - y_p
    r_rmse = pd.Series(err**2).rolling(30, min_periods=5).mean().apply(np.sqrt)
    ax2    = fig.add_subplot(gs[1,i])
    ax2.plot(r_rmse.index, r_rmse.values, color=c, lw=1.5)
    ax2.axhline(rmse_, color="black", ls="--", lw=1.0)
    ax2.set_title(f"Rolling RMSE\nGlobal={rmse_:.3f}", fontweight="bold", fontsize=9)
    ax2.set_xlabel("Index", fontsize=8)
    ax3 = fig.add_subplot(gs[2,i])
    ax3.hist(err, bins=35, color=c, alpha=0.82, edgecolor="white")
    ax3.axvline(0, color="black", ls="--", lw=1.2)
    ax3.set_title(f"Errors\nMAE={mae_:.3f}", fontweight="bold", fontsize=9)
    ax3.set_xlabel("Actual − Predicted", fontsize=8)
fig.suptitle("Deployment Dashboard — Complete Performance Overview",
             fontsize=14, fontweight="bold", y=1.01)
plt.savefig(os.path.join(OUTPUTS_DIR, "p4_dashboard.png"), dpi=120, bbox_inches="tight")
plt.close()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("PROJECT COMPLETE — FINAL PERFORMANCE SUMMARY")
print("="*72)
print(f"  {'Target':<30}  {'Model':<22}  {'MAE':>8}  {'RMSE':>8}  {'R²':>7}")
print("-"*72)
for t in TARGET_COLS:
    m = test_metrics[t]
    print(f"  {t:<30}  {m['model']:<22}  {m['MAE']:>8.4f}  {m['RMSE']:>8.4f}  {m['R2']:>7.4f}")
print("="*72)
print("\n✅ Phase 4 complete.")

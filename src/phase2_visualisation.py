"""
phase2_visualisation.py
Phase 2 — Data Visualisation & Model Selection
Run: python src/phase2_visualisation.py
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config import (
    KPI_CSV, CLEAN_CSV, CFG_JSON,
    TARGET_COLS, TARGET_LABELS, FEATURE_COLS,
    OUTPUTS_DIR, COLORS, LAYER_COLORS, DOW_LABELS
)

os.makedirs(OUTPUTS_DIR, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

print("=" * 60)
print("PHASE 2 — Data Visualisation & Model Selection")
print("=" * 60)

# ── Load data ────────────────────────────────────────────────────────────────
with open(CFG_JSON) as f:
    cfg = json.load(f)
FEATURE_COLS = cfg["FEATURE_COLS"]
TARGET_COLS  = cfg["TARGET_COLS"]

df_raw = pd.read_csv(KPI_CSV, parse_dates=["Date"])
df_raw["DayOfWeek"] = df_raw["Date"].dt.dayofweek
df_raw = df_raw.sort_values("Date").reset_index(drop=True)

df = pd.read_csv(CLEAN_CSV)
# Safety rebuild of Layer_enc
if "Layer_enc" not in df.columns and "Layer" in df.columns:
    df["Layer_enc"] = LabelEncoder().fit_transform(df["Layer"])

print(f"Raw dataset   : {df_raw.shape}")
print(f"Clean dataset : {df.shape}")

# ── STEP 2: Daily temporal trends ───────────────────────────────────────────
daily = df_raw.groupby("Date")[TARGET_COLS].mean().reset_index()
fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True)
for i, (col, c) in enumerate(zip(TARGET_COLS, COLORS)):
    axes[i].plot(daily["Date"], daily[col], color=c, lw=1.6, alpha=0.9)
    axes[i].fill_between(daily["Date"], daily[col], alpha=0.12, color=c)
    trend = daily[col].rolling(7, center=True, min_periods=4).mean()
    axes[i].plot(daily["Date"], trend, color="black", lw=1.2, ls="--", alpha=0.7, label="7-day avg")
    axes[i].set_ylabel(TARGET_LABELS[col], fontsize=9)
    axes[i].legend(fontsize=8, loc="upper right")
    axes[i].grid(True, alpha=0.3)
axes[0].set_title("Daily KPI Trends — Dakar 4G (Mar–Jun 2024)", fontsize=13, fontweight="bold")
plt.xlabel("Date")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "p2_temporal_trends.png"))
plt.close()

# ── STEP 3: KPI by Layer — Bar + Boxplot ────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
for i, col in enumerate(TARGET_COLS):
    lm  = df_raw.groupby("Layer")[col].mean().sort_values()
    ls  = df_raw.groupby("Layer")[col].std().reindex(lm.index)
    bars = axes[i].bar(lm.index, lm.values,
                       color=[LAYER_COLORS[l] for l in lm.index],
                       edgecolor="white", alpha=0.88)
    axes[i].errorbar(lm.index, lm.values, yerr=ls.values,
                     fmt="none", color="black", capsize=6, lw=1.8)
    axes[i].set_title(TARGET_LABELS[col], fontweight="bold")
    for bar, val in zip(bars, lm.values):
        axes[i].text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                     f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")
plt.suptitle("Mean KPI per Layer (±1 std)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "p2_kpi_by_layer.png"))
plt.close()

# Boxplots per layer
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
layers_ordered = ["L800", "L1800", "L2600"]
for i, col in enumerate(TARGET_COLS):
    data_layers = [df_raw[df_raw["Layer"]==l][col].dropna().values for l in layers_ordered]
    bp = axes[i].boxplot(data_layers, positions=[1,2,3], patch_artist=True,
                         showfliers=True,
                         medianprops=dict(color="black", linewidth=2.5),
                         whiskerprops=dict(linewidth=1.5),
                         flierprops=dict(marker="o", markersize=3, alpha=0.4))
    for patch, color in zip(bp["boxes"], list(LAYER_COLORS.values())):
        patch.set_facecolor(color); patch.set_alpha(0.70)
    axes[i].set_xticks([1,2,3]); axes[i].set_xticklabels(layers_ordered)
    axes[i].set_title(TARGET_LABELS[col], fontweight="bold")
    axes[i].grid(True, axis="y", alpha=0.3)
plt.suptitle("KPI Distribution per Layer — Boxplots", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "p2_boxplots_layer.png"))
plt.close()

# ── STEP 4: Weekly patterns ──────────────────────────────────────────────────
dow_mean = df_raw.groupby("DayOfWeek")[TARGET_COLS].mean()
dow_std  = df_raw.groupby("DayOfWeek")[TARGET_COLS].std()
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
for i, (col, c) in enumerate(zip(TARGET_COLS, COLORS)):
    axes[i].bar(DOW_LABELS, dow_mean[col].values, color=c, alpha=0.82, edgecolor="white")
    axes[i].errorbar(DOW_LABELS, dow_mean[col].values, yerr=dow_std[col].values,
                     fmt="none", color="black", capsize=5, lw=1.5)
    axes[i].axvspan(4.5, 6.5, alpha=0.10, color="gray", label="Weekend")
    axes[i].set_title(TARGET_LABELS[col], fontweight="bold")
    axes[i].legend(fontsize=8)
plt.suptitle("Weekly KPI Patterns — Mean per Day of Week", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "p2_weekly_patterns.png"))
plt.close()

# Weekend vs weekday stats
print("\n── Weekend vs Weekday ──")
for col in TARGET_COLS:
    wkday = df_raw[df_raw["DayOfWeek"]<5][col].mean()
    wkend = df_raw[df_raw["DayOfWeek"]>=5][col].mean()
    diff  = ((wkend-wkday)/wkday)*100
    print(f"  {TARGET_LABELS[col]:<28}  Weekday={wkday:.2f}  Weekend={wkend:.2f}  Diff={diff:+.1f}%")

# ── STEP 5: Correlation heatmap ──────────────────────────────────────────────
numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
corr = df_raw[numeric_cols].corr()

plt.figure(figsize=(18, 14))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="RdYlBu_r", center=0,
            annot=False, linewidths=0.2, cbar_kws={"shrink": 0.55})
plt.title("Full Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "p2_correlation_heatmap.png"))
plt.close()

# Top correlated features
print("\n── Top 10 features correlated with each target ──")
corr_targets = {}
for col in TARGET_COLS:
    if col not in corr.columns:
        continue
    top = corr[col].drop([c for c in TARGET_COLS if c in corr.columns], errors="ignore")
    top = top.abs().sort_values(ascending=False).head(10)
    corr_targets[col] = top
    print(f"\n  {TARGET_LABELS[col]}:")
    for feat, val in top.items():
        print(f"    {feat:<35}  |r| = {val:.3f}")

# ── STEP 6: Scatter — target vs top feature ──────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, (col, c) in enumerate(zip(TARGET_COLS, COLORS)):
    if col not in corr_targets:
        continue
    top_feat = corr_targets[col].index[0]
    if top_feat not in df_raw.columns:
        continue
    for layer, lc in LAYER_COLORS.items():
        mask_l = df_raw["Layer"] == layer
        axes[i].scatter(df_raw.loc[mask_l, top_feat], df_raw.loc[mask_l, col],
                        alpha=0.20, s=10, color=lc, label=layer)
    x_vals = df_raw[top_feat].dropna().values
    y_vals = df_raw[col].dropna().values
    min_len = min(len(x_vals), len(y_vals))
    z = np.polyfit(x_vals[:min_len], y_vals[:min_len], 1)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
    axes[i].plot(x_line, np.poly1d(z)(x_line), color="black", lw=1.8, ls="--", label="Trend")
    axes[i].set_xlabel(top_feat, fontsize=9)
    axes[i].set_ylabel(TARGET_LABELS[col], fontsize=9)
    axes[i].set_title(f"{TARGET_LABELS[col]} vs {top_feat}", fontweight="bold", fontsize=9)
    axes[i].legend(fontsize=7, ncol=2)
plt.suptitle("Scatter Plots — Target vs Best Correlated Feature", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "p2_scatter_top_features.png"))
plt.close()

# ── STEP 7: Pair plot ────────────────────────────────────────────────────────
sample = df_raw[TARGET_COLS + ["Layer"]].dropna().sample(min(2000, len(df_raw)), random_state=42)
g = sns.pairplot(sample, vars=TARGET_COLS, hue="Layer", palette=LAYER_COLORS,
                 diag_kind="kde", plot_kws={"alpha": 0.25, "s": 12})
g.figure.suptitle("Pair Plot — Target Variables by Layer", y=1.02, fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "p2_pair_plot.png"))
plt.close()

# ── STEP 8: Linearity check ──────────────────────────────────────────────────
exclude  = TARGET_COLS + ["Date", "DayOfWeek", "Month"]
lr_feats = [c for c in df_raw.select_dtypes(include=[np.number]).columns if c not in exclude]
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
for i, (col, c) in enumerate(zip(TARGET_COLS, COLORS)):
    subset = df_raw[lr_feats + [col]].dropna()
    X_  = subset[lr_feats].values
    y_  = subset[col].values
    sc  = StandardScaler()
    X_sc = sc.fit_transform(X_)
    lr   = LinearRegression().fit(X_sc, y_)
    y_pred    = lr.predict(X_sc)
    residuals = y_ - y_pred
    r2_linear = lr.score(X_sc, y_)
    axes[i].scatter(y_pred, residuals, alpha=0.20, s=10, color=c)
    axes[i].axhline(0, color="black", ls="--", lw=1.5)
    idx_s  = np.argsort(y_pred)
    smooth = pd.Series(residuals[idx_s]).rolling(200, min_periods=50).mean().values
    axes[i].plot(np.sort(y_pred), smooth, color="red", lw=1.5, label="Residual trend")
    axes[i].set_title(f"{TARGET_LABELS[col]}\nLinear R² = {r2_linear:.3f}", fontweight="bold", fontsize=9)
    axes[i].set_xlabel("Predicted"); axes[i].set_ylabel("Residual")
    axes[i].legend(fontsize=8)
plt.suptitle("Linearity Check — Non-random residuals confirm non-linear models needed",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "p2_linearity_check.png"))
plt.close()

print("\n✅ Phase 2 complete. All plots saved to outputs/. Run phase3_model_training.py next.")

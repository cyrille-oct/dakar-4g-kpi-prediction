"""
phase3_model_training.py
Phase 3 — Model Training & Testing
Run: python src/phase3_model_training.py
"""
import os, sys, json, joblib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection  import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.metrics          import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model     import LinearRegression, Ridge, Lasso
from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree             import DecisionTreeRegressor
from xgboost                  import XGBRegressor
from lightgbm                 import LGBMRegressor

from config import (
    CLEAN_CSV, CFG_JSON, MODELS_DIR, OUTPUTS_DIR,
    TARGET_COLS, TARGET_LABELS, FEATURE_COLS,
    COLORS, SEED, CV_FOLDS, TRAIN_SPLIT
)

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

print("=" * 60)
print("PHASE 3 — Model Training & Testing")
print("=" * 60)

# ── Load data ────────────────────────────────────────────────────────────────
with open(CFG_JSON) as f:
    cfg = json.load(f)
FEATURE_COLS = cfg["FEATURE_COLS"]
TARGET_COLS  = cfg["TARGET_COLS"]

df = pd.read_csv(CLEAN_CSV)

# Auto-rebuild engineered columns if missing
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
    print(f"  WARNING: {len(FEATURE_COLS)-len(available)} features missing — using {len(available)}")
    FEATURE_COLS = available

df = df[FEATURE_COLS + TARGET_COLS].dropna().reset_index(drop=True)
print(f"Dataset: {df.shape}")

# ── STEP 2: Train/test split ──────────────────────────────────────────────────
split_idx  = int(TRAIN_SPLIT * len(df))
X          = df[FEATURE_COLS].values
X_train    = X[:split_idx]
X_test     = X[split_idx:]

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

y_sets = {}
for col in TARGET_COLS:
    y_sets[col] = {"train": df[col].values[:split_idx], "test": df[col].values[split_idx:]}

print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# ── STEP 3: Model catalogue ───────────────────────────────────────────────────
MODELS = {
    "LinearRegression" : LinearRegression(),
    "Ridge"            : Ridge(alpha=1.0),
    "Lasso"            : Lasso(alpha=0.01, max_iter=5000),
    "DecisionTree"     : DecisionTreeRegressor(max_depth=8, random_state=SEED),
    "RandomForest"     : RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
    "GradientBoosting" : GradientBoostingRegressor(n_estimators=100, random_state=SEED),
    "XGBoost"          : XGBRegressor(n_estimators=100, random_state=SEED, tree_method="hist", verbosity=0),
    "LightGBM"         : LGBMRegressor(n_estimators=100, random_state=SEED, verbose=-1),
}

# ── STEP 4: Cross-validation benchmark ───────────────────────────────────────
kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
benchmark_results = {}

for target in TARGET_COLS:
    y_train = y_sets[target]["train"]
    print(f"\nTarget: {TARGET_LABELS[target]}")
    benchmark_results[target] = {}
    for name, model in MODELS.items():
        neg_mse   = cross_val_score(model, X_train_sc, y_train, cv=kf,
                                    scoring="neg_mean_squared_error", n_jobs=-1)
        r2_scores = cross_val_score(model, X_train_sc, y_train, cv=kf,
                                    scoring="r2", n_jobs=-1)
        rmse = float(np.sqrt(-neg_mse.mean()))
        r2   = float(r2_scores.mean())
        benchmark_results[target][name] = {"RMSE": rmse, "R2": r2}
        print(f"  {name:<22}  RMSE={rmse:.4f}  R²={r2:.4f}")

with open(os.path.join(OUTPUTS_DIR, "benchmark_results.json"), "w") as f:
    json.dump(benchmark_results, f, indent=2)

# ── STEP 5: Benchmark chart ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()
for i, target in enumerate(TARGET_COLS):
    bm     = benchmark_results[target]
    names  = list(bm.keys())
    rmses  = [bm[m]["RMSE"] for m in names]
    r2s    = [bm[m]["R2"]   for m in names]
    order  = np.argsort(rmses)
    names_ = [names[j] for j in order]
    rmses_ = [rmses[j] for j in order]
    r2s_   = [r2s[j]   for j in order]
    x      = np.arange(len(names_))
    ax2    = axes[i].twinx()
    b1 = axes[i].bar(x-0.2, rmses_, 0.38, color="#3F88C5", alpha=0.85, label="RMSE ↓")
    b2 = ax2.bar(    x+0.2, r2s_,   0.38, color="#F4720B", alpha=0.85, label="R² ↑")
    b1[0].set_linewidth(2.5); b1[0].set_edgecolor("gold")
    b2[0].set_linewidth(2.5); b2[0].set_edgecolor("gold")
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(names_, rotation=38, ha="right", fontsize=8)
    axes[i].set_title(TARGET_LABELS[target], fontweight="bold", fontsize=10)
    axes[i].set_ylabel("RMSE ↓", color="#3F88C5")
    ax2.set_ylabel("R² ↑", color="#F4720B")
    axes[i].legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
plt.suptitle("Model Benchmark — 5-Fold CV  |  ⭐ = Best", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "p3_benchmark.png"))
plt.close()

# ── STEP 6: Best model selection ─────────────────────────────────────────────
best_model_name = {}
for target in TARGET_COLS:
    bm   = benchmark_results[target]
    best = min(bm, key=lambda m: bm[m]["RMSE"])
    best_model_name[target] = best
    print(f"\n  {TARGET_LABELS[target]}  ->  {best}  (RMSE={bm[best]['RMSE']:.4f}  R²={bm[best]['R2']:.4f})")

# ── STEP 7: Hyperparameter tuning ────────────────────────────────────────────
PARAM_GRIDS = {
    "XGBoost"          : {"n_estimators":[100,200], "max_depth":[4,6], "learning_rate":[0.05,0.1], "subsample":[0.8,1.0]},
    "LightGBM"         : {"n_estimators":[100,200], "max_depth":[4,6], "learning_rate":[0.05,0.1], "num_leaves":[31,63]},
    "RandomForest"     : {"n_estimators":[100,200], "max_depth":[None,10], "min_samples_split":[2,5]},
    "GradientBoosting" : {"n_estimators":[100,200], "max_depth":[3,5], "learning_rate":[0.05,0.1]},
    "DecisionTree"     : {"max_depth":[6,8,10], "min_samples_split":[2,5,10]},
}

tuned_models = {}
for target in TARGET_COLS:
    best_name = best_model_name[target]
    y_tr      = y_sets[target]["train"]
    base      = MODELS[best_name]
    print(f"\n  Tuning {best_name} for {TARGET_LABELS[target]} ...")
    if best_name in PARAM_GRIDS:
        gs = GridSearchCV(base, PARAM_GRIDS[best_name], cv=3,
                          scoring="neg_mean_squared_error", n_jobs=-1, verbose=0)
        gs.fit(X_train_sc, y_tr)
        tuned_models[target] = gs.best_estimator_
        print(f"  Best params: {gs.best_params_}")
    else:
        base.fit(X_train_sc, y_tr)
        tuned_models[target] = base

# ── STEP 8: Test evaluation ───────────────────────────────────────────────────
test_results = {}
print("\n" + "="*72)
print(f"  {'Target':<30}  {'MAE':>8}  {'RMSE':>8}  {'R²':>7}  Model")
print("-"*72)
for target in TARGET_COLS:
    y_test = y_sets[target]["test"]
    y_pred = tuned_models[target].predict(X_test_sc)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2     = float(r2_score(y_test, y_pred))
    test_results[target] = {"model": best_model_name[target],
                            "MAE": round(mae,4), "RMSE": round(rmse,4), "R2": round(r2,4),
                            "y_test": y_test.tolist(), "y_pred": y_pred.tolist()}
    print(f"  {target:<30}  {mae:>8.4f}  {rmse:>8.4f}  {r2:>7.4f}  [{best_model_name[target]}]")
print("="*72)

# ── STEP 9: Actual vs Predicted ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, (target, c) in enumerate(zip(TARGET_COLS, COLORS)):
    y_t = np.array(test_results[target]["y_test"])
    y_p = np.array(test_results[target]["y_pred"])
    axes[i].scatter(y_t, y_p, alpha=0.30, s=12, color=c)
    lims = [min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())]
    axes[i].plot(lims, lims, "k--", lw=1.5, label="Perfect fit")
    axes[i].set_title(
        f"{TARGET_LABELS[target]}\nR²={test_results[target]['R2']:.3f}  RMSE={test_results[target]['RMSE']:.3f}",
        fontweight="bold", fontsize=9)
    axes[i].set_xlabel("Actual"); axes[i].set_ylabel("Predicted")
    axes[i].legend(fontsize=8)
plt.suptitle("Actual vs Predicted — Test Set", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "p3_actual_vs_predicted.png"))
plt.close()

# ── STEP 10: Residual analysis ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for i, (target, c) in enumerate(zip(TARGET_COLS, COLORS)):
    y_t = np.array(test_results[target]["y_test"])
    y_p = np.array(test_results[target]["y_pred"])
    res = y_t - y_p
    axes[0,i].scatter(y_p, res, alpha=0.25, s=10, color=c)
    axes[0,i].axhline(0, color="black", ls="--", lw=1.5)
    axes[0,i].set_title(TARGET_LABELS[target], fontweight="bold", fontsize=9)
    axes[0,i].set_xlabel("Predicted")
    if i==0: axes[0,i].set_ylabel("Residual")
    axes[1,i].hist(res, bins=40, color=c, edgecolor="white", alpha=0.85)
    axes[1,i].axvline(0, color="black", ls="--", lw=1.5)
    axes[1,i].axvline(res.mean(), color="red", ls=":", lw=1.5, label=f"Mean={res.mean():.3f}")
    axes[1,i].legend(fontsize=8)
    axes[1,i].set_xlabel("Residual")
    if i==0: axes[1,i].set_ylabel("Frequency")
plt.suptitle("Residual Analysis — Test Set", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "p3_residuals.png"))
plt.close()

# ── STEP 11: Feature importance ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, (target, c) in enumerate(zip(TARGET_COLS, COLORS)):
    model = tuned_models[target]
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True).tail(15)
        fi.plot(kind="barh", ax=axes[i], color=c, alpha=0.85, edgecolor="white")
    axes[i].set_title(f"{TARGET_LABELS[target]}  [{best_model_name[target]}]", fontweight="bold", fontsize=9)
plt.suptitle("Feature Importance — Top 15", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "p3_feature_importance.png"))
plt.close()

# ── STEP 12: Save models & metrics ───────────────────────────────────────────
for target in TARGET_COLS:
    safe = target.replace("/","_")
    joblib.dump(tuned_models[target], os.path.join(MODELS_DIR, f"model_{safe}.pkl"))

metrics_export = {t: {k:v for k,v in d.items() if k not in ("y_test","y_pred")}
                  for t,d in test_results.items()}
with open(os.path.join(OUTPUTS_DIR, "test_metrics.json"), "w") as f:
    json.dump(metrics_export, f, indent=2)
with open(os.path.join(OUTPUTS_DIR, "benchmark_results.json"), "w") as f:
    json.dump(benchmark_results, f, indent=2)

print("\n✅ Phase 3 complete. Run phase4_deployment.py next.")

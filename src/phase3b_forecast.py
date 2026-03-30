"""
phase3b_forecast.py
Phase 3b — 6-Month KPI Forecast (July → December 2024)
Dakar 4G Network — KPI Prediction Project

Run: python src/phase3b_forecast.py
"""

import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (works without a display)
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Make src/ importable regardless of where script is run from ──────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    KPI_CSV, CELL_XLSX, CLEAN_CSV, CFG_JSON,
    MODELS_DIR, OUTPUTS_DIR,
    TARGET_COLS, TARGET_LABELS, FEATURE_COLS,
    COLORS, LAYER_COLORS,
)

from sklearn.preprocessing import LabelEncoder

# ── Constants ────────────────────────────────────────────────────────────────
MONTH_NAMES    = {7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
LAST_TRAIN_MONTH = 5          # May 2024 = last complete historical month
FORECAST_START = '2024-07-01'
FORECAST_END   = '2024-12-31'

# Network feature columns used to project features forward
NETWORK_FEAT_COLS = [
    'Overload_Flag', 'LTE_PDCCH_CCE_Load_pct', 'LTE_CFI3_Share_pct',
    'LTE_CceAgg_1', 'LTE_CceAgg_2', 'LTE_CceAgg_4', 'LTE_CceAgg_8',
    'PDCCH_Adm_Rej_Count', 'LTE_Max_RRC_CU',
    'LTE_Active_UL_Users_TTI', 'SR_Fail_Ratio', 'PUCCH_SR_Cong_Count',
    'DataVol_DL_DRB_GB', 'DataVol_UL_DRB_GB',
    'LTE_PrbUtil_DL_pct', 'LTE_PrbUtil_UL_pct',
]


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load artefacts (models, scaler, metrics)
# ════════════════════════════════════════════════════════════════════════════
def load_artefacts():
    """Load trained models, scaler, and test metrics from models/ folder."""
    print("\nSTEP 1 — Loading artefacts")
    print("=" * 50)

    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    print("  Loaded: scaler.pkl")

    models = {}
    for target in TARGET_COLS:
        safe  = target.replace('/', '_')
        path  = os.path.join(MODELS_DIR, f'model_{safe}.pkl')
        models[target] = joblib.load(path)
        print(f"  Loaded: model_{safe}.pkl  [{type(models[target]).__name__}]")

    metrics_path = os.path.join(OUTPUTS_DIR, 'test_metrics.json')
    with open(metrics_path) as f:
        test_metrics = json.load(f)
    print("  Loaded: test_metrics.json")

    print("  All artefacts loaded successfully.\n")
    return scaler, models, test_metrics


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Build future feature matrix (Jul–Dec 2024)
# ════════════════════════════════════════════════════════════════════════════
def build_future_features():
    """
    Build a daily feature matrix for every cell+layer combination
    across the forecast period July 1 – December 31, 2024.

    Method:
    - Compute per-cell-layer baseline from last 60 days of historical data
    - Compute per-layer monthly growth slopes from Mar/Apr/May 2024
    - Project each network feature forward using: baseline + slope × months_ahead
    - Compute exact temporal features from future dates
    """
    print("STEP 2 — Building future feature matrix")
    print("=" * 50)

    # ── Load historical KPI data ─────────────────────────────────────────────
    df_hist = pd.read_csv(KPI_CSV)
    df_hist['Date'] = pd.to_datetime(df_hist['Date'])
    df_hist = df_hist.sort_values('Date').reset_index(drop=True)

    # ── Load cell reference file ─────────────────────────────────────────────
    df_cell   = pd.read_excel(CELL_XLSX, sheet_name='4G')
    cell_meta = df_cell[['Cell_ID','Layer','Lat','Lon','Azimuth','PCI','EARFCN','TAC']].copy()

    # ── Per-cell-layer baseline: mean of last 60 days ────────────────────────
    cutoff = df_hist['Date'].max() - pd.Timedelta(days=60)
    df_base = df_hist[df_hist['Date'] >= cutoff].copy()
    cell_layer_means = (
        df_base.groupby(['Cell_ID','Layer'])[NETWORK_FEAT_COLS]
        .mean()
        .reset_index()
    )
    print(f"  Cell-layer baselines : {len(cell_layer_means)} rows")
    print(f"  Baseline window      : {cutoff.date()} → {df_hist['Date'].max().date()}")

    # ── Per-layer monthly growth slopes (Mar/Apr/May 2024) ───────────────────
    df_hist['Month'] = df_hist['Date'].dt.month
    complete_months  = df_hist[df_hist['Month'].isin([3, 4, 5])]
    layer_growth = {}
    for layer in ['L800', 'L1800', 'L2600']:
        sub = complete_months[complete_months['Layer'] == layer]
        monthly_layer = sub.groupby('Month')[NETWORK_FEAT_COLS].mean()
        layer_growth[layer] = {}
        for col in NETWORK_FEAT_COLS:
            vals = monthly_layer[col].values
            layer_growth[layer][col] = np.polyfit([3, 4, 5], vals, 1)[0]
    print("  Per-layer growth slopes computed.")

    # ── Label encoders (fit on full historical data) ─────────────────────────
    le_layer = LabelEncoder().fit(df_hist['Layer'])
    le_cell  = LabelEncoder().fit(df_hist['Cell_ID'])
    le_bts   = LabelEncoder().fit(df_hist['BTS_ID'])

    # ── Build one row per (date × cell × layer) ──────────────────────────────
    future_dates = pd.date_range(start=FORECAST_START, end=FORECAST_END, freq='D')
    print(f"  Forecast period      : {future_dates[0].date()} → {future_dates[-1].date()}")
    print(f"  Total days           : {len(future_dates)}")
    print(f"  Total rows to build  : {len(future_dates) * len(cell_layer_means):,}")

    rows = []
    for date in future_dates:
        months_ahead = (date.month - LAST_TRAIN_MONTH) + (date.year - 2024) * 12

        for _, cell_row in cell_layer_means.iterrows():
            cell_id = cell_row['Cell_ID']
            layer   = cell_row['Layer']
            bts_id  = df_hist[df_hist['Cell_ID'] == cell_id]['BTS_ID'].iloc[0]

            # Temporal features — computed exactly from the future date
            row = {
                'Date'       : date,
                'Cell_ID'    : cell_id,
                'Layer'      : layer,
                'BTS_ID'     : bts_id,
                'Layer_enc'  : int(le_layer.transform([layer])[0]),
                'Cell_enc'   : int(le_cell.transform([cell_id])[0]),
                'BTS_enc'    : int(le_bts.transform([bts_id])[0]),
                'DayOfWeek'  : date.dayofweek,
                'DayOfMonth' : date.day,
                'Month'      : date.month,
                'WeekOfYear' : date.isocalendar()[1],
                'IsWeekend'  : int(date.dayofweek >= 5),
            }

            # Network features — projected using per-layer slopes
            for feat in NETWORK_FEAT_COLS:
                slope     = layer_growth[layer][feat]
                projected = cell_row[feat] + slope * months_ahead
                row[feat] = max(0, projected)

            rows.append(row)

    df_future = pd.DataFrame(rows)

    # Derived features (same as training)
    df_future['DL_UL_Vol_Ratio'] = (
        df_future['DataVol_DL_DRB_GB'] / (df_future['DataVol_UL_DRB_GB'] + 1e-6)
    )
    df_future['PRB_Total'] = (
        df_future['LTE_PrbUtil_DL_pct'] + df_future['LTE_PrbUtil_UL_pct']
    )

    # Merge geographic/RF metadata from cell reference file
    df_future = df_future.merge(cell_meta, on=['Cell_ID','Layer'], how='left')

    print(f"  Feature matrix shape : {df_future.shape}")
    missing = [f for f in FEATURE_COLS if f not in df_future.columns]
    if missing:
        print(f"  WARNING — missing features: {missing}")
    else:
        print(f"  All {len(FEATURE_COLS)} features present. Ready for prediction.")

    return df_future, df_hist


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Run predictions on future feature matrix
# ════════════════════════════════════════════════════════════════════════════
def run_predictions(df_future, scaler, models):
    """Apply trained models to the future feature matrix."""
    print("\nSTEP 3 — Running predictions")
    print("=" * 50)

    X_future    = df_future[FEATURE_COLS].values
    X_future_sc = scaler.transform(X_future)

    for target in TARGET_COLS:
        df_future[f'Pred_{target}'] = models[target].predict(X_future_sc)
        pred_col = df_future[f'Pred_{target}']
        print(f"  {TARGET_LABELS[target]:<28}"
              f"  mean={pred_col.mean():.3f}"
              f"  min={pred_col.min():.3f}"
              f"  max={pred_col.max():.3f}")

    print("  Predictions complete.\n")
    return df_future


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — Monthly aggregation
# ════════════════════════════════════════════════════════════════════════════
def aggregate_monthly(df_future, df_hist):
    """
    Aggregate daily predictions to monthly network-wide averages.
    Also compute historical monthly means for comparison plots.
    """
    print("STEP 4 — Monthly aggregation")
    print("=" * 50)

    df_future['Month'] = df_future['Date'].dt.month
    pred_cols   = [f'Pred_{t}' for t in TARGET_COLS]

    # Network-wide monthly forecast
    monthly_fct = df_future.groupby('Month')[pred_cols].mean().reset_index()
    monthly_fct['MonthName'] = monthly_fct['Month'].map(MONTH_NAMES)

    # Historical monthly means (complete months only)
    df_hist['Month'] = df_hist['Date'].dt.month
    hist_monthly = (
        df_hist[df_hist['Month'].isin([3, 4, 5])]
        .groupby('Month')[TARGET_COLS]
        .mean()
        .reset_index()
    )
    hist_monthly['MonthName'] = hist_monthly['Month'].map({3:'Mar', 4:'Apr', 5:'May'})

    # Per-layer monthly forecast
    layer_monthly = (
        df_future.groupby(['Month','Layer'])[pred_cols]
        .mean()
        .reset_index()
    )
    layer_monthly['MonthName'] = layer_monthly['Month'].map(MONTH_NAMES)

    print("  Monthly forecast (network average):")
    for _, row in monthly_fct.iterrows():
        vals = "  ".join(f"{TARGET_LABELS[t][:14]}={row[f'Pred_{t}']:.2f}" for t in TARGET_COLS)
        print(f"    {row['MonthName']}: {vals}")

    return monthly_fct, hist_monthly, layer_monthly


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — Plots
# ════════════════════════════════════════════════════════════════════════════
def plot_historical_vs_forecast(monthly_fct, hist_monthly, test_metrics):
    """Historical + forecast combined trend lines with ±RMSE confidence band."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    hist_month_names = ['Mar', 'Apr', 'May']
    fct_month_names  = list(MONTH_NAMES.values())

    for i, (target, c) in enumerate(zip(TARGET_COLS, COLORS)):
        ax        = axes[i]
        hist_vals = hist_monthly[target].values
        fct_vals  = monthly_fct[f'Pred_{target}'].values
        rmse      = test_metrics[target]['RMSE']

        # Historical (solid)
        ax.plot(range(len(hist_month_names)), hist_vals,
                color=c, lw=2.5, marker='o', markersize=7, label='Historical (actual)')

        # Bridge dashed line between May and Jul
        ax.plot([len(hist_month_names)-1, len(hist_month_names)],
                [hist_vals[-1], fct_vals[0]],
                color=c, lw=1.5, ls='--', alpha=0.5)

        # Forecast (dashed)
        fct_x = range(len(hist_month_names), len(hist_month_names) + len(fct_month_names))
        ax.plot(list(fct_x), fct_vals,
                color=c, lw=2.5, marker='s', markersize=7, ls='--', label='Forecast')

        # Confidence band ±1 RMSE
        ax.fill_between(list(fct_x),
                        [v - rmse for v in fct_vals],
                        [v + rmse for v in fct_vals],
                        color=c, alpha=0.12, label=f'±RMSE band')

        ax.axvline(x=len(hist_month_names) - 0.5, color='gray', ls=':', lw=1.5)
        ax.annotate(f'Dec: {fct_vals[-1]:.2f}',
                    xy=(list(fct_x)[-1], fct_vals[-1]),
                    xytext=(list(fct_x)[-1] - 0.9, fct_vals[-1] + rmse * 0.7),
                    fontsize=8, fontweight='bold', color=c,
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.2))

        all_labels = hist_month_names + fct_month_names
        ax.set_xticks(range(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=30)
        ax.set_title(TARGET_LABELS[target], fontweight='bold', fontsize=11)
        ax.set_ylabel('Mean Value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('KPI Forecast — Jul to Dec 2024  |  Dakar 4G Network',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, 'p3b_forecast_trend.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: p3b_forecast_trend.png")


def plot_forecast_by_layer(layer_monthly, df_hist):
    """Historical + forecast split by radio layer."""
    df_hist['Month'] = df_hist['Date'].dt.month
    hist_layer_monthly = (
        df_hist[df_hist['Month'].isin([3,4,5])]
        .groupby(['Month','Layer'])[TARGET_COLS]
        .mean()
        .reset_index()
    )
    hist_layer_monthly['MonthName'] = hist_layer_monthly['Month'].map(
        {3:'Mar', 4:'Apr', 5:'May'})

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, target in enumerate(TARGET_COLS):
        ax = axes[i]
        for layer, lc in LAYER_COLORS.items():
            hist_sub  = hist_layer_monthly[hist_layer_monthly['Layer']==layer].sort_values('Month')
            hist_x    = list(range(len(hist_sub)))
            hist_vals = hist_sub[target].values
            ax.plot(hist_x, hist_vals, color=lc, lw=2.2, marker='o', markersize=6, ls='-')

            fct_sub  = layer_monthly[layer_monthly['Layer']==layer].sort_values('Month')
            fct_x    = list(range(len(hist_sub), len(hist_sub)+len(fct_sub)))
            fct_vals = fct_sub[f'Pred_{target}'].values
            ax.plot([hist_x[-1], fct_x[0]], [hist_vals[-1], fct_vals[0]],
                    color=lc, lw=1.2, ls='--', alpha=0.4)
            ax.plot(fct_x, fct_vals, color=lc, lw=2.2, marker='s',
                    markersize=6, ls='--', label=layer)
            ax.annotate(f'{layer}: {fct_vals[-1]:.1f}',
                        xy=(fct_x[-1], fct_vals[-1]),
                        xytext=(fct_x[-1]+0.1, fct_vals[-1]),
                        fontsize=8, color=lc, fontweight='bold')

        ax.axvline(x=2.5, color='gray', ls=':', lw=1.5, alpha=0.6)
        all_labels = ['Mar','Apr','May'] + list(MONTH_NAMES.values())
        ax.set_xticks(range(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=30)
        ax.set_title(TARGET_LABELS[target], fontweight='bold', fontsize=11)
        ax.set_ylabel('Predicted Mean Value')
        ax.legend(title='Layer', fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.suptitle('KPI Forecast by Layer — Historical (solid) + Forecast Jul–Dec 2024 (dashed)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, 'p3b_forecast_by_layer.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: p3b_forecast_by_layer.png")


def plot_december_dashboard(layer_monthly, test_metrics):
    """December 2024 bar chart per layer with ±RMSE error bars."""
    dec_layer = layer_monthly[layer_monthly['Month'] == 12]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for i, target in enumerate(TARGET_COLS):
        pred_col = f'Pred_{target}'
        bars = axes[i].bar(
            dec_layer['Layer'], dec_layer[pred_col],
            color=[LAYER_COLORS[l] for l in dec_layer['Layer']],
            edgecolor='white', alpha=0.88, width=0.5
        )
        for bar, val in zip(bars, dec_layer[pred_col]):
            axes[i].text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()*1.01,
                         f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')

        rmse = test_metrics[target]['RMSE']
        axes[i].errorbar(dec_layer['Layer'], dec_layer[pred_col],
                         yerr=rmse, fmt='none', color='black',
                         capsize=6, linewidth=1.5)
        axes[i].set_title(TARGET_LABELS[target], fontweight='bold', fontsize=11)
        axes[i].set_ylabel('Predicted Value')
        axes[i].grid(True, axis='y', alpha=0.3)

    plt.suptitle('December 2024 Forecast per Layer  (error bars = ±RMSE)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, 'p3b_december_dashboard.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved: p3b_december_dashboard.png")


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — Print summary table
# ════════════════════════════════════════════════════════════════════════════
def print_summary(monthly_fct, hist_monthly):
    print("\nSTEP 6 — Forecast Summary")
    print("=" * 75)
    print("KPI FORECAST — Jul to Dec 2024 (Network Average, All Cells)")
    print("=" * 75)
    header = f"{'Month':<8}" + "".join(f"  {TARGET_LABELS[t][:18]:<20}" for t in TARGET_COLS)
    print(header)
    print("-" * 75)
    for _, row in monthly_fct.iterrows():
        line = f"{row['MonthName']:<8}"
        for t in TARGET_COLS:
            line += f"  {row[f'Pred_{t}']:>20.3f}"
        print(line)
    print("-" * 75)

    may_vals = hist_monthly[hist_monthly['MonthName'] == 'May'].iloc[0]
    dec_vals = monthly_fct[monthly_fct['MonthName'] == 'Dec'].iloc[0]
    print(f"\nChange May 2024 → December 2024:")
    print("-" * 55)
    for t in TARGET_COLS:
        may_v = may_vals[t]
        dec_v = dec_vals[f'Pred_{t}']
        chg   = ((dec_v - may_v) / may_v) * 100
        arrow = '↑' if chg > 0 else '↓'
        print(f"  {TARGET_LABELS[t]:<28}  May={may_v:.3f}  Dec={dec_v:.3f}  "
              f"{arrow} {abs(chg):.1f}%")
    print("=" * 75)


# ════════════════════════════════════════════════════════════════════════════
# STEP 7 — Save forecast CSVs
# ════════════════════════════════════════════════════════════════════════════
def save_outputs(df_future, monthly_fct):
    """Save daily and monthly forecast to CSV in the outputs/ folder."""
    print("\nSTEP 7 — Saving outputs")
    print("=" * 50)

    # Daily forecast (all cells × layers × days)
    forecast_cols = ['Date','Cell_ID','Layer','Month'] + [f'Pred_{t}' for t in TARGET_COLS]
    df_save = df_future[forecast_cols].copy()
    df_save['Date'] = df_save['Date'].dt.strftime('%Y-%m-%d')

    daily_path = os.path.join(OUTPUTS_DIR, 'forecast_Jul_Dec_2024_daily.csv')
    df_save.to_csv(daily_path, index=False)
    print(f"  Daily forecast   → outputs/forecast_Jul_Dec_2024_daily.csv  "
          f"({len(df_save):,} rows)")

    # Monthly summary
    monthly_path = os.path.join(OUTPUTS_DIR, 'forecast_Jul_Dec_2024_monthly.csv')
    monthly_fct.to_csv(monthly_path, index=False)
    print(f"  Monthly summary  → outputs/forecast_Jul_Dec_2024_monthly.csv")

    print("\n" + "=" * 65)
    print("PHASE 3b COMPLETE")
    print("=" * 65)
    print(f"  Forecast period : {FORECAST_START}  →  {FORECAST_END}")
    print(f"  Cells forecast  : {df_future['Cell_ID'].nunique()}")
    print(f"  Layers          : {sorted(df_future['Layer'].unique().tolist())}")
    print(f"  Total rows      : {len(df_save):,}  (cell × layer × day)")
    print("=" * 65)


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC API — callable from streamlit_app.py
# ════════════════════════════════════════════════════════════════════════════
def run_forecast():
    """
    Run the full forecast pipeline and return results.

    Returns
    -------
    dict with keys:
        'df_future'    : pd.DataFrame — daily predictions per cell+layer
        'monthly_fct'  : pd.DataFrame — monthly network-wide averages
        'layer_monthly': pd.DataFrame — monthly averages per layer
        'hist_monthly' : pd.DataFrame — historical monthly means (Mar/Apr/May)
        'test_metrics' : dict         — model evaluation metrics
    """
    scaler, models, test_metrics            = load_artefacts()
    df_future, df_hist                      = build_future_features()
    df_future                               = run_predictions(df_future, scaler, models)
    monthly_fct, hist_monthly, layer_monthly = aggregate_monthly(df_future, df_hist)

    print("\nSTEP 5 — Generating plots")
    print("=" * 50)
    plot_historical_vs_forecast(monthly_fct, hist_monthly, test_metrics)
    plot_forecast_by_layer(layer_monthly, df_hist)
    plot_december_dashboard(layer_monthly, test_metrics)

    print_summary(monthly_fct, hist_monthly)
    save_outputs(df_future, monthly_fct)

    return {
        'df_future'    : df_future,
        'monthly_fct'  : monthly_fct,
        'layer_monthly': layer_monthly,
        'hist_monthly' : hist_monthly,
        'test_metrics' : test_metrics,
    }


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    run_forecast()

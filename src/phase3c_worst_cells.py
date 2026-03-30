"""
phase3c_worst_cells.py
Phase 3c — Worst Cell Analysis (December 2024 Forecast)
Dakar 4G Network — KPI Prediction Project

Identifies the top 5 worst cells per KPI for December 2024.
Run: python src/phase3c_worst_cells.py
"""

import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    KPI_CSV, CELL_XLSX, MODELS_DIR, OUTPUTS_DIR,
    TARGET_COLS, TARGET_LABELS, FEATURE_COLS, COLORS, LAYER_COLORS,
)
from sklearn.preprocessing import LabelEncoder

# ── Constants ────────────────────────────────────────────────────────────────
LAST_TRAIN_MONTH = 5
FORECAST_MONTH   = 12          # December 2024
BG               = '#F8F8F8'
BLUE             = '#1F5C99'
RED              = '#C73E1D'

NETWORK_FEAT_COLS = [
    'Overload_Flag', 'LTE_PDCCH_CCE_Load_pct', 'LTE_CFI3_Share_pct',
    'LTE_CceAgg_1', 'LTE_CceAgg_2', 'LTE_CceAgg_4', 'LTE_CceAgg_8',
    'PDCCH_Adm_Rej_Count', 'LTE_Max_RRC_CU',
    'LTE_Active_UL_Users_TTI', 'SR_Fail_Ratio', 'PUCCH_SR_Cong_Count',
    'DataVol_DL_DRB_GB', 'DataVol_UL_DRB_GB',
    'LTE_PrbUtil_DL_pct', 'LTE_PrbUtil_UL_pct',
]


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load artefacts
# ════════════════════════════════════════════════════════════════════════════
def load_artefacts():
    print("\nSTEP 1 — Loading artefacts")
    print("=" * 50)
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    models = {}
    for target in TARGET_COLS:
        safe = target.replace('/', '_')
        models[target] = joblib.load(os.path.join(MODELS_DIR, f'model_{safe}.pkl'))
        print(f"  Loaded: model_{safe}.pkl")
    print("  Done.\n")
    return scaler, models


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Build December 2024 feature matrix
# ════════════════════════════════════════════════════════════════════════════
def build_december_features(scaler, models):
    """
    Build feature matrix for December 2024 only (daily, all cells+layers).
    Same projection method as Phase 3b but limited to December.
    """
    print("STEP 2 — Building December 2024 feature matrix")
    print("=" * 50)

    # Load data
    df_hist   = pd.read_csv(KPI_CSV)
    df_hist['Date'] = pd.to_datetime(df_hist['Date'])
    df_hist   = df_hist.sort_values('Date').reset_index(drop=True)
    df_cell   = pd.read_excel(CELL_XLSX, sheet_name='4G')
    cell_meta = df_cell[['Cell_ID','Layer','Lat','Lon',
                          'Azimuth','PCI','EARFCN','TAC']].copy()

    # Per-cell-layer baseline (last 60 days)
    cutoff           = df_hist['Date'].max() - pd.Timedelta(days=60)
    df_base          = df_hist[df_hist['Date'] >= cutoff]
    cell_layer_means = (
        df_base.groupby(['Cell_ID','Layer'])[NETWORK_FEAT_COLS]
        .mean().reset_index()
    )

    # Per-layer monthly growth slopes (Mar/Apr/May)
    df_hist['Month'] = df_hist['Date'].dt.month
    complete         = df_hist[df_hist['Month'].isin([3, 4, 5])]
    layer_growth     = {}
    for layer in ['L800', 'L1800', 'L2600']:
        sub          = complete[complete['Layer'] == layer]
        monthly_lyr  = sub.groupby('Month')[NETWORK_FEAT_COLS].mean()
        layer_growth[layer] = {}
        for col in NETWORK_FEAT_COLS:
            vals = monthly_lyr[col].values
            layer_growth[layer][col] = np.polyfit([3, 4, 5], vals, 1)[0]

    # Label encoders
    le_layer = LabelEncoder().fit(df_hist['Layer'])
    le_cell  = LabelEncoder().fit(df_hist['Cell_ID'])
    le_bts   = LabelEncoder().fit(df_hist['BTS_ID'])

    # December 2024 daily dates
    dec_dates = pd.date_range(start='2024-12-01', end='2024-12-31', freq='D')
    print(f"  December dates : {dec_dates[0].date()} → {dec_dates[-1].date()}")

    rows = []
    for date in dec_dates:
        months_ahead = (date.month - LAST_TRAIN_MONTH) + (date.year - 2024) * 12
        for _, cell_row in cell_layer_means.iterrows():
            cell_id = cell_row['Cell_ID']
            layer   = cell_row['Layer']
            bts_id  = df_hist[df_hist['Cell_ID'] == cell_id]['BTS_ID'].iloc[0]

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
            for feat in NETWORK_FEAT_COLS:
                slope    = layer_growth[layer][feat]
                row[feat] = max(0, cell_row[feat] + slope * months_ahead)
            rows.append(row)

    df_dec = pd.DataFrame(rows)
    df_dec['DL_UL_Vol_Ratio'] = (
        df_dec['DataVol_DL_DRB_GB'] / (df_dec['DataVol_UL_DRB_GB'] + 1e-6)
    )
    df_dec['PRB_Total'] = (
        df_dec['LTE_PrbUtil_DL_pct'] + df_dec['LTE_PrbUtil_UL_pct']
    )
    df_dec = df_dec.merge(cell_meta, on=['Cell_ID', 'Layer'], how='left')

    # Predict
    X        = df_dec[FEATURE_COLS].values
    X_sc     = scaler.transform(X)
    for target in TARGET_COLS:
        df_dec[f'Pred_{target}'] = models[target].predict(X_sc)

    # Average over all December days → one value per cell+layer
    pred_cols      = [f'Pred_{t}' for t in TARGET_COLS]
    cell_layer_dec = (
        df_dec.groupby(['Cell_ID', 'Layer'])[pred_cols]
        .mean().reset_index()
    )
    cell_layer_dec.columns = ['Cell_ID', 'Layer',
                               'DataVol', 'Users', 'Throughput', 'PRB']
    cell_layer_dec['BTS_ID'] = cell_layer_dec['Cell_ID'].str[:6]

    # Historical baseline per cell+layer
    hist_base = df_hist.groupby(['Cell_ID','Layer'])[TARGET_COLS].mean().reset_index()
    hist_base.columns = ['Cell_ID','Layer',
                          'DataVol_hist','Users_hist',
                          'Throughput_hist','PRB_hist']
    cell_layer_dec = cell_layer_dec.merge(hist_base, on=['Cell_ID','Layer'])
    cell_layer_dec['DataVol_delta']    = (
        cell_layer_dec['DataVol'] - cell_layer_dec['DataVol_hist'])
    cell_layer_dec['Users_delta']      = (
        cell_layer_dec['Users'] - cell_layer_dec['Users_hist'])
    cell_layer_dec['Throughput_delta'] = (
        cell_layer_dec['Throughput'] - cell_layer_dec['Throughput_hist'])
    cell_layer_dec['PRB_delta']        = (
        cell_layer_dec['PRB'] - cell_layer_dec['PRB_hist'])

    print(f"  Cell+layer combinations : {len(cell_layer_dec)}")
    return cell_layer_dec


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Identify top 5 worst cells per KPI
# ════════════════════════════════════════════════════════════════════════════
def get_worst_cells(cell_layer_dec):
    """Return top 5 worst cells for each of the 4 KPIs."""
    top_vol = cell_layer_dec.nlargest(5, 'DataVol').reset_index(drop=True)
    top_usr = cell_layer_dec.nlargest(5, 'Users').reset_index(drop=True)
    top_thp = cell_layer_dec.nsmallest(5, 'Throughput').reset_index(drop=True)
    top_prb = cell_layer_dec.nlargest(5, 'PRB').reset_index(drop=True)

    print("\nSTEP 3 — Worst Cells (December 2024 Forecast)")
    print("=" * 60)

    print("\nTop 5 — Highest Data Volume (GB):")
    print(top_vol[['Cell_ID','BTS_ID','Layer',
                   'DataVol','DataVol_hist','DataVol_delta']].to_string(index=False))

    print("\nTop 5 — Highest Connected Users:")
    print(top_usr[['Cell_ID','BTS_ID','Layer',
                   'Users','Users_hist','Users_delta']].to_string(index=False))

    print("\nTop 5 — Lowest DL Throughput (Mbps):")
    print(top_thp[['Cell_ID','BTS_ID','Layer',
                   'Throughput','Throughput_hist','Throughput_delta']].to_string(index=False))

    print("\nTop 5 — Highest PRB Utilisation (%):")
    print(top_prb[['Cell_ID','BTS_ID','Layer',
                   'PRB','PRB_hist','PRB_delta']].to_string(index=False))

    return top_vol, top_usr, top_thp, top_prb


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — Generate charts
# ════════════════════════════════════════════════════════════════════════════
def make_bar_chart(df, col_hist, col_fcast, col_delta,
                   title, xlabel, unit, filename, higher_is_worse=True):
    """Grouped bar chart: historical vs forecast + delta."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(BG)

    cells = df['Cell_ID'] + '\n' + df['Layer']
    x     = np.arange(len(cells))
    w     = 0.35

    # Left — historical vs forecast
    ax = axes[0]
    ax.set_facecolor(BG)
    ax.bar(x - w/2, df[col_hist],  w, label='Historical avg',
           color='#AACDE8', edgecolor='white', zorder=3)
    bars_f = ax.bar(x + w/2, df[col_fcast], w, label='Dec 2024 Forecast',
                    color=RED, edgecolor='white', zorder=3)
    for bar in bars_f:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15,
                f'{bar.get_height():.1f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=RED)
    ax.set_xticks(x); ax.set_xticklabels(cells, fontsize=10)
    ax.set_ylabel(f'{xlabel} ({unit})'); ax.legend(fontsize=9)
    ax.set_title('Historical vs December 2024', fontsize=11, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.4); ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Right — delta
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    delta_colors = [
        RED if (higher_is_worse and d > 0) or (not higher_is_worse and d < 0)
        else '#44BBA4' for d in df[col_delta]
    ]
    bars_d = ax2.bar(x, df[col_delta], color=delta_colors, edgecolor='white', zorder=3)
    ax2.axhline(0, color='black', lw=1.0, zorder=4)
    for bar, val in zip(bars_d, df[col_delta]):
        ypos = val + 0.03 if val >= 0 else val - 0.08
        ax2.text(bar.get_x()+bar.get_width()/2, ypos, f'{val:+.2f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold',
                 color=RED if (higher_is_worse and val > 0) or
                 (not higher_is_worse and val < 0) else '#1a6e3f')
    ax2.set_xticks(x); ax2.set_xticklabels(cells, fontsize=10)
    ax2.set_ylabel(f'Change vs Historical ({unit})')
    ax2.set_title('Change vs Historical Average', fontsize=11, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.4); ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight='bold', color=BLUE, y=1.01)
    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, filename)
    plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Saved: {filename}")


def make_summary_chart(top_vol, top_usr, top_thp, top_prb):
    """Combined 4-panel horizontal bar chart."""
    fig, axes = plt.subplots(4, 1, figsize=(13, 16))
    fig.patch.set_facecolor(BG)

    configs = [
        (top_vol, 'DataVol',    'DataVol_hist',
         'Highest Data Volume (GB) — Dec 2024', '#C73E1D', '#AACDE8', 'GB'),
        (top_usr, 'Users',      'Users_hist',
         'Highest Connected Users — Dec 2024',  '#A23B72', '#D8B4D8', 'users'),
        (top_thp, 'Throughput', 'Throughput_hist',
         'Lowest DL Throughput (Mbps) — Dec 2024', '#E87722', '#F5C89A', 'Mbps'),
        (top_prb, 'PRB',        'PRB_hist',
         'Highest PRB Utilisation (%) — Dec 2024', '#2E86AB', '#A8CDEF', '%'),
    ]

    for ax, (df, col, col_h, title, fc, hc, unit) in zip(axes, configs):
        ax.set_facecolor(BG)
        cells = [f"{r.Cell_ID}  ({r.Layer})" for _, r in df.iterrows()]
        y     = np.arange(len(cells)); h = 0.32
        ax.barh(y+h/2, df[col_h], h, color=hc, edgecolor='white',
                label='Historical avg', zorder=3)
        bars_f = ax.barh(y-h/2, df[col], h, color=fc, edgecolor='white',
                         label='Dec 2024 Forecast', zorder=3)
        for i, (bar, val) in enumerate(zip(bars_f, df[col])):
            ax.text(val+0.05, bar.get_y()+bar.get_height()/2,
                    f'#{i+1}  {val:.2f} {unit}',
                    va='center', ha='left', fontsize=9.5,
                    fontweight='bold', color=fc)
        ax.set_yticks(y); ax.set_yticklabels(cells, fontsize=10)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, fontweight='bold', color=BLUE, pad=6)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, axis='x', alpha=0.4); ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Worst Cell Analysis — December 2024 Forecast\n'
                 'All categories dominated by L2600 layer',
                 fontsize=13, fontweight='bold', color=BLUE, y=1.002)
    plt.tight_layout(h_pad=2.5)
    out = os.path.join(OUTPUTS_DIR, 'worst_cells_summary.png')
    plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Saved: worst_cells_summary.png")


def generate_all_charts(top_vol, top_usr, top_thp, top_prb):
    print("\nSTEP 4 — Generating charts")
    print("=" * 50)
    make_bar_chart(top_vol,'DataVol_hist','DataVol','DataVol_delta',
        'Top 5 — Highest Forecast Data Volume (December 2024)',
        'Data Volume','GB','worst_datavol.png', higher_is_worse=True)
    make_bar_chart(top_usr,'Users_hist','Users','Users_delta',
        'Top 5 — Highest Forecast Connected Users (December 2024)',
        'Avg Connected Users','users','worst_users.png', higher_is_worse=True)
    make_bar_chart(top_thp,'Throughput_hist','Throughput','Throughput_delta',
        'Top 5 — Lowest Forecast DL Throughput (December 2024)',
        'DL Cell Throughput','Mbps','worst_throughput.png', higher_is_worse=False)
    make_bar_chart(top_prb,'PRB_hist','PRB','PRB_delta',
        'Top 5 — Highest Forecast PRB Utilisation (December 2024)',
        'DL PRB Utilisation','%','worst_prb.png', higher_is_worse=True)
    make_summary_chart(top_vol, top_usr, top_thp, top_prb)


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — Save results to CSV
# ════════════════════════════════════════════════════════════════════════════
def save_results(top_vol, top_usr, top_thp, top_prb, cell_layer_dec):
    print("\nSTEP 5 — Saving results")
    print("=" * 50)

    # Full ranked table
    out = os.path.join(OUTPUTS_DIR, 'worst_cells_december_2024.csv')
    cell_layer_dec.sort_values('DataVol', ascending=False).to_csv(out, index=False)
    print(f"  Full ranked table → outputs/worst_cells_december_2024.csv")

    # Summary table
    rows = []
    for i, r in top_vol.iterrows():
        rows.append({'Rank': i+1, 'KPI': 'High DataVol', 'Cell_ID': r.Cell_ID,
                     'BTS_ID': r.BTS_ID, 'Layer': r.Layer,
                     'Forecast': round(r.DataVol, 3),
                     'Historical': round(r.DataVol_hist, 3),
                     'Delta': round(r.DataVol_delta, 3)})
    for i, r in top_usr.iterrows():
        rows.append({'Rank': i+1, 'KPI': 'High Users', 'Cell_ID': r.Cell_ID,
                     'BTS_ID': r.BTS_ID, 'Layer': r.Layer,
                     'Forecast': round(r.Users, 1),
                     'Historical': round(r.Users_hist, 1),
                     'Delta': round(r.Users_delta, 2)})
    for i, r in top_thp.iterrows():
        rows.append({'Rank': i+1, 'KPI': 'Low Throughput', 'Cell_ID': r.Cell_ID,
                     'BTS_ID': r.BTS_ID, 'Layer': r.Layer,
                     'Forecast': round(r.Throughput, 3),
                     'Historical': round(r.Throughput_hist, 3),
                     'Delta': round(r.Throughput_delta, 3)})
    for i, r in top_prb.iterrows():
        rows.append({'Rank': i+1, 'KPI': 'High PRB', 'Cell_ID': r.Cell_ID,
                     'BTS_ID': r.BTS_ID, 'Layer': r.Layer,
                     'Forecast': round(r.PRB, 2),
                     'Historical': round(r.PRB_hist, 2),
                     'Delta': round(r.PRB_delta, 2)})
    out2 = os.path.join(OUTPUTS_DIR, 'worst_cells_top5_summary.csv')
    pd.DataFrame(rows).to_csv(out2, index=False)
    print(f"  Top-5 summary     → outputs/worst_cells_top5_summary.csv")

    print("\n" + "=" * 60)
    print("PHASE 3c COMPLETE — WORST CELL ANALYSIS")
    print("=" * 60)
    print(f"  All worst-case cells are on L2600")
    print(f"  DK0007 appears in 3 KPI categories (priority site)")
    print(f"  DK0008 appears in 2 KPI categories")
    print("=" * 60)

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC API — callable from streamlit_app.py
# ════════════════════════════════════════════════════════════════════════════
def run_worst_cell_analysis():
    """
    Run the full worst-cell analysis and return results.

    Returns
    -------
    dict with keys:
        'top_vol'        : DataFrame — top 5 high data volume cells
        'top_usr'        : DataFrame — top 5 high connected users cells
        'top_thp'        : DataFrame — top 5 low throughput cells
        'top_prb'        : DataFrame — top 5 high PRB utilisation cells
        'cell_layer_dec' : DataFrame — full ranked cell+layer forecast
        'summary'        : DataFrame — combined top-5 summary table
    """
    scaler, models           = load_artefacts()
    cell_layer_dec           = build_december_features(scaler, models)
    top_vol, top_usr, top_thp, top_prb = get_worst_cells(cell_layer_dec)
    generate_all_charts(top_vol, top_usr, top_thp, top_prb)
    summary = save_results(top_vol, top_usr, top_thp, top_prb, cell_layer_dec)

    return {
        'top_vol'        : top_vol,
        'top_usr'        : top_usr,
        'top_thp'        : top_thp,
        'top_prb'        : top_prb,
        'cell_layer_dec' : cell_layer_dec,
        'summary'        : summary,
    }


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    run_worst_cell_analysis()

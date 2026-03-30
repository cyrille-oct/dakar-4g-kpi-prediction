"""
app/streamlit_app.py
Dakar 4G KPI Prediction — Interactive Deployment Dashboard
Run: streamlit run app/streamlit_app.py
"""
import os, sys, json, joblib, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dakar 4G KPI Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

CFG_JSON  = os.path.join(DATA_DIR,    "column_config.json")
CLEAN_CSV = os.path.join(DATA_DIR,    "clean_dataset.csv")
METRICS_J = os.path.join(OUTPUTS_DIR, "test_metrics.json")
BENCH_J   = os.path.join(OUTPUTS_DIR, "benchmark_results.json")
KPI_CSV   = os.path.join(DATA_DIR,    "KPI_Dataset_Dakar_4G.csv")

TARGET_LABELS = {
    "Total_DataVol_GB"     : "Data Volume (GB)",
    "LTE_Avg_RRC_CU"       : "Avg Connected Users",
    "LTE_Cell_THP_DL_Mbps" : "DL Cell Throughput (Mbps)",
    "LTE_PrbUtil_DL_pct"   : "DL PRB Utilisation (%)",
}
COLORS       = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
LAYER_COLORS = {"L800": "#3F88C5", "L1800": "#F4720B", "L2600": "#44BBA4"}
MONTH_NAMES  = {7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}

# ── Load artefacts (cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    with open(CFG_JSON) as f:
        cfg = json.load(f)
    feature_cols = cfg["FEATURE_COLS"]
    target_cols  = cfg["TARGET_COLS"]

    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    models = {}
    for t in target_cols:
        safe = t.replace("/","_")
        models[t] = joblib.load(os.path.join(MODELS_DIR, f"model_{safe}.pkl"))

    with open(METRICS_J) as f:
        test_metrics = json.load(f)
    with open(BENCH_J) as f:
        benchmark = json.load(f)

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

    available    = [f for f in feature_cols if f in df.columns]
    df           = df[available + target_cols].dropna().reset_index(drop=True)
    feature_cols = available

    split_idx = int(0.80 * len(df))
    df_test   = df.iloc[split_idx:].reset_index(drop=True)
    X_test_sc = scaler.transform(df_test[feature_cols].values)

    return scaler, models, test_metrics, benchmark, df, df_test, X_test_sc, feature_cols, target_cols


# ── Load forecast data (cached, runs phase3b on demand) ──────────────────────
@st.cache_data
def load_forecast():
    """
    Load pre-computed forecast CSVs if they exist.
    If not, run phase3b_forecast.py to generate them.
    Returns monthly_fct, layer_monthly, hist_monthly, test_metrics.
    """
    monthly_path = os.path.join(OUTPUTS_DIR, "forecast_Jul_Dec_2024_monthly.csv")
    daily_path   = os.path.join(OUTPUTS_DIR, "forecast_Jul_Dec_2024_daily.csv")

    # Run forecast if CSVs don't exist yet
    if not os.path.exists(monthly_path) or not os.path.exists(daily_path):
        from phase3b_forecast import run_forecast
        results = run_forecast()
        return (
            results["monthly_fct"],
            results["layer_monthly"],
            results["hist_monthly"],
            results["test_metrics"],
        )

    # Load from saved CSVs
    monthly_fct   = pd.read_csv(monthly_path)
    daily_fct     = pd.read_csv(daily_path)
    daily_fct["Date"] = pd.to_datetime(daily_fct["Date"])

    # Re-derive layer_monthly from daily forecast
    daily_fct["Month"] = daily_fct["Date"].dt.month
    pred_cols = [c for c in daily_fct.columns if c.startswith("Pred_")]
    layer_monthly = daily_fct.groupby(["Month","Layer"])[pred_cols].mean().reset_index()
    layer_monthly["MonthName"] = layer_monthly["Month"].map(MONTH_NAMES)

    # Historical monthly means from KPI CSV
    df_hist = pd.read_csv(KPI_CSV)
    df_hist["Date"]  = pd.to_datetime(df_hist["Date"])
    df_hist["Month"] = df_hist["Date"].dt.month
    target_cols = ["Total_DataVol_GB","LTE_Avg_RRC_CU","LTE_Cell_THP_DL_Mbps","LTE_PrbUtil_DL_pct"]
    hist_monthly = (
        df_hist[df_hist["Month"].isin([3,4,5])]
        .groupby("Month")[target_cols].mean().reset_index()
    )
    hist_monthly["MonthName"] = hist_monthly["Month"].map({3:"Mar", 4:"Apr", 5:"May"})

    with open(METRICS_J) as f:
        test_metrics = json.load(f)

    return monthly_fct, layer_monthly, hist_monthly, test_metrics


try:
    scaler, models, test_metrics, benchmark, df, df_test, X_test_sc, FEATURE_COLS, TARGET_COLS = load_artefacts()
    data_loaded = True
except Exception as e:
    data_loaded = False
    load_error  = str(e)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f2/Senegalese_flag.svg/320px-Senegalese_flag.svg.png", width=60)
st.sidebar.title("📡 Dakar 4G KPI")
st.sidebar.markdown("**Network:** Dakar Dense Urban  \n**Layers:** L800 / L1800 / L2600  \n**Period:** Mar–Jun 2024")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📊 Model Performance",
    "🔮 Single Prediction",
    "📈 Monitoring Dashboard",
    "🔍 Benchmark Comparison",
    "📅 6-Month Forecast",
])

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📡 Dakar 4G Network — KPI Prediction Dashboard")
st.markdown("Regression-based prediction of **Data Volume · Connected Users · Throughput · PRB Utilisation**")

if not data_loaded:
    st.error(f"⚠️ Could not load artefacts: {load_error}")
    st.stop()

# ── PAGE: Overview ────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.header("Project Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Cells", "30")
    col2.metric("BTS Sites", "10")
    col3.metric("Observation Period", "Mar–Jun 2024")
    col4.metric("Training Records", f"{int(0.8 * len(df)):,}")

    st.markdown("---")
    st.subheader("Final Model Performance")
    rows = []
    for t in TARGET_COLS:
        m = test_metrics[t]
        rows.append({"Target": TARGET_LABELS[t], "Model": m["model"],
                     "MAE": m["MAE"], "RMSE": m["RMSE"], "R²": m["R2"]})
    df_metrics = pd.DataFrame(rows)
    st.dataframe(df_metrics.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}),
                 use_container_width=True)

    st.markdown("---")
    st.subheader("Dataset Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Target variable statistics:**")
        st.dataframe(df[TARGET_COLS].describe().round(3))
    with col2:
        st.write("**Records per layer:**")
        if os.path.exists(KPI_CSV):
            df_raw = pd.read_csv(KPI_CSV)
            st.dataframe(df_raw["Layer"].value_counts().reset_index().rename(
                columns={"index":"Layer","Layer":"Count"}))

# ── PAGE: Model Performance ───────────────────────────────────────────────────
elif page == "📊 Model Performance":
    st.header("Model Performance — Test Set")

    target_sel = st.selectbox("Select target:", TARGET_COLS,
                              format_func=lambda x: TARGET_LABELS[x])

    y_t       = df_test[target_sel].values
    y_p       = models[target_sel].predict(X_test_sc)
    residuals = y_t - y_p
    c         = COLORS[TARGET_COLS.index(target_sel)]

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE",  f"{mean_absolute_error(y_t, y_p):.4f}")
    col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_t, y_p)):.4f}")
    col3.metric("R²",   f"{r2_score(y_t, y_p):.4f}")

    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_t, y_p, alpha=0.30, s=12, color=c)
        lims = [min(y_t.min(),y_p.min()), max(y_t.max(),y_p.max())]
        ax.plot(lims, lims, "k--", lw=1.5, label="Perfect fit")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted", fontweight="bold")
        ax.legend(); st.pyplot(fig)

    with col_b:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].scatter(y_p, residuals, alpha=0.25, s=10, color=c)
        axes[0].axhline(0, color="black", ls="--", lw=1.5)
        axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")
        axes[0].set_title("Residuals vs Predicted", fontweight="bold")
        axes[1].hist(residuals, bins=40, color=c, edgecolor="white", alpha=0.85)
        axes[1].axvline(residuals.mean(), color="red", ls=":", lw=1.5,
                        label=f"Mean={residuals.mean():.3f}")
        axes[1].set_xlabel("Residual")
        axes[1].set_title("Residual Distribution", fontweight="bold")
        axes[1].legend(fontsize=8); plt.tight_layout(); st.pyplot(fig)

    model = models[target_sel]
    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importance — Top 15")
        fi = pd.Series(model.feature_importances_, index=FEATURE_COLS)
        fi = fi.sort_values(ascending=False).head(15).reset_index()
        fi.columns = ["Feature", "Importance"]
        st.bar_chart(fi.set_index("Feature"))

# ── PAGE: Single Prediction ───────────────────────────────────────────────────
elif page == "🔮 Single Prediction":
    st.header("Single Cell KPI Prediction")
    st.markdown("Configure cell parameters and get an instant 4-KPI prediction.")

    col1, col2, col3 = st.columns(3)
    with col1:
        layer       = st.selectbox("Layer", ["L800", "L1800", "L2600"])
        layer_enc   = {"L800": 1, "L1800": 0, "L2600": 2}.get(layer, 1)
        day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 1)
        month       = st.slider("Month", 1, 12, 7)
        is_weekend  = 1 if day_of_week >= 5 else 0
    with col2:
        pdcch_load = st.slider("PDCCH CCE Load (%)", 0.0, 100.0, 30.0)
        max_rrc    = st.slider("Max RRC Users", 50.0, 300.0, 130.0)
        prb_dl     = st.slider("PRB DL Utilisation (%)", 10.0, 100.0, 52.0)
        prb_ul     = st.slider("PRB UL Utilisation (%)", 5.0, 60.0, 25.0)
    with col3:
        sr_fail  = st.slider("SR Fail Ratio", 0.0, 0.1, 0.012, step=0.001)
        cfi3     = st.slider("CFI3 Share (%)", 0.0, 100.0, 55.0)
        overload = st.selectbox("Overload Flag", [0, 1])

    if st.button("🔮 Predict KPIs", type="primary"):
        sample_defaults = dict(zip(FEATURE_COLS, df_test[FEATURE_COLS].values[0]))
        sample_defaults.update({
            "Layer_enc": layer_enc, "DayOfWeek": day_of_week,
            "Month": month, "IsWeekend": is_weekend,
            "DayOfMonth": 15, "WeekOfYear": 28,
            "LTE_PDCCH_CCE_Load_pct": pdcch_load,
            "LTE_Max_RRC_CU": max_rrc,
            "LTE_PrbUtil_DL_pct": prb_dl,
            "LTE_PrbUtil_UL_pct": prb_ul,
            "SR_Fail_Ratio": sr_fail,
            "LTE_CFI3_Share_pct": cfi3,
            "Overload_Flag": overload,
            "PRB_Total": prb_dl + prb_ul,
        })
        x    = np.array([[sample_defaults.get(f, 0) for f in FEATURE_COLS]])
        x_sc = scaler.transform(x)
        st.markdown("---")
        st.subheader("Predictions")
        cols = st.columns(4)
        for j, t in enumerate(TARGET_COLS):
            cols[j].metric(TARGET_LABELS[t], f"{float(models[t].predict(x_sc)[0]):.3f}")

# ── PAGE: Monitoring Dashboard ────────────────────────────────────────────────
elif page == "📈 Monitoring Dashboard":
    st.header("Monitoring Dashboard")
    WINDOW    = st.sidebar.slider("Rolling window size", 10, 100, 50)
    ALERT_PCT = st.sidebar.slider("Alert threshold (%)", 5, 50, 20) / 100

    df_preds = df_test[TARGET_COLS].copy()
    for target in TARGET_COLS:
        df_preds[f"Pred_{target}"] = models[target].predict(X_test_sc)
        df_preds[f"Err_{target}"]  = df_preds[target] - df_preds[f"Pred_{target}"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    for i, (target, c) in enumerate(zip(TARGET_COLS, COLORS)):
        err       = df_preds[f"Err_{target}"].values
        r_rmse    = pd.Series(err**2).rolling(WINDOW, min_periods=5).mean().apply(np.sqrt)
        glob_rmse = float(np.sqrt(np.mean(err**2)))
        alert_lvl = glob_rmse * (1 + ALERT_PCT)
        axes[i].plot(r_rmse.index, r_rmse.values, color=c, lw=1.8, label=f"Rolling RMSE (w={WINDOW})")
        axes[i].axhline(glob_rmse, color="black", ls="--", lw=1.2, label=f"Global = {glob_rmse:.3f}")
        axes[i].axhline(alert_lvl, color="red", ls=":", lw=1.5, label=f"Alert (+{int(ALERT_PCT*100)}%)")
        axes[i].fill_between(r_rmse.index, glob_rmse, alert_lvl, alpha=0.08, color="orange")
        alert_pts = r_rmse[r_rmse > alert_lvl]
        if len(alert_pts):
            axes[i].scatter(alert_pts.index, alert_pts.values, color="red", s=20, zorder=5)
        axes[i].set_title(TARGET_LABELS[target], fontweight="bold")
        axes[i].set_xlabel("Test sample index"); axes[i].set_ylabel("RMSE")
        axes[i].legend(fontsize=7, ncol=2); axes[i].grid(True, alpha=0.3)
    plt.suptitle("Rolling RMSE Monitoring with Alert Thresholds", fontsize=13, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig)

    st.subheader("Error Percentile Summary")
    rows = []
    for target in TARGET_COLS:
        abserr = df_preds[f"Err_{target}"].abs().values
        p50, p75, p90, p95, p99 = np.percentile(abserr, [50, 75, 90, 95, 99])
        rows.append({"Target": TARGET_LABELS[target],
                     "P50": round(p50,4), "P75": round(p75,4),
                     "P90": round(p90,4), "P95": round(p95,4), "P99": round(p99,4)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ── PAGE: Benchmark Comparison ────────────────────────────────────────────────
elif page == "🔍 Benchmark Comparison":
    st.header("Model Benchmark — 5-Fold Cross-Validation")
    target_sel = st.selectbox("Select target:", TARGET_COLS,
                              format_func=lambda x: TARGET_LABELS[x])
    bm   = benchmark[target_sel]
    rows = [{"Model": m, "CV RMSE": v["RMSE"], "CV R²": v["R2"]} for m, v in bm.items()]
    df_bm = pd.DataFrame(rows).sort_values("CV RMSE").reset_index(drop=True)
    df_bm.index += 1
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Ranked Table")
        st.dataframe(df_bm.style.format({"CV RMSE": "{:.4f}", "CV R²": "{:.4f}"}),
                     use_container_width=True)
    with col2:
        st.subheader("RMSE Comparison")
        st.bar_chart(df_bm.set_index("Model")[["CV RMSE"]])
    st.info(f"✅ **Best model:** {df_bm.iloc[0]['Model']}  "
            f"(RMSE = {df_bm.iloc[0]['CV RMSE']:.4f}  |  R² = {df_bm.iloc[0]['CV R²']:.4f})")

# ── PAGE: 6-Month Forecast ────────────────────────────────────────────────────
elif page == "📅 6-Month Forecast":
    st.header("📅 6-Month KPI Forecast — July to December 2024")
    st.markdown("Forecasts are generated by projecting per-layer growth trends "
                "from March–May 2024 forward using the trained models.")

    with st.spinner("Loading forecast data..."):
        try:
            monthly_fct, layer_monthly, hist_monthly, fct_metrics = load_forecast()
            forecast_loaded = True
        except Exception as e:
            forecast_loaded = False
            st.error(f"Forecast error: {e}")

    if not forecast_loaded:
        st.stop()

    pred_cols = [f"Pred_{t}" for t in TARGET_COLS]

    # ── Summary metrics ────────────────────────────────────────────────────────
    st.subheader("Network-Wide Monthly Forecast")
    may_row = hist_monthly[hist_monthly["MonthName"] == "May"].iloc[0]
    dec_row = monthly_fct[monthly_fct["MonthName"] == "Dec"].iloc[0]
    cols = st.columns(4)
    for j, t in enumerate(TARGET_COLS):
        may_v  = may_row[t]
        dec_v  = dec_row[f"Pred_{t}"]
        delta  = ((dec_v - may_v) / may_v) * 100
        cols[j].metric(
            label=TARGET_LABELS[t],
            value=f"{dec_v:.2f}",
            delta=f"{delta:+.1f}% vs May 2024"
        )

    st.markdown("---")

    # ── Monthly forecast table ─────────────────────────────────────────────────
    st.subheader("Monthly Forecast Table (All Cells Average)")
    disp = monthly_fct[["MonthName"] + pred_cols].copy()
    disp.columns = ["Month"] + [TARGET_LABELS[t] for t in TARGET_COLS]
    st.dataframe(disp.style.format({TARGET_LABELS[t]: "{:.3f}" for t in TARGET_COLS}),
                 use_container_width=True)

    st.markdown("---")

    # ── Trend chart: historical + forecast ────────────────────────────────────
    st.subheader("Historical + Forecast Trend")
    target_sel = st.selectbox("Select KPI to plot:", TARGET_COLS,
                              format_func=lambda x: TARGET_LABELS[x],
                              key="fct_target")

    c         = COLORS[TARGET_COLS.index(target_sel)]
    rmse      = fct_metrics[target_sel]["RMSE"]
    hist_vals = hist_monthly[target_sel].values
    fct_vals  = monthly_fct[f"Pred_{target_sel}"].values
    h_names   = ["Mar", "Apr", "May"]
    f_names   = list(MONTH_NAMES.values())

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(h_names)), hist_vals, color=c, lw=2.5,
            marker="o", markersize=8, label="Historical (actual)")
    ax.plot([len(h_names)-1, len(h_names)], [hist_vals[-1], fct_vals[0]],
            color=c, lw=1.5, ls="--", alpha=0.5)
    fct_x = range(len(h_names), len(h_names) + len(f_names))
    ax.plot(list(fct_x), fct_vals, color=c, lw=2.5,
            marker="s", markersize=8, ls="--", label="Forecast Jul–Dec 2024")
    ax.fill_between(list(fct_x),
                    [v - rmse for v in fct_vals],
                    [v + rmse for v in fct_vals],
                    color=c, alpha=0.15, label=f"±RMSE band (±{rmse:.3f})")
    ax.axvline(x=len(h_names)-0.5, color="gray", ls=":", lw=1.5)
    ax.text(len(h_names)-0.3, ax.get_ylim()[0], "Forecast →", fontsize=9, color="gray")
    ax.annotate(f"Dec: {fct_vals[-1]:.2f}",
                xy=(list(fct_x)[-1], fct_vals[-1]),
                xytext=(list(fct_x)[-1]-0.9, fct_vals[-1] + rmse*0.8),
                fontsize=9, fontweight="bold", color=c,
                arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
    all_labels = h_names + f_names
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=30)
    ax.set_title(f"{TARGET_LABELS[target_sel]} — Historical + Forecast",
                 fontweight="bold", fontsize=12)
    ax.set_ylabel("Mean Value (all cells)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig)

    st.markdown("---")

    # ── By layer chart ─────────────────────────────────────────────────────────
    st.subheader("Forecast by Radio Layer — December 2024")
    dec_layer = layer_monthly[layer_monthly["Month"] == 12]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for i, target in enumerate(TARGET_COLS):
        pred_col = f"Pred_{target}"
        bars = axes[i].bar(
            dec_layer["Layer"], dec_layer[pred_col],
            color=[LAYER_COLORS[l] for l in dec_layer["Layer"]],
            edgecolor="white", alpha=0.88, width=0.5
        )
        for bar, val in zip(bars, dec_layer[pred_col]):
            axes[i].text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()*1.01,
                         f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
        rmse = fct_metrics[target]["RMSE"]
        axes[i].errorbar(dec_layer["Layer"], dec_layer[pred_col],
                         yerr=rmse, fmt="none", color="black", capsize=6, linewidth=1.5)
        axes[i].set_title(TARGET_LABELS[target], fontweight="bold", fontsize=10)
        axes[i].set_ylabel("Predicted"); axes[i].grid(True, axis="y", alpha=0.3)
    plt.suptitle("December 2024 Forecast per Layer  (error bars = ±RMSE)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig)

    st.markdown("---")
    st.info("💡 **All forecasts use the trained Random Forest and Ridge models. "
            "Error bars represent ±1 RMSE from the test set evaluation.**")

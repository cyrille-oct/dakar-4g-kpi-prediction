"""
app/streamlit_app.py — Dakar 4G KPI Prediction Dashboard
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
from collections import defaultdict

st.set_page_config(page_title="Dakar 4G KPI Prediction",
                   page_icon="📡", layout="wide",
                   initial_sidebar_state="expanded")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

CFG_JSON  = os.path.join(DATA_DIR,    "column_config.json")
CLEAN_CSV = os.path.join(DATA_DIR,    "clean_dataset.csv")
KPI_CSV   = os.path.join(DATA_DIR,    "KPI_Dataset_Dakar_4G.csv")
METRICS_J = os.path.join(OUTPUTS_DIR, "test_metrics.json")
BENCH_J   = os.path.join(OUTPUTS_DIR, "benchmark_results.json")

TARGET_LABELS = {
    "Total_DataVol_GB"     : "Data Volume (GB)",
    "LTE_Avg_RRC_CU"       : "Avg Connected Users",
    "LTE_Cell_THP_DL_Mbps" : "DL Cell Throughput (Mbps)",
    "LTE_PrbUtil_DL_pct"   : "DL PRB Utilisation (%)",
}
COLORS       = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
LAYER_COLORS = {"L800": "#3F88C5", "L1800": "#F4720B", "L2600": "#44BBA4"}
MONTH_NAMES  = {7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

# ── Load core artefacts ───────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    with open(CFG_JSON) as f: cfg = json.load(f)
    feature_cols = cfg["FEATURE_COLS"]
    target_cols  = cfg["TARGET_COLS"]
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    models = {}
    for t in target_cols:
        safe = t.replace("/","_")
        models[t] = joblib.load(os.path.join(MODELS_DIR, f"model_{safe}.pkl"))
    with open(METRICS_J) as f: test_metrics = json.load(f)
    with open(BENCH_J)   as f: benchmark    = json.load(f)
    df = pd.read_csv(CLEAN_CSV)
    for ec, sc in [("Layer_enc","Layer"),("Cell_enc","Cell_ID"),("BTS_enc","BTS_ID")]:
        if ec not in df.columns and sc in df.columns:
            df[ec] = LabelEncoder().fit_transform(df[sc])
    if "DayOfWeek" not in df.columns and "Date" in df.columns:
        df["Date"]       = pd.to_datetime(df["Date"])
        df               = df.sort_values("Date").reset_index(drop=True)
        df["DayOfWeek"]  = df["Date"].dt.dayofweek
        df["DayOfMonth"] = df["Date"].dt.day
        df["Month"]      = df["Date"].dt.month
        df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
        df["IsWeekend"]  = (df["DayOfWeek"] >= 5).astype(int)
    if "DL_UL_Vol_Ratio" not in df.columns:
        df["DL_UL_Vol_Ratio"] = df["DataVol_DL_DRB_GB"]/(df["DataVol_UL_DRB_GB"]+1e-6)
        df["PRB_Total"]       = df["LTE_PrbUtil_DL_pct"]+df["LTE_PrbUtil_UL_pct"]
    available    = [f for f in feature_cols if f in df.columns]
    df           = df[available+target_cols].dropna().reset_index(drop=True)
    feature_cols = available
    split_idx    = int(0.80*len(df))
    df_test      = df.iloc[split_idx:].reset_index(drop=True)
    X_test_sc    = scaler.transform(df_test[feature_cols].values)
    return scaler,models,test_metrics,benchmark,df,df_test,X_test_sc,feature_cols,target_cols

# ── Load worst cell data ──────────────────────────────────────────────────────
@st.cache_data
def load_worst_cells():
    summary_csv = os.path.join(OUTPUTS_DIR, "worst_cells_top5_summary.csv")
    full_csv    = os.path.join(OUTPUTS_DIR, "worst_cells_december_2024.csv")
    if not os.path.exists(summary_csv) or not os.path.exists(full_csv):
        from phase3c_worst_cells import run_worst_cell_analysis
        return run_worst_cell_analysis()
    cell_layer_dec = pd.read_csv(full_csv)
    summary        = pd.read_csv(summary_csv)
    top_vol = cell_layer_dec.nlargest(5,"DataVol").reset_index(drop=True)
    top_usr = cell_layer_dec.nlargest(5,"Users").reset_index(drop=True)
    top_thp = cell_layer_dec.nsmallest(5,"Throughput").reset_index(drop=True)
    top_prb = cell_layer_dec.nlargest(5,"PRB").reset_index(drop=True)
    return {"top_vol":top_vol,"top_usr":top_usr,"top_thp":top_thp,
            "top_prb":top_prb,"cell_layer_dec":cell_layer_dec,"summary":summary}

# ── Load forecast data ────────────────────────────────────────────────────────
@st.cache_data
def load_forecast():
    monthly_path = os.path.join(OUTPUTS_DIR,"forecast_Jul_Dec_2024_monthly.csv")
    daily_path   = os.path.join(OUTPUTS_DIR,"forecast_Jul_Dec_2024_daily.csv")
    if not os.path.exists(monthly_path) or not os.path.exists(daily_path):
        from phase3b_forecast import run_forecast
        results = run_forecast()
        return results["monthly_fct"],results["layer_monthly"],\
               results["hist_monthly"],results["test_metrics"]
    monthly_fct   = pd.read_csv(monthly_path)
    daily_fct     = pd.read_csv(daily_path)
    daily_fct["Date"] = pd.to_datetime(daily_fct["Date"])
    daily_fct["Month"] = daily_fct["Date"].dt.month
    pred_cols     = [c for c in daily_fct.columns if c.startswith("Pred_")]
    layer_monthly = daily_fct.groupby(["Month","Layer"])[pred_cols].mean().reset_index()
    layer_monthly["MonthName"] = layer_monthly["Month"].map(MONTH_NAMES)
    df_hist = pd.read_csv(KPI_CSV)
    df_hist["Date"]  = pd.to_datetime(df_hist["Date"])
    df_hist["Month"] = df_hist["Date"].dt.month
    tc = ["Total_DataVol_GB","LTE_Avg_RRC_CU","LTE_Cell_THP_DL_Mbps","LTE_PrbUtil_DL_pct"]
    hist_monthly = (df_hist[df_hist["Month"].isin([3,4,5])]
                    .groupby("Month")[tc].mean().reset_index())
    hist_monthly["MonthName"] = hist_monthly["Month"].map({3:"Mar",4:"Apr",5:"May"})
    with open(METRICS_J) as f: test_metrics = json.load(f)
    return monthly_fct, layer_monthly, hist_monthly, test_metrics

# ── Load artefacts ────────────────────────────────────────────────────────────
try:
    scaler,models,test_metrics,benchmark,df,df_test,X_test_sc,FEATURE_COLS,TARGET_COLS = load_artefacts()
    data_loaded = True
except Exception as e:
    data_loaded = False; load_error = str(e)

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
    "🚨 Worst Cell Analysis",
    "📅 6-Month Forecast",
])

st.title("📡 Dakar 4G Network — KPI Prediction Dashboard")
st.markdown("Regression-based prediction of **Data Volume · Connected Users · Throughput · PRB Utilisation**")

if not data_loaded:
    st.error(f"⚠️ Could not load artefacts: {load_error}"); st.stop()

# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.header("Project Overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Cells","30"); c2.metric("BTS Sites","10")
    c3.metric("Period","Mar–Jun 2024"); c4.metric("Training Records",f"{int(0.8*len(df)):,}")
    st.markdown("---"); st.subheader("Final Model Performance")
    rows=[]
    for t in TARGET_COLS:
        m=test_metrics[t]
        rows.append({"Target":TARGET_LABELS[t],"Model":m["model"],
                     "MAE":m["MAE"],"RMSE":m["RMSE"],"R²":m["R2"]})
    st.dataframe(pd.DataFrame(rows).style.format({"MAE":"{:.4f}","RMSE":"{:.4f}","R²":"{:.4f}"}),
                 use_container_width=True)
    st.markdown("---"); st.subheader("Dataset Statistics")
    c1,c2=st.columns(2)
    with c1:
        st.write("**Target variable statistics:**")
        st.dataframe(df[TARGET_COLS].describe().round(3))
    with c2:
        st.write("**Records per layer:**")
        if os.path.exists(KPI_CSV):
            st.dataframe(pd.read_csv(KPI_CSV)["Layer"].value_counts().reset_index()
                         .rename(columns={"index":"Layer","Layer":"Count"}))

# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.header("Model Performance — Test Set")
    tsel = st.selectbox("Select target:",TARGET_COLS,format_func=lambda x:TARGET_LABELS[x])
    y_t=df_test[tsel].values; y_p=models[tsel].predict(X_test_sc)
    res=y_t-y_p; c=COLORS[TARGET_COLS.index(tsel)]
    c1,c2,c3=st.columns(3)
    c1.metric("MAE",f"{mean_absolute_error(y_t,y_p):.4f}")
    c2.metric("RMSE",f"{np.sqrt(mean_squared_error(y_t,y_p)):.4f}")
    c3.metric("R²",f"{r2_score(y_t,y_p):.4f}")
    ca,cb=st.columns(2)
    with ca:
        fig,ax=plt.subplots(figsize=(6,5))
        ax.scatter(y_t,y_p,alpha=0.30,s=12,color=c)
        lims=[min(y_t.min(),y_p.min()),max(y_t.max(),y_p.max())]
        ax.plot(lims,lims,"k--",lw=1.5,label="Perfect fit")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted",fontweight="bold"); ax.legend(); st.pyplot(fig)
    with cb:
        fig,axes=plt.subplots(1,2,figsize=(8,4))
        axes[0].scatter(y_p,res,alpha=0.25,s=10,color=c)
        axes[0].axhline(0,color="black",ls="--",lw=1.5)
        axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")
        axes[0].set_title("Residuals vs Predicted",fontweight="bold")
        axes[1].hist(res,bins=40,color=c,edgecolor="white",alpha=0.85)
        axes[1].axvline(res.mean(),color="red",ls=":",lw=1.5,label=f"Mean={res.mean():.3f}")
        axes[1].set_xlabel("Residual"); axes[1].set_title("Residual Distribution",fontweight="bold")
        axes[1].legend(fontsize=8); plt.tight_layout(); st.pyplot(fig)
    if hasattr(models[tsel],"feature_importances_"):
        st.subheader("Feature Importance — Top 15")
        fi=pd.Series(models[tsel].feature_importances_,index=FEATURE_COLS)
        fi=fi.sort_values(ascending=False).head(15).reset_index()
        fi.columns=["Feature","Importance"]; st.bar_chart(fi.set_index("Feature"))

# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Single Prediction":
    st.header("Single Cell KPI Prediction")
    c1,c2,c3=st.columns(3)
    with c1:
        layer=st.selectbox("Layer",["L800","L1800","L2600"])
        layer_enc={"L800":1,"L1800":0,"L2600":2}.get(layer,1)
        dow=st.slider("Day of Week (0=Mon)",0,6,1)
        month=st.slider("Month",1,12,7)
    with c2:
        pdcch=st.slider("PDCCH CCE Load (%)",0.0,100.0,30.0)
        max_rrc=st.slider("Max RRC Users",50.0,300.0,130.0)
        prb_dl=st.slider("PRB DL (%)",10.0,100.0,52.0)
        prb_ul=st.slider("PRB UL (%)",5.0,60.0,25.0)
    with c3:
        sr_fail=st.slider("SR Fail Ratio",0.0,0.1,0.012,step=0.001)
        cfi3=st.slider("CFI3 Share (%)",0.0,100.0,55.0)
        overload=st.selectbox("Overload Flag",[0,1])
    if st.button("🔮 Predict KPIs",type="primary"):
        sd=dict(zip(FEATURE_COLS,df_test[FEATURE_COLS].values[0]))
        sd.update({"Layer_enc":layer_enc,"DayOfWeek":dow,"Month":month,
                   "IsWeekend":1 if dow>=5 else 0,"DayOfMonth":15,"WeekOfYear":28,
                   "LTE_PDCCH_CCE_Load_pct":pdcch,"LTE_Max_RRC_CU":max_rrc,
                   "LTE_PrbUtil_DL_pct":prb_dl,"LTE_PrbUtil_UL_pct":prb_ul,
                   "SR_Fail_Ratio":sr_fail,"LTE_CFI3_Share_pct":cfi3,
                   "Overload_Flag":overload,"PRB_Total":prb_dl+prb_ul})
        x=np.array([[sd.get(f,0) for f in FEATURE_COLS]])
        x_sc=scaler.transform(x)
        st.markdown("---"); st.subheader("Predictions")
        cols=st.columns(4)
        for j,t in enumerate(TARGET_COLS):
            cols[j].metric(TARGET_LABELS[t],f"{float(models[t].predict(x_sc)[0]):.3f}")

# ════════════════════════════════════════════════════════════════════════════
elif page == "📈 Monitoring Dashboard":
    st.header("Monitoring Dashboard")
    WINDOW=st.sidebar.slider("Rolling window",10,100,50)
    ALERT_PCT=st.sidebar.slider("Alert threshold (%)",5,50,20)/100
    dp=df_test[TARGET_COLS].copy()
    for t in TARGET_COLS:
        dp[f"Pred_{t}"]=models[t].predict(X_test_sc)
        dp[f"Err_{t}"]=dp[t]-dp[f"Pred_{t}"]
    fig,axes=plt.subplots(2,2,figsize=(14,8)); axes=axes.flatten()
    for i,(t,c) in enumerate(zip(TARGET_COLS,COLORS)):
        err=dp[f"Err_{t}"].values
        rr=pd.Series(err**2).rolling(WINDOW,min_periods=5).mean().apply(np.sqrt)
        gr=float(np.sqrt(np.mean(err**2))); al=gr*(1+ALERT_PCT)
        axes[i].plot(rr.index,rr.values,color=c,lw=1.8,label=f"Rolling RMSE (w={WINDOW})")
        axes[i].axhline(gr,color="black",ls="--",lw=1.2,label=f"Global={gr:.3f}")
        axes[i].axhline(al,color="red",ls=":",lw=1.5,label=f"Alert (+{int(ALERT_PCT*100)}%)")
        axes[i].fill_between(rr.index,gr,al,alpha=0.08,color="orange")
        ap=rr[rr>al]
        if len(ap): axes[i].scatter(ap.index,ap.values,color="red",s=20,zorder=5)
        axes[i].set_title(TARGET_LABELS[t],fontweight="bold")
        axes[i].set_xlabel("Test sample"); axes[i].set_ylabel("RMSE")
        axes[i].legend(fontsize=7,ncol=2); axes[i].grid(True,alpha=0.3)
    plt.suptitle("Rolling RMSE Monitoring",fontsize=13,fontweight="bold")
    plt.tight_layout(); st.pyplot(fig)
    st.subheader("Error Percentile Summary")
    rows=[]
    for t in TARGET_COLS:
        ae=dp[f"Err_{t}"].abs().values
        p50,p75,p90,p95,p99=np.percentile(ae,[50,75,90,95,99])
        rows.append({"Target":TARGET_LABELS[t],"P50":round(p50,4),"P75":round(p75,4),
                     "P90":round(p90,4),"P95":round(p95,4),"P99":round(p99,4)})
    st.dataframe(pd.DataFrame(rows),use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Benchmark Comparison":
    st.header("Model Benchmark — 5-Fold Cross-Validation")
    tsel=st.selectbox("Select target:",TARGET_COLS,format_func=lambda x:TARGET_LABELS[x])
    bm=benchmark[tsel]
    rows=[{"Model":m,"CV RMSE":v["RMSE"],"CV R²":v["R2"]} for m,v in bm.items()]
    df_bm=pd.DataFrame(rows).sort_values("CV RMSE").reset_index(drop=True); df_bm.index+=1
    c1,c2=st.columns([1,1])
    with c1:
        st.subheader("Ranked Table")
        st.dataframe(df_bm.style.format({"CV RMSE":"{:.4f}","CV R²":"{:.4f}"}),use_container_width=True)
    with c2:
        st.subheader("RMSE Comparison"); st.bar_chart(df_bm.set_index("Model")[["CV RMSE"]])
    st.info(f"✅ **Best:** {df_bm.iloc[0]['Model']}  RMSE={df_bm.iloc[0]['CV RMSE']:.4f}  R²={df_bm.iloc[0]['CV R²']:.4f}")

# ════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Worst Cell Analysis":
    st.header("🚨 Worst Cell Analysis — December 2024 Forecast")
    st.markdown("Top 5 cells most at risk per KPI. All worst-case cells are on **L2600**.")

    with st.spinner("Loading worst cell data..."):
        try:
            wc=load_worst_cells(); wc_ok=True
        except Exception as e:
            wc_ok=False; st.error(f"Error: {e}")
    if not wc_ok: st.stop()

    top_vol=wc["top_vol"]; top_usr=wc["top_usr"]
    top_thp=wc["top_thp"]; top_prb=wc["top_prb"]

    kpi_sel=st.radio("Select KPI category:",[
        "📦 High Data Volume","👥 High Connected Users",
        "📉 Low Throughput","📡 High PRB Utilisation"],horizontal=True)
    st.markdown("---")

    def wc_table(df,col_f,col_h,col_d,unit):
        return pd.DataFrame({"Rank":[f"#{i+1}" for i in range(len(df))],
            "Cell ID":df["Cell_ID"],"BTS":df["BTS_ID"],"Layer":df["Layer"],
            f"Dec 2024 ({unit})":df[col_f].round(3),
            f"Historical ({unit})":df[col_h].round(3),
            f"Delta ({unit})":df[col_d].round(3)})

    def wc_chart(df,col_h,col_f,col_d,title,unit,hiw=True):
        BG="#F8F8F8"; RD="#C73E1D"
        fig,axes=plt.subplots(1,2,figsize=(13,5)); fig.patch.set_facecolor(BG)
        cells=df["Cell_ID"]+"\n"+df["Layer"]; x=np.arange(len(cells)); w=0.35
        for ax in axes: ax.set_facecolor(BG)
        axes[0].bar(x-w/2,df[col_h],w,label="Historical avg",color="#AACDE8",edgecolor="white",zorder=3)
        bf=axes[0].bar(x+w/2,df[col_f],w,label="Dec 2024 Forecast",color=RD,edgecolor="white",zorder=3)
        for bar in bf:
            axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.15,
                f"{bar.get_height():.2f}",ha="center",va="bottom",fontsize=9,fontweight="bold",color=RD)
        axes[0].set_xticks(x); axes[0].set_xticklabels(cells,fontsize=9)
        axes[0].set_ylabel(unit); axes[0].legend(fontsize=8)
        axes[0].set_title("Historical vs Forecast",fontweight="bold")
        axes[0].grid(True,axis="y",alpha=0.4)
        axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)
        dc=[RD if (hiw and d>0) or (not hiw and d<0) else "#44BBA4" for d in df[col_d]]
        bd=axes[1].bar(x,df[col_d],color=dc,edgecolor="white",zorder=3)
        axes[1].axhline(0,color="black",lw=1.0,zorder=4)
        for bar,val in zip(bd,df[col_d]):
            axes[1].text(bar.get_x()+bar.get_width()/2,val+0.03 if val>=0 else val-0.08,
                f"{val:+.2f}",ha="center",va="bottom",fontsize=9,fontweight="bold",
                color=RD if (hiw and val>0) or (not hiw and val<0) else "#1a6e3f")
        axes[1].set_xticks(x); axes[1].set_xticklabels(cells,fontsize=9)
        axes[1].set_ylabel("Change vs Historical")
        axes[1].set_title("Delta vs Historical",fontweight="bold")
        axes[1].grid(True,axis="y",alpha=0.4)
        axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
        fig.suptitle(title,fontsize=12,fontweight="bold",color="#1F5C99",y=1.01)
        plt.tight_layout(); return fig

    if kpi_sel=="📦 High Data Volume":
        st.subheader("Top 5 — Highest Forecast Data Volume (GB)")
        c1,c2=st.columns([1,2])
        with c1:
            st.dataframe(wc_table(top_vol,"DataVol","DataVol_hist","DataVol_delta","GB"),
                         use_container_width=True,hide_index=True)
            st.info("All 5 cells on L2600. Forecast >27 GB/day (+14.6% vs historical).")
        with c2:
            st.pyplot(wc_chart(top_vol,"DataVol_hist","DataVol","DataVol_delta",
                "Top 5 — Highest Data Volume Dec 2024","GB"))

    elif kpi_sel=="👥 High Connected Users":
        st.subheader("Top 5 — Highest Forecast Connected Users")
        c1,c2=st.columns([1,2])
        with c1:
            st.dataframe(wc_table(top_usr,"Users","Users_hist","Users_delta","users"),
                         use_container_width=True,hide_index=True)
            st.info("DK0007 and DK0008 appear twice — priority sites for load balancing.")
        with c2:
            st.pyplot(wc_chart(top_usr,"Users_hist","Users","Users_delta",
                "Top 5 — Highest Connected Users Dec 2024","users"))

    elif kpi_sel=="📉 Low Throughput":
        st.subheader("Top 5 — Lowest Forecast DL Throughput (Mbps)")
        c1,c2=st.columns([1,2])
        with c1:
            st.dataframe(wc_table(top_thp,"Throughput","Throughput_hist","Throughput_delta","Mbps"),
                         use_container_width=True,hide_index=True)
            st.info("All C-sector cells. DK0008C most degraded: 6.18 Mbps.")
        with c2:
            st.pyplot(wc_chart(top_thp,"Throughput_hist","Throughput","Throughput_delta",
                "Top 5 — Lowest Throughput Dec 2024","Mbps",hiw=False))

    else:
        st.subheader("Top 5 — Highest Forecast PRB Utilisation (%)")
        c1,c2=st.columns([1,2])
        with c1:
            st.dataframe(wc_table(top_prb,"PRB","PRB_hist","PRB_delta","%"),
                         use_container_width=True,hide_index=True)
            st.info("DK0007A leads at 61.8%. Negative delta = L2600 efficiency improving.")
        with c2:
            st.pyplot(wc_chart(top_prb,"PRB_hist","PRB","PRB_delta",
                "Top 5 — Highest PRB Utilisation Dec 2024","%"))

    st.markdown("---")
    st.subheader("Cross-KPI Priority Summary")
    app=defaultdict(list)
    for _,r in top_vol.iterrows(): app[r.Cell_ID].append("High DataVol")
    for _,r in top_usr.iterrows(): app[r.Cell_ID].append("High Users")
    for _,r in top_thp.iterrows(): app[r.Cell_ID].append("Low Throughput")
    for _,r in top_prb.iterrows(): app[r.Cell_ID].append("High PRB")
    prows=[]
    for cell,cats in sorted(app.items(),key=lambda x:-len(x[1])):
        n=len(cats)
        pr="🔴 CRITICAL" if n>=2 else "🟠 HIGH"
        prows.append({"Cell ID":cell,"BTS":cell[:6],"Issues":" · ".join(cats),"#":n,"Priority":pr})
    st.dataframe(pd.DataFrame(prows),use_container_width=True,hide_index=True)
    st.markdown("---")
    st.subheader("Network Planning Recommendations")
    recs=[("🔴 Immediate","Set PRB alert at 65% for L2600","DK0007A, DK0002A, DK0004B, DK0005B, DK0006A"),
          ("🟠 Short-term","Load balance B/C sectors DK0007 and DK0008","DK0007B/C, DK0008B/C, DK0009C"),
          ("🟠 Short-term","Capacity upgrade feasibility study","DK0004A, DK0002A, DK0005A"),
          ("🟡 Ongoing","Weekly PRB monitoring all L2600 cells","All L2600 cells")]
    st.dataframe(pd.DataFrame(recs,columns=["Priority","Action","Target Cells"]),
                 use_container_width=True,hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
elif page == "📅 6-Month Forecast":
    st.header("📅 6-Month KPI Forecast — July to December 2024")
    st.markdown("Forecasts generated by projecting per-layer growth trends from Mar–May 2024.")
    with st.spinner("Loading forecast data..."):
        try:
            monthly_fct,layer_monthly,hist_monthly,fct_metrics=load_forecast(); fok=True
        except Exception as e:
            fok=False; st.error(f"Forecast error: {e}")
    if not fok: st.stop()

    pred_cols=[f"Pred_{t}" for t in TARGET_COLS]
    st.subheader("Network-Wide Monthly Forecast — December 2024")
    may_row=hist_monthly[hist_monthly["MonthName"]=="May"].iloc[0]
    dec_row=monthly_fct[monthly_fct["MonthName"]=="Dec"].iloc[0]
    cols=st.columns(4)
    for j,t in enumerate(TARGET_COLS):
        may_v=may_row[t]; dec_v=dec_row[f"Pred_{t}"]
        delta=((dec_v-may_v)/may_v)*100
        cols[j].metric(TARGET_LABELS[t],f"{dec_v:.2f}",f"{delta:+.1f}% vs May 2024")
    st.markdown("---")
    st.subheader("Monthly Forecast Table")
    disp=monthly_fct[["MonthName"]+pred_cols].copy()
    disp.columns=["Month"]+[TARGET_LABELS[t] for t in TARGET_COLS]
    st.dataframe(disp.style.format({TARGET_LABELS[t]:"{:.3f}" for t in TARGET_COLS}),
                 use_container_width=True)
    st.markdown("---")
    st.subheader("Historical + Forecast Trend")
    tsel=st.selectbox("Select KPI:",TARGET_COLS,format_func=lambda x:TARGET_LABELS[x],key="fct_t")
    c=COLORS[TARGET_COLS.index(tsel)]; rmse=fct_metrics[tsel]["RMSE"]
    hv=hist_monthly[tsel].values; fv=monthly_fct[f"Pred_{tsel}"].values
    hn=["Mar","Apr","May"]; fn=list(MONTH_NAMES.values())
    fig,ax=plt.subplots(figsize=(12,5))
    ax.plot(range(len(hn)),hv,color=c,lw=2.5,marker="o",markersize=8,label="Historical (actual)")
    ax.plot([len(hn)-1,len(hn)],[hv[-1],fv[0]],color=c,lw=1.5,ls="--",alpha=0.5)
    fx=range(len(hn),len(hn)+len(fn))
    ax.plot(list(fx),fv,color=c,lw=2.5,marker="s",markersize=8,ls="--",label="Forecast Jul–Dec 2024")
    ax.fill_between(list(fx),[v-rmse for v in fv],[v+rmse for v in fv],
                    color=c,alpha=0.15,label=f"±RMSE (±{rmse:.3f})")
    ax.axvline(x=len(hn)-0.5,color="gray",ls=":",lw=1.5)
    ax.text(len(hn)-0.3,ax.get_ylim()[0],"Forecast →",fontsize=9,color="gray")
    ax.annotate(f"Dec: {fv[-1]:.2f}",xy=(list(fx)[-1],fv[-1]),
                xytext=(list(fx)[-1]-0.9,fv[-1]+rmse*0.8),fontsize=9,fontweight="bold",color=c,
                arrowprops=dict(arrowstyle="->",color=c,lw=1.2))
    al=hn+fn; ax.set_xticks(range(len(al))); ax.set_xticklabels(al,rotation=30)
    ax.set_title(f"{TARGET_LABELS[tsel]} — Historical + Forecast",fontweight="bold",fontsize=12)
    ax.set_ylabel("Mean Value (all cells)"); ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
    plt.tight_layout(); st.pyplot(fig)
    st.markdown("---")
    st.subheader("December 2024 Forecast by Layer")
    dec_layer=layer_monthly[layer_monthly["Month"]==12]
    fig,axes=plt.subplots(1,4,figsize=(16,5))
    for i,target in enumerate(TARGET_COLS):
        pc=f"Pred_{target}"
        bars=axes[i].bar(dec_layer["Layer"],dec_layer[pc],
                         color=[LAYER_COLORS[l] for l in dec_layer["Layer"]],
                         edgecolor="white",alpha=0.88,width=0.5)
        for bar,val in zip(bars,dec_layer[pc]):
            axes[i].text(bar.get_x()+bar.get_width()/2,bar.get_height()*1.01,
                         f"{val:.2f}",ha="center",fontsize=10,fontweight="bold")
        rm=fct_metrics[target]["RMSE"]
        axes[i].errorbar(dec_layer["Layer"],dec_layer[pc],yerr=rm,
                         fmt="none",color="black",capsize=6,linewidth=1.5)
        axes[i].set_title(TARGET_LABELS[target],fontweight="bold",fontsize=10)
        axes[i].set_ylabel("Predicted"); axes[i].grid(True,axis="y",alpha=0.3)
    plt.suptitle("December 2024 per Layer  (error bars = ±RMSE)",fontsize=12,fontweight="bold")
    plt.tight_layout(); st.pyplot(fig)
    st.info("💡 All forecasts use trained Random Forest and Ridge models. Error bars = ±1 RMSE.")

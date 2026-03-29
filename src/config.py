"""
config.py — Central configuration for Dakar 4G KPI Prediction Project
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
# BASE_DIR = the project root (parent of the src/ folder)
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
MODELS_DIR  = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

KPI_CSV   = DATA_DIR / "KPI_Dataset_Dakar_4G.csv"
CELL_XLSX = DATA_DIR / "4g_Dakar_RAN_Cellfile.xlsx"   # actual filename on disk
CLEAN_CSV = DATA_DIR / "clean_dataset.csv"
ENC_PKL   = MODELS_DIR / "encoders.pkl"
CFG_JSON  = DATA_DIR / "column_config.json"

# Convert to strings for libraries that don't accept Path objects
KPI_CSV     = str(KPI_CSV)
CELL_XLSX   = str(CELL_XLSX)
CLEAN_CSV   = str(CLEAN_CSV)
ENC_PKL     = str(ENC_PKL)
CFG_JSON    = str(CFG_JSON)
DATA_DIR    = str(DATA_DIR)
MODELS_DIR  = str(MODELS_DIR)
OUTPUTS_DIR = str(OUTPUTS_DIR)

# ── Prediction targets ──────────────────────────────────────────────────────
TARGET_COLS = [
    "Total_DataVol_GB",
    "LTE_Avg_RRC_CU",
    "LTE_Cell_THP_DL_Mbps",
    "LTE_PrbUtil_DL_pct",
]
TARGET_LABELS = {
    "Total_DataVol_GB"     : "Data Volume (GB)",
    "LTE_Avg_RRC_CU"       : "Avg Connected Users",
    "LTE_Cell_THP_DL_Mbps" : "DL Cell Throughput (Mbps)",
    "LTE_PrbUtil_DL_pct"   : "DL PRB Utilisation (%)",
}

# ── Feature columns ─────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Layer_enc", "Cell_enc", "BTS_enc",
    "DayOfWeek", "DayOfMonth", "Month", "WeekOfYear", "IsWeekend",
    "Overload_Flag",
    "LTE_PDCCH_CCE_Load_pct", "LTE_CFI3_Share_pct",
    "LTE_CceAgg_1", "LTE_CceAgg_2", "LTE_CceAgg_4", "LTE_CceAgg_8",
    "PDCCH_Adm_Rej_Count", "LTE_Max_RRC_CU",
    "LTE_Active_UL_Users_TTI", "SR_Fail_Ratio", "PUCCH_SR_Cong_Count",
    "DL_UL_Vol_Ratio", "PRB_Total",
    "Lat", "Lon", "Azimuth", "PCI", "EARFCN", "TAC",
]

# ── ML settings ─────────────────────────────────────────────────────────────
SEED         = 42
TRAIN_SPLIT  = 0.80
CV_FOLDS     = 5

# ── Plot colours ────────────────────────────────────────────────────────────
COLORS       = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
LAYER_COLORS = {"L800": "#3F88C5", "L1800": "#F4720B", "L2600": "#44BBA4"}
DOW_LABELS   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

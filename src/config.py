from pathlib import Path

# --- Yollar ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
HOURLY_PATH = PROCESSED_DIR / "milan_hourly.parquet"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"
BASELINE_MODEL_PATH = MODEL_OUTPUT_DIR / "baseline_lstm.pt"
BASELINE_SCALER_PATH = MODEL_OUTPUT_DIR / "baseline_scaler.parquet"

# --- Izgara ---
GRID_SIZE = 100
CENTER_CELL = 5161
WINDOW_RADIUS = 10

# --- Veri yükleme ---
FILE_GLOB = "sms-call-internet-mi-*.txt"
NUM_DAYS = 21
VALUE_COL = "internet"
RAW_FREQ = "10min"
TARGET_FREQ = "1h"

# --- Hücre eleme eşikleri ---
MIN_TOTAL = 1000
MIN_STD = 10

# --- Merkezi baseline ---
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
WINDOW_SIZE = 24
FORECAST_HORIZON = 1
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
BASELINE_SEED = 123
COMPARISON_SEEDS = (42, 123, 456)
SELECTED_MODEL = "LSTM"
LSTM_HIDDEN_SIZE = 64

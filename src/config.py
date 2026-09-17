from pathlib import Path

# --- Yollar ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
HOURLY_PATH = PROCESSED_DIR / "milan_hourly.parquet"

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
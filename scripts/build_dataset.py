import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    RAW_DIR,
    PROCESSED_DIR,
    HOURLY_PATH,
    CENTER_CELL,
    WINDOW_RADIUS,
    NUM_DAYS,
    VALUE_COL,
    MIN_TOTAL,
    MIN_STD,
)
from src.data.loader import load_total_data
from src.data.grid import get_window
from src.data.preprocess import select_valid_cells, to_hourly_pivot



def main():
    df = load_total_data(NUM_DAYS, RAW_DIR, verbose=True)
    window = get_window(CENTER_CELL, WINDOW_RADIUS)
    sub = df[df["square_id"].isin(window)]
    
    valid, dropped = select_valid_cells(sub, MIN_TOTAL, MIN_STD)
    sub = sub[sub["square_id"].isin(valid)]
    
    hourly = to_hourly_pivot(sub, VALUE_COL)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(HOURLY_PATH)
    print(f"kaydedildi: {HOURLY_PATH}")


if __name__ == "__main__":
    main()


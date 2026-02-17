# batch_backtest.py
# Generates JSON snapshots for multiple historical dates (for training data)
# Now with parallel processing per day for faster generation (at least 4 threads)

import yaml
from orchestrator import generate_snapshot
from modules.loader import load_nq_csv
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load base config
with open("config/config.yaml") as f:
    base_config = yaml.safe_load(f)

# LIST OF DATES YOU WANT TO BACKTEST / TRAIN ON
# Add as many as you have CSVs for (must match session_date column exactly)
start_date = datetime(2024,12,24)
end_date = datetime(2025, 12,23)
dates = []
current = start_date
while current <= end_date:
    if current.weekday() < 5:  # 0=Mon, 4=Fri
        dates.append(current.strftime("%Y-%m-%d"))
    current += timedelta(days=1)

# Snapshot times per day (key moments you care about)
times = ["09:30", "09:35", "09:45", "09:50", "09:55", "10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30", "11:45", "12:00", "12:15", "12:30", "12:45", "13:00", "13:15", "13:30", "13:45", "14:00", "14:15", "14:30", "14:45", "15:00", "15:15", "15:30", "15:45", "16:00"]

print(f"Starting batch generation for {len(dates)} dates × {len(times)} snapshots each... (with 4+ parallel threads per day)")

def process_time(date, t, df_extended, df_current):
    try:
        config = base_config.copy()
        config['session_date'] = date
        config['current_time'] = t
        snapshot = generate_snapshot(config)  # Assumes generate_snapshot uses df_extended/df_current if needed; adjust if not
        return f"Generated: {date} {t}", None
    except Exception as e:
        return None, f"Failed {date} {t}: {e}"

for date in dates:
    try:
        df_extended, df_current = load_nq_csv(base_config['csv_paths']['nq'], date)
        if df_current.empty:
            print(f"No data for {date}, skipping")
            continue
        max_time = df_current.index.max().strftime('%H:%M')
        times_filtered = [t for t in times if t <= max_time]
        print(f"For {date}, max time: {max_time}, generating {len(times_filtered)} snapshots in parallel...")
        
        with ThreadPoolExecutor(max_workers=10) as executor:  # At least 4 threads
            future_to_t = {executor.submit(process_time, date, t, df_extended, df_current): t for t in times_filtered}
            for future in as_completed(future_to_t):
                success, error = future.result()
                if success:
                    print(success)
                else:
                    print(error)
    except Exception as e:
        print(f"Failed to load data for {date}: {e}")

print("\nBatch complete! JSONs saved in data/json_snapshots/")
print("Next: Drop them here one by one for ROCKIT markup → build training pairs")
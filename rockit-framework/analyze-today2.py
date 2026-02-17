#!/usr/bin/env python3

"""
analyze-today.py

Automates real-time data processing for today's trading day.
Downloads CSV data from Google Drive, generates JSON snapshots, and processes them through a local LLM.
"""

import os
import time
import yaml
import json
import requests
import gdown
import sys
import shutil
import subprocess
import pandas as pd
from datetime import datetime, timedelta
from orchestrator import generate_snapshot
from modules.loader import load_nq_csv

# Generate time slices every 5 minutes from 09:30 to 16:00
start_time = datetime.strptime("09:30", "%H:%M").time()
end_time = datetime.strptime("16:00", "%H:%M").time()
TIMES = []
current = start_time
while current <= end_time:
    TIMES.append(current.strftime('%H:%M'))
    current_dt = datetime.combine(datetime.today(), current)
    current_dt += timedelta(minutes=5)
    current = current_dt.time()

# Configuration
WORKING_FOLDER = os.path.expanduser("~/rockit_working_folder")
TODAY_FOLDER = os.path.join(WORKING_FOLDER, "today")
CSV_FOLDER = os.path.join(WORKING_FOLDER, "csvs")

# Google Drive URLs
CSV_URLS = {
    'nq': 'https://drive.google.com/uc?id=17pcZ1QKq-XTf0WKCv8cG32_MhcJpO9Sg&export=download',
    'es': 'https://drive.google.com/uc?id=1tUe5jFHbPUF0IG7vnVo1rv9ARXRMFDoj&export=download',
    'ym': 'https://drive.google.com/uc?id=1CWh3hLNnZRjkfThbLRCqJphZQqtjEKoI&export=download'
}

# LLM settings
LLM_URL = "http://localhost:8001/v1/chat/completions"
PROMPT_PATH = "training/prompts/inference-json.md"

def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

def validate_csv(file_path, symbol):
    """Validate if CSV has today's data and reasonable time range."""
    if not os.path.exists(file_path):
        print(f"WARNING: {symbol} CSV not found")
        return False

    try:
        df = pd.read_csv(file_path, skiprows=1, parse_dates=['timestamp'], date_format='%Y-%m-%dT%H:%M:%S.%f')
        df.set_index('timestamp', inplace=True)
        today = datetime.now().strftime("%Y-%m-%d")
        df_today = df[df.index.date == pd.to_datetime(today).date()]
        if df_today.empty:
            print(f"WARNING: {symbol} CSV missing today's date ({today})")
            return False

        min_time = df_today.index.min().time()
        max_time = df_today.index.max().time()
        expected_min = pd.to_datetime("09:30").time()
        current_time = datetime.now().time()

        if min_time > expected_min:
            print(f"WARNING: {symbol} CSV starts at {min_time}, expected 09:30")
        if max_time < pd.to_datetime("09:30").time():
            print(f"WARNING: {symbol} CSV ends too early at {max_time}")
        else:
            time_diff = (datetime.combine(datetime.today(), current_time) - datetime.combine(datetime.today(), max_time)).total_seconds() / 60
            if time_diff > 10:  # Allow 10 min lag
                print(f"WARNING: {symbol} CSV last data at {max_time}, {time_diff:.1f} min behind current time")

        # Check for 5-min intervals (approximate)
        timestamps = df_today.index
        intervals = [(t2 - t1).total_seconds() / 60 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
        avg_interval = sum(intervals) / len(intervals) if intervals else 0
        if abs(avg_interval - 5) > 1:
            print(f"WARNING: {symbol} CSV average interval {avg_interval:.1f} min, expected ~5 min")

        return True
    except Exception as e:
        print(f"WARNING: Error validating {symbol} CSV: {e}")
        return False

def download_csvs():
    """Download CSV files from Google Drive."""
    # Clear existing files
    if os.path.exists(CSV_FOLDER):
        for f in os.listdir(CSV_FOLDER):
            os.remove(os.path.join(CSV_FOLDER, f))
    os.makedirs(CSV_FOLDER, exist_ok=True)
    for symbol, url in CSV_URLS.items():
        output_path = os.path.join(CSV_FOLDER, f"{symbol.upper()}_Minute_5_CME US Index Futures ETH_VWAP_Tick.csv")
        print(f"Downloading {symbol} to {output_path}")
        gdown.download(url, output_path, quiet=False)
        validate_csv(output_path, symbol)

def generate_snapshots_for_time(current_time):
    """Generate JSON snapshot for a specific time if it doesn't exist."""
    snapshot_path = os.path.join(TODAY_FOLDER, f"slice-{current_time.replace(':', '')}.json")
    if os.path.exists(snapshot_path):
        print(f"Snapshot for {current_time} already exists, skipping")
        return snapshot_path

    # Load base config and modify
    config = load_config()
    today = datetime.now().strftime("%Y-%m-%d")
    config['session_date'] = today
    config['current_time'] = current_time
    config['output_dir'] = TODAY_FOLDER
    config['csv_paths'] = {
        'nq': os.path.join(CSV_FOLDER, "NQ_Minute_5_CME US Index Futures ETH_VWAP_Tick.csv"),
        'es': os.path.join(CSV_FOLDER, "ES_Minute_5_CME US Index Futures ETH_VWAP_Tick.csv"),
        'ym': os.path.join(CSV_FOLDER, "YM_Minute_5_CME US Index Futures ETH_VWAP_Tick.csv")
    }

    # Generate snapshot
    snapshot = generate_snapshot(config)

    # Rename the file
    generated_path = os.path.join(TODAY_FOLDER, f"{today}_{current_time.replace(':', '')}.json")
    if os.path.exists(generated_path):
        os.rename(generated_path, snapshot_path)
    else:
        # Save manually if not saved
        with open(snapshot_path, 'w') as f:
            json.dump(snapshot, f, indent=2)
    print(f"Generated snapshot for {current_time}")
    return snapshot_path

def load_prompt():
    with open(PROMPT_PATH, 'r') as f:
        return f.read().strip()

def query_llm(system_prompt, user_message):
    payload = {
        "model": "/model",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.0,
        "max_tokens": 2048
    }
    response = requests.post(LLM_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage", {})
    return content, usage

def get_processed_times():
    """Get set of already processed time slots from today.jsonl."""
    jsonl_path = os.path.join(WORKING_FOLDER, "today.jsonl")
    processed = set()
    if not os.path.exists(jsonl_path):
        return processed
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    entry = json.loads(line)
                    time_slot = entry["input"].get("current_et_time")
                    if time_slot:
                        processed.add(time_slot)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num} in {jsonl_path}: {e}")
                    continue
    print(f"get_processed_times: found {len(processed)} processed times: {sorted(processed)}")
    return processed

def process_snapshots(available_times):
    """Process all snapshots through LLM and create jsonl and tldr."""
    system_prompt = load_prompt()
    jsonl_path = os.path.join(WORKING_FOLDER, "today.jsonl")
    tldr_path = os.path.join(WORKING_FOLDER, "today-tldr.log")
    processed_times = get_processed_times()

    print(f"process_snapshots: available_times={available_times}, processed_times={processed_times}")

    with open(jsonl_path, 'a') as jsonl_f, open(tldr_path, 'a') as tldr_f:
        for time_slot in available_times:
            if time_slot in processed_times:
                print(f"Time slot {time_slot} already processed, skipping")
                continue
            print(f"Processing time_slot: {time_slot}")
            snapshot_path = os.path.join(TODAY_FOLDER, f"slice-{time_slot.replace(':', '')}.json")
            if not os.path.exists(snapshot_path):
                print(f"Snapshot for {time_slot} not found, skipping")
                continue

            with open(snapshot_path, 'r') as f:
                snapshot = json.load(f)

            # Clean snapshot (remove not_available)
            def clean(obj):
                if isinstance(obj, dict):
                    return {k: clean(v) for k, v in obj.items() if v != "not_available"}
                elif isinstance(obj, list):
                    return [clean(item) for item in obj if item != "not_available"]
                return obj
            snapshot = clean(snapshot)

            user_msg = json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n\nrun trainer"
            start_time = time.time()
            print(f"Starting LLM query for {time_slot}")
            output, usage = query_llm(system_prompt, user_msg)
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"Finished LLM query for {time_slot}, elapsed {elapsed:.2f}s")
            total_tokens = usage.get("total_tokens", 0)
            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0

            # Log LLM output
            llm_log_path = os.path.join(WORKING_FOLDER, "llm.log")
            with open(llm_log_path, 'a') as f:
                f.write(f"--- {time_slot} ---\n{output}\n\n")

            # Write jsonl
            try:
                output_json = json.loads(output)
                entry = {"input": snapshot, "output": output_json}
            except json.JSONDecodeError as e:
                print(f"Failed to parse LLM output as JSON for {time_slot}: {e}")
                print(f"LLM output: {output}")
                # Use raw output as string
                entry = {"input": snapshot, "output": output}
            jsonl_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            jsonl_f.flush()
            print(f"Added entry for {time_slot} to today.jsonl")

            # Write tldr
            today = datetime.now().strftime("%Y-%m-%d")
            if isinstance(entry["output"], dict):
                tldr_content = json.dumps(entry["output"], indent=2)
            else:
                tldr_content = f"[JSON PARSE ERROR]\n{entry['output']}"
            tldr_f.write(f"--- {today} {time_slot} ---\n{tldr_content}\nElapsed: {elapsed:.2f}s, Tokens: {total_tokens}, Tokens/sec: {tokens_per_sec:.2f}\n\n")
            tldr_f.flush()
            print(f"Added entry for {time_slot} to today-tldr.log")

def get_available_times():
    """Get the list of times available in the current CSV data."""
    nq_path = os.path.join(CSV_FOLDER, "NQ_Minute_5_CME US Index Futures ETH_VWAP_Tick.csv")
    if not os.path.exists(nq_path):
        return []
    df_extended, df_current = load_nq_csv(nq_path, datetime.now().strftime("%Y-%m-%d"))
    if df_current.empty:
        return []
    max_time = df_current.index.max().strftime('%H:%M')
    return [t for t in TIMES if t <= max_time]

def upload_today_jsonl():
    """Upload the today.jsonl file to GitHub repository."""
    date = datetime.now().strftime("%Y-%m-%d")
    repo_dir = os.path.join(WORKING_FOLDER, "rockit-data-feed")
    token = os.environ.get('GIT_PAT')
    if not token:
        print("GIT_PAT environment variable not set")
        return

    if not os.path.exists(repo_dir):
        try:
            subprocess.run(["git", "clone", f"https://{token}@github.com/LePhanFF/RockitDataFeed.git", repo_dir], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repo: {e}")
            return
    else:
        try:
            subprocess.run(["git", "pull"], cwd=repo_dir, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to pull repo: {e}")
            return

    target_dir = os.path.join(repo_dir, "local-analysis-format")
    os.makedirs(target_dir, exist_ok=True)

    source = os.path.join(WORKING_FOLDER, "today.jsonl")
    if not os.path.exists(source):
        print(f"Source file {source} does not exist")
        return

    target = os.path.join(target_dir, f"{date}.jsonl")
    shutil.copy(source, target)

    try:
        subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
        # Check if there are changes to commit
        result = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo_dir)
        if result.returncode != 0:  # There are staged changes
            subprocess.run(["git", "commit", "-m", f"Add {date}.jsonl"], cwd=repo_dir, check=True)
            subprocess.run(["git", "push"], cwd=repo_dir, check=True)
            print(f"Successfully uploaded {date}.jsonl to GitHub")
        else:
            print(f"No changes to {date}.jsonl, skipping commit")
    except subprocess.CalledProcessError as e:
        print(f"Failed to commit/push: {e}")

def main():
    # Wipe if requested
    if len(sys.argv) > 1 and sys.argv[1] == "wipe":
        if os.path.exists(WORKING_FOLDER):
            for item in os.listdir(WORKING_FOLDER):
                if item == "rockit-data-feed":
                    continue  # Skip the repository directory
                item_path = os.path.join(WORKING_FOLDER, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        print("Wiped working folder")

    # Create directories
    os.makedirs(WORKING_FOLDER, exist_ok=True)
    os.makedirs(TODAY_FOLDER, exist_ok=True)

    print("Starting analyze-today.py")

    while True:
        try:
            # Download CSVs
            download_csvs()

            # Get available times
            available_times = get_available_times()
            if not available_times:
                print("No data available yet, waiting...")
                time.sleep(120)
                continue

            print(f"Available times: {available_times}")
            print(f"Available times up to {available_times[-1]}")

            # Get processed times
            processed_times = get_processed_times()
            print(f"Processed times: {processed_times}")

            # Generate snapshots for available times
            for t in available_times:
                generate_snapshots_for_time(t)

            # Process snapshots
            process_snapshots(available_times)

            upload_today_jsonl()

            print("Cycle complete, sleeping for 2 minutes")
            time.sleep(120)  # 2 minutes

        except Exception as e:
            print(f"Error in cycle: {e}")
            time.sleep(60)  # Wait 1 minute on error

if __name__ == "__main__":
    main()


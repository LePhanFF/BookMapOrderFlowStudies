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
import numpy as np
import traceback
import concurrent.futures
import argparse
from datetime import datetime, timedelta
from orchestrator import generate_snapshot
from modules.loader import load_nq_csv

# Set Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.path.dirname(__file__), 'google-rockitsa-key.json')
from google.cloud import storage

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
CSV_FOLDER = os.path.join(WORKING_FOLDER, "csvs")
target_date = None  # Global variable for the current processing date
TODAY_FOLDER = None  # Global variable for the current output folder

# Google Drive URLs
CSV_URLS = {
    'nq': 'https://drive.google.com/uc?id=17pcZ1QKq-XTf0WKCv8cG32_MhcJpO9Sg&export=download',
    'es': 'https://drive.google.com/uc?id=1tUe5jFHbPUF0IG7vnVo1rv9ARXRMFDoj&export=download',
    'ym': 'https://drive.google.com/uc?id=1CWh3hLNnZRjkfThbLRCqJphZQqtjEKoI&export=download'
}

# LLM settings
LLM_URL = "http://localhost:8356/v1/chat/completions"
PROMPT_PATH = "training/prompts/inference-json.md"

def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

def validate_csv(file_path, symbol):
    """Validate if CSV has target_date data and reasonable time range."""
    if not os.path.exists(file_path):
        print(f"WARNING: {symbol} CSV not found")
        return False

    try:
        df = pd.read_csv(file_path, skiprows=1, parse_dates=['timestamp'], date_format='%Y-%m-%dT%H:%M:%S.%f')
        df.set_index('timestamp', inplace=True)
        df_target = df[df.index.date == pd.to_datetime(target_date).date()]
        if df_target.empty:
            print(f"WARNING: {symbol} CSV missing target date ({target_date})")
            return False

        min_time = df_target.index.min().time()
        max_time = df_target.index.max().time()
        expected_min = pd.to_datetime("09:30").time()

        if min_time > expected_min:
            print(f"WARNING: {symbol} CSV starts at {min_time}, expected 09:30")
        if max_time < pd.to_datetime("09:30").time():
            print(f"WARNING: {symbol} CSV ends too early at {max_time}")

        # Check for 5-min intervals (approximate)
        timestamps = df_target.index
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
    config['session_date'] = target_date
    config['current_time'] = current_time
    config['output_dir'] = TODAY_FOLDER
    print(f"config output_dir: {config['output_dir']}")
    config['csv_paths'] = {
        'nq': os.path.join(CSV_FOLDER, "NQ_Minute_5_CME US Index Futures ETH_VWAP_Tick.csv"),
        'es': os.path.join(CSV_FOLDER, "ES_Minute_5_CME US Index Futures ETH_VWAP_Tick.csv"),
        'ym': os.path.join(CSV_FOLDER, "YM_Minute_5_CME US Index Futures ETH_VWAP_Tick.csv")
    }

    # Generate snapshot
    snapshot = generate_snapshot(config)

    # Rename the file
    generated_path = os.path.join(TODAY_FOLDER, f"{target_date}_{current_time.replace(':', '')}.json")
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
        "model": "GadflyII/GLM-4.7-Flash-NVFP4",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "response_format": { "type": "json_object" },
        "max_tokens": 2048
    }
    response = requests.post(LLM_URL, json=payload, timeout=240)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage", {})
    return content, usage

def get_processed_times():
    """Get set of already processed time slots from target_date.jsonl."""
    jsonl_path = os.path.join(WORKING_FOLDER, f"{target_date}.jsonl")
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

def process_single_snapshot(time_slot, system_prompt):
    """Process a single snapshot through LLM."""
    snapshot_path = os.path.join(TODAY_FOLDER, f"slice-{time_slot.replace(':', '')}.json")
    if not os.path.exists(snapshot_path):
        print(f"Snapshot for {time_slot} not found, skipping")
        return None

    with open(snapshot_path, 'r') as f:
        snapshot = json.load(f)

    # Clean snapshot (remove not_available from dicts only, convert numpy types)
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items() if v != "not_available"}
        elif isinstance(obj, list):
            return [clean(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj
    snapshot = clean(snapshot)

    user_msg = json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n\nrun trainer"
    start_time = time.time()
    print(f"Starting LLM query for {time_slot}")

    # Retry logic for incomplete responses
    max_retries = 2
    output_json = None
    for attempt in range(max_retries):
        output, usage = query_llm(system_prompt, user_msg)
        try:
            output_json = json.loads(output)
            # Check for incomplete response indicators
            if (output_json.get("one_liner") == "Decoding market stream" or
                not output_json.get("day_type_reasoning", [])):
                if attempt < max_retries - 1:
                    print(f"Incomplete LLM response for {time_slot}, retrying (attempt {attempt + 1})")
                    continue
                else:
                    print(f"LLM response still incomplete after {max_retries} attempts for {time_slot}")
            break  # Valid response or max retries reached
        except json.JSONDecodeError:
            print(f"Invalid JSON from LLM for {time_slot}, not retrying")
            break  # Will handle as string below

    end_time = time.time()
    elapsed = end_time - start_time
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
    print(f"Finished LLM query for {time_slot}, elapsed {elapsed:.2f}s, Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Total tokens: {total_tokens}")
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0

    # Log LLM output
    llm_log_path = os.path.join(WORKING_FOLDER, "llm.log")
    with open(llm_log_path, 'a', encoding='utf-8') as f:
        f.write(f"--- {time_slot} ---\n{output}\n\n")

    # Prepare entry
    if output_json is not None:
        entry = {"input": snapshot, "output": output_json}
    else:
        print(f"Failed to parse LLM output as JSON for {time_slot}")
        print(f"LLM output: {output}")
        # Use raw output as string
        entry = {"input": snapshot, "output": output}

    # Prepare tldr content
    if isinstance(entry["output"], dict):
        tldr_content = json.dumps(entry["output"], indent=2)
    else:
        tldr_content = f"[JSON PARSE ERROR]\n{entry['output']}"

    return time_slot, entry, tldr_content, elapsed, total_tokens, tokens_per_sec

def process_snapshots(available_times):
    """Process all snapshots through LLM and create jsonl and tldr."""
    system_prompt = load_prompt()
    jsonl_path = os.path.join(WORKING_FOLDER, f"{target_date}.jsonl")
    tldr_path = os.path.join(WORKING_FOLDER, f"{target_date}-tldr.log")
    processed_times = get_processed_times()

    print(f"process_snapshots: available_times={available_times}, processed_times={processed_times}")

    to_process = [time_slot for time_slot in available_times if time_slot not in processed_times]
    if not to_process:
        print("No new time slots to process")
        return

    # Open files for appending
    with open(jsonl_path, 'a', encoding='utf-8') as jsonl_f, open(tldr_path, 'a', encoding='utf-8') as tldr_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = {executor.submit(process_single_snapshot, time_slot, system_prompt): time_slot for time_slot in to_process}
            for future in concurrent.futures.as_completed(futures):
                time_slot = futures[future]
                try:
                    result = future.result()
                    if result:
                        time_slot, entry, tldr_content, elapsed, total_tokens, tokens_per_sec = result
                        jsonl_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        jsonl_f.flush()
                        print(f"Added entry for {time_slot} to today.jsonl")

                        tldr_f.write(f"--- {target_date} {time_slot} ---\n{tldr_content}\nElapsed: {elapsed:.2f}s, Tokens: {total_tokens}, Tokens/sec: {tokens_per_sec:.2f}\n\n")
                        tldr_f.flush()
                        print(f"Added entry for {time_slot} to today-tldr.log")

                        # Incremental upload after each entry
                        upload_today_jsonl()
                except Exception as e:
                    print(f"Error processing {time_slot}: {e}")
                    traceback.print_exc()

    # Sort the entire jsonl file by time_slot
    all_entries = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                time_slot = entry["input"].get("current_et_time")
                all_entries.append((time_slot, line.strip()))
    all_entries.sort(key=lambda x: x[0])
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for _, line in all_entries:
            f.write(line + "\n")
    print("Sorted today.jsonl by time")

    # Final check: validate all entries
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON in sorted today.jsonl line {line_num}: {e}")
    print("Final validation of today.jsonl completed")

    # Final upload after sorting
    upload_today_jsonl()

def get_available_times():
    """Get the list of times available in the current CSV data."""
    nq_path = os.path.join(CSV_FOLDER, "NQ_Minute_5_CME US Index Futures ETH_VWAP_Tick.csv")
    if not os.path.exists(nq_path):
        return []
    df_extended, df_current = load_nq_csv(nq_path, target_date)
    if df_current.empty:
        return []
    max_time = df_current.index.max().strftime('%H:%M')
    return [t for t in TIMES if t <= max_time]

def upload_today_jsonl():
    """Upload the target_date.jsonl file to Google Cloud Storage."""
    source = os.path.join(WORKING_FOLDER, f"{target_date}.jsonl")
    if not os.path.exists(source):
        print(f"Source file {source} does not exist")
        return

    try:
        client = storage.Client()
        bucket = client.bucket('rockit-data')
        blob = bucket.blob(f"{target_date}.jsonl")
        blob.upload_from_filename(source)
        print(f"Successfully uploaded {target_date}.jsonl to Google Cloud Storage")
    except Exception as e:
        print(f"Failed to upload {target_date}.jsonl to GCS: {e}")

def repair_today_jsonl():
    """Repair incomplete entries in today's jsonl file."""
    global target_date, TODAY_FOLDER
    target_date = datetime.now().strftime("%Y-%m-%d")
    TODAY_FOLDER = os.path.join(WORKING_FOLDER, target_date)

    jsonl_path = os.path.join(WORKING_FOLDER, f"{target_date}.jsonl")
    tldr_path = os.path.join(WORKING_FOLDER, f"{target_date}-tldr.log")

    if not os.path.exists(jsonl_path):
        print(f"No jsonl file found for today: {jsonl_path}")
        return

    # Read all entries
    entries = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON in line {line_num}: {e}")

    # Find bad entries
    bad_time_slots = []
    for entry in entries:
        output = entry.get("output", {})
        time_slot = entry["input"].get("current_et_time")
        print(f"Checking entry for {time_slot}: output type {type(output)}")
        is_bad = False
        if isinstance(output, dict):
            one_liner = output.get("one_liner")
            day_type_reasoning = output.get("day_type_reasoning", [])
            confidence = output.get("confidence")
            print(f"  one_liner: {repr(one_liner)}, day_type_reasoning: {day_type_reasoning}, confidence: {repr(confidence)}")
            if (one_liner == "Decoding market stream" or
                not day_type_reasoning or
                confidence == "0%"):
                is_bad = True
        elif isinstance(output, str):
            # Strings indicate JSON parse failure, mark as bad
            print(f"  -> Output is string (JSON parse error), marking as bad")
            is_bad = True

        if is_bad:
            print(f"  -> Bad entry found")
            if time_slot:
                bad_time_slots.append(time_slot)

    if not bad_time_slots:
        print("No bad entries found to repair")
        return

    print(f"Found {len(bad_time_slots)} bad entries to repair: {bad_time_slots}")

    # Process bad time slots
    system_prompt = load_prompt()
    repaired_entries = {}
    for time_slot in bad_time_slots:
        print(f"Repairing {time_slot}")
        result = process_single_snapshot(time_slot, system_prompt)
        if result:
            time_slot, entry, tldr_content, elapsed, total_tokens, tokens_per_sec = result
            repaired_entries[time_slot] = entry

    # Update entries
    for i, entry in enumerate(entries):
        time_slot = entry["input"].get("current_et_time")
        if time_slot in repaired_entries:
            entries[i] = repaired_entries[time_slot]
            print(f"Updated entry for {time_slot}")

    # Sort entries by time
    entries.sort(key=lambda e: e["input"].get("current_et_time", ""))

    # Write back jsonl
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Update tldr by rewriting it
    with open(tldr_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            time_slot = entry["input"].get("current_et_time")
            output = entry.get("output", {})
            if isinstance(output, dict):
                tldr_content = json.dumps(output, indent=2)
            else:
                tldr_content = f"[JSON PARSE ERROR]\n{output}"
            f.write(f"--- {target_date} {time_slot} ---\n{tldr_content}\n\n")

    print(f"Repaired {len(bad_time_slots)} entries")

def main():
    global target_date, TODAY_FOLDER

    # Handle wipe argument before parsing
    wipe = False
    if len(sys.argv) > 1 and sys.argv[1] == "wipe":
        wipe = True
        sys.argv.pop(1)  # Remove 'wipe' from argv so argparse doesn't see it

    parser = argparse.ArgumentParser(description="Process trading data for specified dates or today.")
    parser.add_argument('--dates', type=str, help='Comma-separated list of dates in yyyy.mm.dd format (e.g., 2025.01.01,2025.01.02)')
    parser.add_argument('--repair', action='store_true', help='Repair incomplete entries in today\'s jsonl file')
    args = parser.parse_args()
    print(f"Parsed args.dates: {args.dates}, args.repair: {args.repair}")

    if args.repair:
        repair_today_jsonl()
        return

    dates_list = []
    if args.dates:
        dates_list = [d.strip().replace('.', '-') for d in args.dates.split(',')]
        print(f"Parsed dates from --dates: {dates_list}")
    else:
        dates_list = [datetime.now().strftime("%Y-%m-%d")]
        print(f"No --dates provided, defaulting to today: {dates_list}")

    # Wipe if requested
    if wipe:
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

    print("Starting analyze-today.py")

    for target_date in dates_list:
        TODAY_FOLDER = os.path.join(WORKING_FOLDER, target_date)
        os.makedirs(TODAY_FOLDER, exist_ok=True)
        print(f"TODAY_FOLDER: {TODAY_FOLDER}")

        print(f"Processing date: {target_date}")

        is_today = target_date == datetime.now().strftime("%Y-%m-%d")
        print(f"is_today: {is_today}")

        if is_today:
            while True:
                try:
                    # Download CSVs
                    print("Starting download_csvs")
                    try:
                        download_csvs()
                    except Exception as e:
                        print(f"Error in download_csvs: {e}")
                        traceback.print_exc()
                        raise

                    # Get available times
                    print("Starting get_available_times")
                    try:
                        available_times = get_available_times()
                        if not available_times:
                            print("No data available yet, waiting...")
                            time.sleep(5)
                            continue

                        print(f"Available times: {available_times}")
                        if available_times:
                            print(f"Available times up to {available_times[-1]}")
                        else:
                            print("Available times up to N/A")
                    except Exception as e:
                        print(f"Error in get_available_times: {e}")
                        traceback.print_exc()
                        raise

                    # Get processed times
                    print("Starting get_processed_times")
                    try:
                        processed_times = get_processed_times()
                        print(f"Processed times: {processed_times}")
                    except Exception as e:
                        print(f"Error in get_processed_times: {e}")
                        traceback.print_exc()
                        raise

                    # Generate snapshots for available times
                    print("Starting generate_snapshots_for_time")
                    try:
                        for t in available_times:
                            generate_snapshots_for_time(t)
                    except Exception as e:
                        print(f"Error in generate_snapshots_for_time: {e}")
                        traceback.print_exc()
                        raise

                    # Process snapshots
                    print("Starting process_snapshots")
                    try:
                        process_snapshots(available_times)
                    except Exception as e:
                        print(f"Error in process_snapshots: {e}")
                        traceback.print_exc()
                        raise

                    print("Starting upload_today_jsonl")
                    try:
                        upload_today_jsonl()
                    except Exception as e:
                        print(f"Error in upload_today_jsonl: {e}")
                        traceback.print_exc()
                        raise

                    print("Cycle complete, sleeping for 2 minutes")
                    time.sleep(120)  # 2 minutes

                except Exception as e:
                    print(f"Error in cycle: {e}")
                    traceback.print_exc()
                    time.sleep(60)  # Wait 1 minute on error
        else:
            try:
                # Download CSVs
                print("Starting download_csvs")
                try:
                    download_csvs()
                except Exception as e:
                    print(f"Error in download_csvs: {e}")
                    traceback.print_exc()
                    raise

                # Get available times
                print("Starting get_available_times")
                try:
                    available_times = get_available_times()
                    if not available_times:
                        print(f"No data available for {target_date}, skipping")
                        continue

                    print(f"Available times: {available_times}")
                    if available_times:
                        print(f"Available times up to {available_times[-1]}")
                    else:
                        print("Available times up to N/A")
                except Exception as e:
                    print(f"Error in get_available_times: {e}")
                    traceback.print_exc()
                    raise

                # Get processed times
                print("Starting get_processed_times")
                try:
                    processed_times = get_processed_times()
                    print(f"Processed times: {processed_times}")
                except Exception as e:
                    print(f"Error in get_processed_times: {e}")
                    traceback.print_exc()
                    raise

                # Generate snapshots for available times
                print("Starting generate_snapshots_for_time")
                try:
                    for t in available_times:
                        generate_snapshots_for_time(t)
                except Exception as e:
                    print(f"Error in generate_snapshots_for_time: {e}")
                    traceback.print_exc()
                    raise

                # Process snapshots
                print("Starting process_snapshots")
                try:
                    process_snapshots(available_times)
                except Exception as e:
                    print(f"Error in process_snapshots: {e}")
                    traceback.print_exc()
                    raise

                print("Starting upload_today_jsonl")
                try:
                    upload_today_jsonl()
                except Exception as e:
                    print(f"Error in upload_today_jsonl: {e}")
                    traceback.print_exc()
                    raise

                print(f"Processing complete for {target_date}")

            except Exception as e:
                print(f"Error processing {target_date}: {e}")
                traceback.print_exc()
if __name__ == "__main__":
    main()

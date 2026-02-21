"""
Merge CSV data from multiple sources into combined files.

Merges:
  csv/partial-data/{INSTRUMENT}_Volumetric_1.csv  (Feb 2025 - Feb 2026, sparser)
  csv/{INSTRUMENT}_Volumetric_1.csv               (Aug 2025 - Feb 2026, denser)

Strategy:
  1. Load both files for each instrument
  2. For the OVERLAP period (Aug 2025+), prefer the main (denser) data
  3. For the NON-OVERLAP period (Feb-Aug 2025), use partial-data
  4. Deduplicate by timestamp
  5. Sort by timestamp ascending
  6. Save to csv/combined/{INSTRUMENT}_Volumetric_1.csv

Usage:
    python scripts/merge_csv_data.py
    python scripts/merge_csv_data.py --instruments NQ ES YM
    python scripts/merge_csv_data.py --dry-run        # Preview only, no write
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd


INSTRUMENTS = ['NQ', 'ES', 'YM']
CSV_DIR = project_root / 'csv'
PARTIAL_DIR = CSV_DIR / 'partial-data'
OUTPUT_DIR = CSV_DIR / 'combined'


def load_csv_raw(path: Path) -> pd.DataFrame:
    """Load a volumetric CSV, skipping the schema header if present."""
    # Check if first line is a schema comment (e.g. #NinjaDataExport/v2.3)
    with open(path, 'r') as f:
        first_line = f.readline().strip()

    skip = 1 if first_line.startswith('#') else 0
    df = pd.read_csv(path, skiprows=skip)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def merge_instrument(instrument: str, dry_run: bool = False) -> dict:
    """Merge data for a single instrument. Returns stats dict."""
    main_path = CSV_DIR / f'{instrument}_Volumetric_1.csv'
    partial_path = PARTIAL_DIR / f'{instrument}_Volumetric_1.csv'

    stats = {'instrument': instrument, 'main_rows': 0, 'partial_rows': 0,
             'combined_rows': 0, 'unique_dates': 0, 'date_range': ''}

    # Load available files
    dfs = []

    if partial_path.exists():
        df_partial = load_csv_raw(partial_path)
        stats['partial_rows'] = len(df_partial)
        stats['partial_range'] = f"{df_partial['timestamp'].min()} -> {df_partial['timestamp'].max()}"
        dfs.append(('partial', df_partial))
        print(f"  Partial: {len(df_partial):>9,} rows  "
              f"({df_partial['timestamp'].min().date()} -> {df_partial['timestamp'].max().date()})")
    else:
        print(f"  Partial: NOT FOUND at {partial_path}")

    if main_path.exists():
        df_main = load_csv_raw(main_path)
        stats['main_rows'] = len(df_main)
        stats['main_range'] = f"{df_main['timestamp'].min()} -> {df_main['timestamp'].max()}"
        dfs.append(('main', df_main))
        print(f"  Main:    {len(df_main):>9,} rows  "
              f"({df_main['timestamp'].min().date()} -> {df_main['timestamp'].max().date()})")
    else:
        print(f"  Main: NOT FOUND at {main_path}")

    if not dfs:
        print(f"  ERROR: No data files found for {instrument}")
        return stats

    if len(dfs) == 1:
        # Only one source, just use it
        combined = dfs[0][1]
        print(f"  Only one source available, using {dfs[0][0]} data directly")
    else:
        # Both sources available — merge with dedup
        df_partial = dfs[0][1]
        df_main = dfs[1][1]

        # Find the overlap boundary: where main data starts
        main_start = df_main['timestamp'].min()
        print(f"  Overlap starts: {main_start.date()}")

        # Take partial-data BEFORE main starts (the unique early data)
        early_data = df_partial[df_partial['timestamp'] < main_start]
        print(f"  Early data (partial only): {len(early_data):>9,} rows  "
              f"({early_data['timestamp'].min().date()} -> {early_data['timestamp'].max().date()})"
              if len(early_data) > 0 else "  Early data: 0 rows")

        # For overlap period, prefer main (denser)
        print(f"  Overlap data (using main): {len(df_main):>9,} rows")

        # Also check if partial has data AFTER main ends
        main_end = df_main['timestamp'].max()
        late_data = df_partial[df_partial['timestamp'] > main_end]
        if len(late_data) > 0:
            print(f"  Late data (partial only):  {len(late_data):>9,} rows  "
                  f"({late_data['timestamp'].min().date()} -> {late_data['timestamp'].max().date()})")

        # Combine: early_partial + main + late_partial
        parts = [early_data, df_main]
        if len(late_data) > 0:
            parts.append(late_data)

        combined = pd.concat(parts, ignore_index=True)

        # Final dedup on timestamp (in case of any edge overlap)
        before_dedup = len(combined)
        combined = combined.drop_duplicates(subset=['timestamp'], keep='last')
        if before_dedup != len(combined):
            print(f"  Dedup removed: {before_dedup - len(combined)} duplicate rows")

    # Sort by timestamp
    combined = combined.sort_values('timestamp').reset_index(drop=True)

    stats['combined_rows'] = len(combined)
    if 'session_date' in combined.columns:
        unique_dates = combined['session_date'].nunique()
    else:
        unique_dates = combined['timestamp'].dt.date.nunique()
    stats['unique_dates'] = unique_dates
    stats['date_range'] = f"{combined['timestamp'].min().date()} -> {combined['timestamp'].max().date()}"

    print(f"  Combined: {len(combined):>9,} rows, {unique_dates} unique dates")
    print(f"  Range:    {combined['timestamp'].min().date()} -> {combined['timestamp'].max().date()}")

    if not dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f'{instrument}_Volumetric_1.csv'
        combined.to_csv(output_path, index=False)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Saved:    {output_path} ({size_mb:.1f} MB)")
    else:
        print(f"  DRY RUN — would save to {OUTPUT_DIR / f'{instrument}_Volumetric_1.csv'}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Merge partial + main CSV data')
    parser.add_argument('--instruments', nargs='+', default=INSTRUMENTS,
                        help=f'Instruments to merge (default: {INSTRUMENTS})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview merge without writing files')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  CSV DATA MERGE UTILITY")
    print(f"{'='*70}")
    print(f"  Main dir:    {CSV_DIR}")
    print(f"  Partial dir: {PARTIAL_DIR}")
    print(f"  Output dir:  {OUTPUT_DIR}")
    print(f"  Instruments: {args.instruments}")
    if args.dry_run:
        print(f"  MODE: DRY RUN (no files written)")
    print()

    all_stats = []
    for instrument in args.instruments:
        print(f"--- {instrument} ---")
        stats = merge_instrument(instrument, dry_run=args.dry_run)
        all_stats.append(stats)
        print()

    # Summary
    print(f"{'='*70}")
    print(f"  MERGE SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Instrument':>10s}  {'Partial':>10s}  {'Main':>10s}  {'Combined':>10s}  {'Dates':>6s}  Range")
    print(f"  {'-'*10:>10s}  {'-'*10:>10s}  {'-'*10:>10s}  {'-'*10:>10s}  {'-'*6:>6s}  {'-'*30}")
    for s in all_stats:
        print(f"  {s['instrument']:>10s}  {s['partial_rows']:>10,}  {s['main_rows']:>10,}  "
              f"{s['combined_rows']:>10,}  {s['unique_dates']:>6,}  {s.get('date_range', 'N/A')}")

    total_combined = sum(s['combined_rows'] for s in all_stats)
    print(f"\n  Total combined rows: {total_combined:,}")

    if not args.dry_run:
        print(f"\n  Output files written to: {OUTPUT_DIR}/")
        print(f"  To use in backtest: python scripts/run_backtest.py --csv-dir csv/combined")
    print()


if __name__ == '__main__':
    main()

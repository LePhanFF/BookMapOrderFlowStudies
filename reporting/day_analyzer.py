"""
Day Type Analyzer — Reports per-session day type classification and confidence.

Shows how well each session matches the Dalton day type checklists,
what the final classification was, and which checklist items matched.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import pandas as pd

from strategy.day_confidence import DayTypeConfidenceScorer, DayTypeConfidence


@dataclass
class SessionAnalysis:
    """Analysis of a single session's day type characteristics."""
    session_date: str
    ib_high: float
    ib_low: float
    ib_range: float
    session_open: float
    session_close: float
    session_range: float
    final_confidence: Optional[DayTypeConfidence] = None
    classified_type: str = ''
    classified_confidence: float = 0.0


def analyze_sessions(df: pd.DataFrame, verbose: bool = True) -> List[SessionAnalysis]:
    """
    Analyze all sessions and report day type confidence scoring.

    Returns a list of SessionAnalysis objects for each session.
    """
    from config.constants import IB_BARS_1MIN

    if 'timestamp' in df.columns:
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'session_date' not in df.columns:
        df['session_date'] = df['timestamp'].dt.date

    sessions = sorted(df['session_date'].unique())
    results = []

    for session_date in sessions:
        session_df = df[df['session_date'] == session_date]

        if len(session_df) < IB_BARS_1MIN:
            continue

        ib_df = session_df.head(IB_BARS_1MIN)
        ib_high = ib_df['high'].max()
        ib_low = ib_df['low'].min()
        ib_range = ib_high - ib_low

        if ib_range <= 0:
            continue

        post_ib = session_df.iloc[IB_BARS_1MIN:]
        if len(post_ib) == 0:
            continue

        # Get ATR if available
        atr = ib_df.iloc[-1].get('atr14', 0.0) if 'atr14' in ib_df.columns else 0.0

        # Run confidence scorer through all post-IB bars
        scorer = DayTypeConfidenceScorer()
        scorer.on_session_start(ib_high, ib_low, ib_range, atr)

        final_confidence = None
        for _, bar in post_ib.iterrows():
            final_confidence = scorer.update(bar, 0)

        session_open = session_df.iloc[0]['open']
        session_close = session_df.iloc[-1]['close']
        session_range = session_df['high'].max() - session_df['low'].min()

        analysis = SessionAnalysis(
            session_date=str(session_date),
            ib_high=ib_high,
            ib_low=ib_low,
            ib_range=ib_range,
            session_open=session_open,
            session_close=session_close,
            session_range=session_range,
            final_confidence=final_confidence,
        )

        if final_confidence:
            analysis.classified_type = final_confidence.best_type
            analysis.classified_confidence = final_confidence.best_confidence

        results.append(analysis)

    return results


def print_day_analysis(analyses: List[SessionAnalysis]):
    """Print a summary table of day type analysis."""
    print(f"\n{'='*120}")
    print(f"  DAY TYPE ANALYSIS — Dalton Checklist Confidence Scoring")
    print(f"{'='*120}")
    print(f"{'Date':>12s}  {'IB Range':>8s}  {'Range':>8s}  {'Close-Open':>10s}  "
          f"{'Best Type':>12s}  {'Conf%':>6s}  "
          f"{'TrBull':>6s}  {'TrBear':>6s}  {'PBull':>6s}  {'PBear':>6s}  {'BDay':>6s}  {'Neut':>6s}")
    print('-' * 120)

    type_counts = {}
    for a in analyses:
        if a.final_confidence is None:
            continue

        c = a.final_confidence
        change = a.session_close - a.session_open

        best_type = a.classified_type
        type_counts[best_type] = type_counts.get(best_type, 0) + 1

        print(f"{a.session_date:>12s}  {a.ib_range:>8.2f}  {a.session_range:>8.2f}  {change:>+10.2f}  "
              f"{best_type:>12s}  {a.classified_confidence*100:>5.1f}%  "
              f"{c.trend_bull*100:>5.1f}%  {c.trend_bear*100:>5.1f}%  "
              f"{c.p_day_bull*100:>5.1f}%  {c.p_day_bear*100:>5.1f}%  "
              f"{c.b_day*100:>5.1f}%  {c.neutral*100:>5.1f}%")

    print('-' * 120)
    print(f"\n--- Day Type Distribution ---")
    for dtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = count / len(analyses) * 100 if analyses else 0
        print(f"  {dtype:>15s}: {count:3d} sessions ({pct:.1f}%)")

    # Show high-confidence sessions
    high_conf = [a for a in analyses if a.classified_confidence >= 0.75]
    medium_conf = [a for a in analyses if 0.5 <= a.classified_confidence < 0.75]
    low_conf = [a for a in analyses if a.classified_confidence < 0.5]

    print(f"\n--- Confidence Distribution ---")
    print(f"  High (>=75%):    {len(high_conf):3d} sessions ({len(high_conf)/len(analyses)*100:.1f}%)")
    print(f"  Medium (50-75%): {len(medium_conf):3d} sessions ({len(medium_conf)/len(analyses)*100:.1f}%)")
    print(f"  Low (<50%):      {len(low_conf):3d} sessions ({len(low_conf)/len(analyses)*100:.1f}%)")
    print(f"{'='*120}")

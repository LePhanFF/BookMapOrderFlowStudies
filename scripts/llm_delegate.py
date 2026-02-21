"""
Local LLM coding delegate.

Sends coding tasks to a local vLLM instance and returns the response.
Used by Claude Code to offload coding, analysis, and review tasks.

Usage:
    python llm_delegate.py --task "Write a function that..."
    python llm_delegate.py --task "Review this code" --files strategy/edge_fade.py
    python llm_delegate.py --task "Analyze these results" --stdin < results.txt
    python llm_delegate.py --task "Fix this bug" --files a.py b.py --context "Error: ..."
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

API_BASE = "http://localhost:8356/v1"
MODEL = "RESMP-DEV/Qwen3-Next-80B-A3B-Instruct-NVFP4"
MAX_TOKENS = 8192
TEMPERATURE = 0.3  # Low temp for coding tasks


def read_files(file_paths: list[str]) -> str:
    """Read files and format them as context."""
    parts = []
    for fp in file_paths:
        p = Path(fp)
        if not p.exists():
            # Try relative to project root
            p = Path(__file__).resolve().parent.parent / fp
        if p.exists():
            content = p.read_text(encoding='utf-8', errors='replace')
            parts.append(f"### File: {fp}\n```\n{content}\n```")
        else:
            parts.append(f"### File: {fp}\n(NOT FOUND)")
    return "\n\n".join(parts)


def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = MAX_TOKENS, temperature: float = TEMPERATURE) -> str:
    """Call the local vLLM API."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        f"{API_BASE}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            return result["choices"][0]["message"]["content"]
    except urllib.error.URLError as e:
        return f"ERROR: Could not reach local LLM at {API_BASE}: {e}"
    except Exception as e:
        return f"ERROR: {e}"


SYSTEM_PROMPT = """You are a senior Python developer working on a NQ/MNQ futures trading backtest system.

Key project context:
- Backtest engine in engine/backtest.py processes 1-min bars session by session
- Strategies in strategy/ extend StrategyBase with on_session_start(), on_bar(), on_session_end()
- Each strategy emits Signal objects with entry/stop/target prices
- Config in config/constants.py, data loading in data/loader.py
- Python 3.11+, pandas, numpy

When writing code:
- Follow existing patterns in the codebase
- Use type hints
- Keep it simple, no over-engineering
- Return complete, working code (not pseudocode)

When reviewing/analyzing:
- Be specific about issues found
- Suggest concrete fixes with code
- Focus on correctness over style"""


def main():
    parser = argparse.ArgumentParser(description='Delegate coding tasks to local LLM')
    parser.add_argument('--task', '-t', required=True, help='Task description')
    parser.add_argument('--files', '-f', nargs='*', default=[], help='Files to include as context')
    parser.add_argument('--context', '-c', default='', help='Additional context string')
    parser.add_argument('--stdin', action='store_true', help='Read additional input from stdin')
    parser.add_argument('--system', '-s', default='', help='Override system prompt')
    parser.add_argument('--max-tokens', type=int, default=MAX_TOKENS, help='Max response tokens')
    parser.add_argument('--temperature', type=float, default=TEMPERATURE, help='Temperature')
    args = parser.parse_args()

    # Build user prompt
    parts = [f"## Task\n{args.task}"]

    if args.files:
        parts.append(f"\n## Files\n{read_files(args.files)}")

    if args.context:
        parts.append(f"\n## Context\n{args.context}")

    if args.stdin and not sys.stdin.isatty():
        stdin_data = sys.stdin.read()
        if stdin_data.strip():
            parts.append(f"\n## Input Data\n```\n{stdin_data}\n```")

    user_prompt = "\n".join(parts)
    system = args.system if args.system else SYSTEM_PROMPT

    response = call_llm(system, user_prompt, max_tokens=args.max_tokens, temperature=args.temperature)
    # Handle Windows encoding by replacing non-ASCII chars
    sys.stdout.buffer.write(response.encode('utf-8', errors='replace'))
    sys.stdout.buffer.write(b'\n')


if __name__ == '__main__':
    main()

"""Compare the same queries against both providers without calling an LLM."""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from web_search import PROVIDERS, search

CASES = (
    ("recent", "latest Apple security releases macOS", "Apple's current security release list or bulletin"),
    ("recent", "latest Federal Reserve FOMC statement", "most recent official FOMC statement"),
    ("recent", "latest NASA Artemis mission update", "a recent dated NASA mission update"),
    ("recent", "latest Python 3.14 release notes", "official notes for a current 3.14 release"),
    ("recent", "USGS latest earthquakes California", "live or recently updated USGS earthquake data"),
    ("local", "Whistler Blackcomb current lift status", "current resort lift or mountain status"),
    ("local", "Vancouver BC Stanley Park parking rates", "applicable official parking rates for Stanley Park"),
    ("local", "Chicago O'Hare Blue Line overnight schedule", "CTA timetable or overnight service details"),
    ("local", "Calgary Central Library Sunday hours", "hours for the correct library branch"),
    ("local", "Tofino Long Beach campground reservations Green Point", "official Green Point reservation information"),
    ("niche", "USB C e-marker cable 5A USB-IF", "a credible explanation of the 5A cable marking requirement"),
    ("niche", "BC avalanche terrain exposure scale ATES mapping", "ATES maps or guidance for British Columbia"),
    ("niche", "CIEDE2000 Sharma supplementary test data", "the authors' reference dataset or paper"),
    ("niche", "NIST WWVB phase modulation frame format", "NIST documentation of WWVB phase modulation"),
    ("niche", "Vancouver Island Garry oak restoration guidelines", "regional Garry oak restoration guidance"),
    ("docs", "Python 3.13 asyncio Queue shutdown documentation", "Python 3.13 documentation for Queue.shutdown"),
    ("docs", "Django 5.2 composite primary key limitations", "Django 5.2 documentation covering limitations"),
    ("docs", "PostgreSQL 17 pg_stat_io columns documentation", "PostgreSQL 17 documentation for pg_stat_io"),
    ("docs", "SQLite WAL checkpoint truncate pragma documentation", "official checkpoint pragma documentation"),
    ("docs", "Apple Swift 6 strict concurrency migration guide", "official Swift 6 migration guidance"),
)


def collect(providers):
    runs = []
    last_start = {}
    started = datetime.now(timezone.utc).isoformat()
    for category, query, target in CASES:
        for provider in providers:
            if provider in last_start:
                pause = 1.1 - (time.monotonic() - last_start[provider])
                if pause > 0:
                    time.sleep(pause)
            start = last_start[provider] = time.monotonic()
            observation = search(query, provider=provider)
            if observation.get("error_code") in {"missing_credentials", "auth", "missing_dependency"}:
                raise RuntimeError(f"{provider}: {observation['error']}")
            runs.append({"category": category, "query": query, "target": target, "provider": provider,
                         "elapsed_ms": round((time.monotonic() - start) * 1000), "useful_top_5": None,
                         "observation": observation})
    return {"started_utc": started, "runs": runs}


def summarize(report):
    summary = {}
    for provider in dict.fromkeys(row['provider'] for row in report['runs']):
        runs = [row for row in report['runs'] if row['provider'] == provider]
        judged = [row['useful_top_5'] for row in runs if type(row['useful_top_5']) is bool]
        summary[provider] = {
            "queries": len(runs),
            "queries_without_links": sum(not row['observation'].get('results') and
                                         row['observation']['status'] != 'error' for row in runs),
            "errors": sum(row['observation']['status'] == 'error' for row in runs),
            "median_elapsed_ms": statistics.median(row['elapsed_ms'] for row in runs),
            "judged": len(judged),
            "useful_top_5": sum(judged),
            "useful_top_5_rate": sum(judged) / len(runs) if len(judged) == len(runs) else None,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--provider', choices=PROVIDERS, help='Run just one provider (default: both)')
    mode.add_argument('--score', type=Path, help='Summarize a previously collected and judged JSON report')
    args = parser.parse_args()
    try:
        if args.score:
            output = summarize(json.loads(args.score.read_text(encoding='utf-8')))
        else:
            output = collect((args.provider,) if args.provider else PROVIDERS)
            output['summary'] = summarize(output)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f'Benchmark failed: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

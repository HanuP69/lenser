"""The table of what the model calls cost.

The account of the calls lives in the logs folder of every session, one
line of JSON per try of one provider. This report reads the lines back
and prints the numbers the improvement plan is judged against:

  - how many tries each provider saw, and what came of them,
  - how many tokens the calls read and wrote,
  - which piece of the structuring work spent the tokens,
  - how many requests one batch of notes takes.

    python eval/usage_report.py output/notes_check1 output/notes_check2
    python eval/usage_report.py output/--logs --batch-gap 120
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

BATCH_GAP_SECONDS = 90.0


def collect_files(paths: list[str]) -> list[Path]:
    """Every usage file the arguments point at.

    A session directory or a logs directory counts, and so does a file
    given directly.
    """
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            print(f"not found: {path}", file=sys.stderr)
            continue
        if path.is_file() and path.suffix == ".jsonl":
            files.append(path)
            continue
        for candidate in sorted(path.glob("**/*llm_usage.jsonl")):
            files.append(candidate)
    return files


def read_rows(files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in files:
        # A file written by a Windows tool can carry a BOM in front of
        # its first line. The reading ignores it, so the first call of
        # the session is not lost.
        with open(path, "r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row["session"] = path.parts[-3] if len(path.parts) >= 3 else path.name
                rows.append(row)
    rows.sort(key=lambda row: float(row.get("ts", 0.0)))
    return rows


def known_total(rows: list[dict[str, Any]], field: str) -> tuple[int, int]:
    """The sum of one token field and how many rows knew the number.

    A provider that does not put its usage in the answer leaves the
    count unknown. The share of the known rows says how much of each
    total can be trusted.
    """
    total = 0
    counted = 0
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        total += int(value)
        counted += 1
    return total, counted


def average(rows: list[dict[str, Any]], field: str) -> Optional[float]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def batches_of(rows: list[dict[str, Any]], gap: float) -> list[list[dict[str, Any]]]:
    """The rows of one task, cut into batches by a long silence.

    A batch of notes is finished when the structuring agent goes quiet
    for longer than the gap. Everything inside one quiet stretch was
    one pass of the graph, even when a provider was skipped and the
    next one was asked.
    """
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    last_time: Optional[float] = None
    for row in rows:
        moment = float(row.get("ts", 0.0))
        if current and moment - float(last_time) > gap:
            batches.append(current)
            current = []
        current.append(row)
        last_time = moment
    if current:
        batches.append(current)
    return batches


def print_task_table(rows: list[dict[str, Any]]) -> None:
    tasks = sorted({row.get("task", "") for row in rows})
    columns = ("task", "tries", "ok", "limited", "open", "failed", "in tokens", "out tokens", "s/ok")
    print("  ".join(f"{name:>9}" for name in columns))
    for task in tasks:
        own = [row for row in rows if row.get("task") == task]
        inbound, inbound_known = known_total(own, "input_tokens")
        outbound, _ = known_total(own, "output_tokens")
        latency = average([row for row in own if row.get("outcome") == "ok"], "request_seconds")
        counts = {
            outcome: sum(1 for row in own if row.get("outcome") == outcome)
            for outcome in ("ok", "limited", "breaker_open", "failed")
        }
        mark = "" if inbound_known == counts["ok"] else f" ({inbound_known}/{counts['ok']} known)"
        cells = (
            task, str(len(own)), str(counts["ok"]), str(counts["limited"]),
            str(counts["breaker_open"]), str(counts["failed"]),
            f"{inbound}{mark}", f"{outbound}",
            f"{latency:.1f}" if latency is not None else "-"
        )
        print("  ".join(f"{name:>9}" for name in cells))
    providers = sorted({row.get("provider", "") for row in rows})
    for provider in providers:
        own = [row for row in rows if row.get("provider") == provider]
        inbound, _ = known_total(own, "input_tokens")
        outbound, _ = known_total(own, "output_tokens")
        counts = {
            outcome: sum(1 for row in own if row.get("outcome") == outcome)
            for outcome in ("ok", "limited", "breaker_open", "failed")
        }
        cells = (
            f" {provider}", str(len(own)), str(counts["ok"]), str(counts["limited"]),
            str(counts["breaker_open"]), str(counts["failed"]),
            str(inbound), str(outbound), "-"
        )
        print("  ".join(f"{name:>9}" for name in cells))


def print_stage_table(rows: list[dict[str, Any]]) -> None:
    stages: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        stage_name = row.get("stage", "")
        if stage_name:
            stages[(row.get("task", ""), stage_name)].append(row)
    if not stages:
        return
    columns = ("task/stage", "tries", "ok", "avg in", "avg out", "avg s")
    print()
    print("  ".join(f"{name:>14}" for name in columns))
    for (task, stage_name), own in sorted(stages.items()):
        likes = [row for row in own if row.get("outcome") == "ok"]
        inbound, _ = known_total(likes, "input_tokens")
        outbound, _ = known_total(likes, "output_tokens")
        avg_in = inbound / len(likes) if likes else 0.0
        avg_out = outbound / len(likes) if likes else 0.0
        latency = average(likes, "request_seconds")
        cells = (
            f"{task}/{stage_name}", str(len(own)), str(len(likes)),
            f"{avg_in:.0f}", f"{avg_out:.0f}",
            f"{latency:.1f}" if latency is not None else "-"
        )
        print("  ".join(f"{name:>14}" for name in cells))


def print_batch_table(rows: list[dict[str, Any]], gap: float) -> None:
    for task in sorted({row.get("task", "") for row in rows}):
        own = [row for row in rows if row.get("task", "") == task]
        batches = batches_of(own, gap)
        if len(batches) <= 1:
            continue
        sizes = [len(batch) for batch in batches]
        inbound, _ = known_total(own, "input_tokens")
        batch_in = inbound / len(batches)
        batch_out, _ = known_total(own, "output_tokens")
        cells = (
            f"{task}", str(len(batches)), f"{sum(sizes) / len(sizes):.1f}",
            f"{batch_in:.0f}", f"{batch_out / len(batches):.0f}"
        )
        print()
        print(f"  batch estimate for {task} (a silence longer than {gap:.0f}s ends a batch)")
        columns = ("task", "batches", "tries/batch", "in/batch", "out/batch")
        print("  " + "  ".join(f"{name:>11}" for name in columns))
        print("  " + "  ".join(f"{name:>11}" for name in cells))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="session folders, logs folders or usage jsonl files")
    parser.add_argument("--batch-gap", type=float, default=BATCH_GAP_SECONDS)
    args = parser.parse_args()

    files = collect_files(args.paths)
    if not files:
        print("no usage files found", file=sys.stderr)
        return 1
    rows = read_rows(files)
    if not rows:
        print("the usage files hold no rows yet", file=sys.stderr)
        return 1

    sources = ", ".join(sorted({row["session"] for row in rows}))
    print(f"sessions: {sources}   rows: {len(rows)}")
    print()
    print_task_table(rows)
    print_stage_table(rows)
    print_batch_table(rows, args.batch_gap)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Usage: avg_iter_time_from_csv.py [--dir DIR | CSV ...] [--glob PATTERN] [--per-file] [--skip-first-per-file] [--no-skip-epoch1-itr0] [--no-save]

Examples
--------
Directory containing per-rank CSVs (auto-saves avg_iter_time_stats.txt in that directory):
  python3 dev/avg_iter_time_from_csv.py /path/to/run_dir

If your files don't match the default *_r*.csv pattern:
  python3 dev/avg_iter_time_from_csv.py /path/to/run_dir --glob "*.csv"
"""
import argparse
import csv
import re
import sys
from pathlib import Path

TIME_COL = "time (ms)"
STATS_FILENAME = "avg_iter_time_stats.txt"


def rank_sort_key(path: Path):
    m = re.search(r"_r(\d+)\.csv$", path.name, flags=re.IGNORECASE)
    if m:
        return (0, int(m.group(1)))
    return (1, path.name.lower())


def parse_rows_from_path(path: Path):
    """Return (rows, n_skipped) with rows as (time_ms, epoch, itr); skip repeated headers."""
    rows: list[tuple[float, int, int]] = []
    skipped = 0
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and TIME_COL not in reader.fieldnames:
            raise SystemExit(
                f"{path}: missing column {TIME_COL!r}; found {reader.fieldnames!r}"
            )
        if reader.fieldnames and "itr" not in reader.fieldnames:
            raise SystemExit(
                f"{path}: missing column 'itr'; found {reader.fieldnames!r}"
            )
        for row in reader:
            epoch_raw = (row.get("epoch") or "").strip()
            itr_raw = (row.get("itr") or "").strip()
            time_raw = (row.get(TIME_COL) or "").strip()
            try:
                epoch = int(epoch_raw)
            except ValueError:
                skipped += 1
                continue
            try:
                itr = int(itr_raw)
            except ValueError:
                skipped += 1
                continue
            try:
                t = float(time_raw)
            except ValueError:
                skipped += 1
                continue
            rows.append((t, epoch, itr))
    return rows, skipped


def resolve_csv_files_and_out_dir(args) -> tuple[list[Path], Path]:
    """Return (csv_paths, directory_where_stats_should_be_written)."""
    if args.dir is not None:
        base = args.dir.resolve()
        if not base.is_dir():
            raise SystemExit(f"Not a directory: {base}")
        files = sorted(base.glob(args.glob), key=rank_sort_key)
        return files, base

    if not args.paths:
        raise SystemExit(
            "Provide a run directory (positional) or --dir, or one or more CSV file paths."
        )

    if len(args.paths) == 1 and args.paths[0].is_dir():
        base = args.paths[0].resolve()
        files = sorted(base.glob(args.glob), key=rank_sort_key)
        return files, base

    files = sorted(args.paths, key=rank_sort_key)
    for p in files:
        if not p.is_file():
            raise SystemExit(f"Not a file: {p}")
    parents = {p.resolve().parent for p in files}
    out_dir = files[0].resolve().parent
    if len(parents) > 1:
        print(
            "Warning: CSV paths span multiple directories; "
            f"writing stats to {out_dir} (directory of first file).",
            file=sys.stderr,
        )
    return files, out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Average pretraining step time from multi-rank CSV logs (time (ms) column)."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        help="Directory to search with --glob (default *_r*.csv).",
    )
    parser.add_argument(
        "--glob",
        default="*_r*.csv",
        help="Pattern under --dir or single-directory positional (default: *_r*.csv).",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="CSV files, or one directory containing rank CSVs.",
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="Print mean and count per CSV in addition to pooled stats.",
    )
    parser.add_argument(
        "--skip-first-per-file",
        action="store_true",
        help="Drop the first valid timing row in each file (warmup / first-iter skew).",
    )
    parser.add_argument(
        "--skip-epoch1-itr0",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude rows with epoch==1 and itr==0 (default: on). Use --no-skip-epoch1-itr0 to keep them.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help=f"Do not write {STATS_FILENAME} under the run directory.",
    )
    args = parser.parse_args()

    if args.dir is not None and args.paths:
        raise SystemExit("Use either --dir or positional paths, not both.")

    files, out_dir = resolve_csv_files_and_out_dir(args)
    if not files:
        raise SystemExit("No CSV files matched.")

    for path in files:
        if not re.search(r"_r(\d+)\.csv$", path.name, flags=re.IGNORECASE):
            print(
                f"Warning: {path.name} does not match *_r<N>.csv rank suffix.",
                file=sys.stderr,
            )

    all_times = []
    per_file_stats = []
    total_skipped = 0
    excluded_epoch1_itr0 = 0

    for path in files:
        rows, skipped = parse_rows_from_path(path)
        total_skipped += skipped
        if args.skip_first_per_file and rows:
            rows = rows[1:]
        if args.skip_epoch1_itr0:
            before = len(rows)
            rows = [(t, e, i) for t, e, i in rows if not (e == 1 and i == 0)]
            excluded_epoch1_itr0 += before - len(rows)
        times = [t for t, _, _ in rows]
        per_file_stats.append((path, times))
        all_times.extend(times)

    if not all_times:
        raise SystemExit("No valid timing rows after filtering.")

    pooled_mean = sum(all_times) / len(all_times)
    lines = [
        f"files: {len(files)}",
        f"rows:  {len(all_times)}",
    ]
    if total_skipped:
        lines.append(f"skipped_rows: {total_skipped}")
    if excluded_epoch1_itr0:
        lines.append(f"excluded_epoch1_itr0: {excluded_epoch1_itr0}")
    lines.append(f"mean_time_ms: {pooled_mean:.2f}")

    if args.per_file:
        lines.append("per_file:")
        for path, times in per_file_stats:
            if not times:
                lines.append(f"  {path.name}: n=0 (no data)")
            else:
                m = sum(times) / len(times)
                lines.append(f"  {path.name}: n={len(times)} mean_ms={m:.2f}")

    text = "\n".join(lines) + "\n"
    print(text, end="")

    if not args.no_save:
        stats_path = out_dir / STATS_FILENAME
        stats_path.write_text(text, encoding="utf-8")
        print(f"saved: {stats_path}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.stderr.close()
        sys.exit(0)

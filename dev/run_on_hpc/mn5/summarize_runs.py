#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def manifest_sort_key(manifest):
    return (
        manifest.get("completed_at")
        or manifest.get("updated_at")
        or manifest.get("started_at")
        or ""
    )


def latest_manifest_from_runs(runs_root):
    manifests = []
    if not runs_root.exists():
        return None

    for manifest_path in runs_root.glob("*/run_manifest.json"):
        manifest = read_json(manifest_path)
        if manifest:
            manifests.append(manifest)

    if not manifests:
        return None
    return max(manifests, key=manifest_sort_key)


def collect_pretraining(pretraining_root):
    summaries = []
    if not pretraining_root.exists():
        return summaries

    for experiment_dir in sorted(path for path in pretraining_root.iterdir() if path.is_dir()):
        latest_run_info = read_json(experiment_dir / "latest_run.json")
        latest_manifest = None
        if latest_run_info and latest_run_info.get("run_manifest_path"):
            latest_manifest = read_json(Path(latest_run_info["run_manifest_path"]))
        if latest_manifest is None:
            latest_manifest = latest_manifest_from_runs(experiment_dir / "runs")
        if latest_manifest:
            summaries.append(latest_manifest)
    return sorted(summaries, key=lambda item: item.get("experiment_tag", ""))


def collect_latest_probes(linprobe_root):
    by_source_tag = {}
    if not linprobe_root.exists():
        return by_source_tag

    for probe_dir in sorted(path for path in linprobe_root.iterdir() if path.is_dir()):
        latest_manifest = latest_manifest_from_runs(probe_dir / "runs")
        if latest_manifest is None:
            continue

        source_tag = latest_manifest.get("source_checkpoint_tag") or probe_dir.name
        current = by_source_tag.get(source_tag)
        if current is None or manifest_sort_key(latest_manifest) > manifest_sort_key(current):
            by_source_tag[source_tag] = latest_manifest

    return by_source_tag


def format_probe_summary(probe_manifest):
    if probe_manifest is None:
        return "no probe runs"

    status = probe_manifest.get("status", "unknown")
    run_id = probe_manifest.get("run_id", "unknown")
    completed_at = probe_manifest.get("completed_at") or probe_manifest.get("started_at", "n/a")
    best_val_acc = probe_manifest.get("best_val_acc")
    if best_val_acc is None:
        score = "best_val_acc=n/a"
    else:
        score = f"best_val_acc={best_val_acc:.2f}"
    return f"{status} | {run_id} | {score} | {completed_at}"


def print_summary(pretraining_runs, latest_probes):
    if not pretraining_runs:
        print("No pretraining runs found.")
        return

    print("Pretraining Experiments")
    print("=======================")
    for manifest in pretraining_runs:
        experiment_tag = manifest.get("experiment_tag", "unknown")
        run_id = manifest.get("run_id", "unknown")
        status = manifest.get("status", "unknown")
        started_at = manifest.get("started_at", "n/a")
        latest_ckpt = manifest.get("latest_checkpoint_path", "n/a")
        probe_summary = format_probe_summary(latest_probes.get(experiment_tag))

        print(f"{experiment_tag}")
        print(f"  latest_run: {run_id}")
        print(f"  status: {status}")
        print(f"  started_at: {started_at}")
        print(f"  latest_checkpoint: {latest_ckpt}")
        print(f"  latest_probe: {probe_summary}")
        print()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Summarize latest I-JEPA pretraining and probe runs from manifest files."
    )
    parser.add_argument(
        "--log-root",
        default=None,
        help="Path to the ijepa log root containing pretraining/ and linprobe/ directories.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the summary as JSON instead of plain text.",
    )
    return parser


def resolve_log_root(log_root_arg):
    if log_root_arg:
        return Path(log_root_arg).expanduser().resolve()

    env_root = Path(os.environ.get("IJEPA_LOG_ROOT", "~/logs/ijepa")).expanduser()
    return env_root.resolve()


def main():
    parser = build_parser()
    args = parser.parse_args()

    log_root = resolve_log_root(args.log_root)
    pretraining_runs = collect_pretraining(log_root / "pretraining")
    latest_probes = collect_latest_probes(log_root / "linprobe")

    if args.json:
        payload = {
            "log_root": str(log_root),
            "pretraining": pretraining_runs,
            "latest_probes_by_source_tag": latest_probes,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"log_root: {log_root}")
    print()
    print_summary(pretraining_runs, latest_probes)


if __name__ == "__main__":
    main()

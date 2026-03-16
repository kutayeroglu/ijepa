import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from functools import lru_cache


def timestamp_for_run_id():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def timestamp_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sanitize_run_component(value):
    sanitized = []
    for char in str(value):
        if char.isalnum() or char in ("-", "_", "."):
            sanitized.append(char)
        else:
            sanitized.append("-")
    cleaned = "".join(sanitized).strip("-_.")
    return cleaned or "run"


def checkpoint_stem(model_path):
    stem = os.path.basename(model_path)
    for suffix in ("-latest.pth.tar", ".pth.tar", ".pth"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return sanitize_run_component(stem)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def build_run_id(prefix, slurm_job_id=None):
    prefix = sanitize_run_component(prefix)
    slurm_job_id = slurm_job_id or os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        return f"{slurm_job_id}_{prefix}"
    return f"{timestamp_for_run_id()}_{prefix}"


@lru_cache(maxsize=None)
def get_git_commit(cwd=None):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def get_runtime_context():
    context = {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_submit_dir": os.environ.get("SLURM_SUBMIT_DIR"),
        "launcher_script": os.environ.get("IJEPA_LAUNCHER_SCRIPT"),
        "git_commit": get_git_commit(),
    }
    return {key: value for key, value in context.items() if value not in (None, "")}


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

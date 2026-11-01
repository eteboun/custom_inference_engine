import json
import subprocess
from datetime import datetime
from pathlib import Path

import torch


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def git_state():
    def run(*cmd):
        return subprocess.run(
            ["git", *cmd], cwd=RESULTS_DIR.parent, capture_output=True, text=True
        ).stdout.strip()

    return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def environment(impl):
    env = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    if impl == "vllm":
        import vllm
        env["vllm"] = vllm.__version__
    return env


def save_results(title, impl, args, table, runs):
    RESULTS_DIR.mkdir(exist_ok=True)
    now = datetime.now()
    path = RESULTS_DIR / f"{now:%Y%m%d-%H%M%S}_{impl}.json"
    with open(path, "w") as f:
        json.dump({
            "title": title,
            "impl": impl,
            "timestamp": now.isoformat(timespec="seconds"),
            "git": git_state(),
            "env": environment(impl),
            "args": args,
            "table": table,
            "runs": runs,
        }, f)
    return path

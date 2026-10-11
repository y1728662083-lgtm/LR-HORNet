"""Dispatch all audited-grid FAST folds over one or more selected GPUs."""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import yaml


ADDITION_ROOT = Path(__file__).resolve().parent
PAPER4_ROOT = ADDITION_ROOT.parent
CONFIG_ROOT = ADDITION_ROOT / "configs"
LOG_ROOT = PAPER4_ROOT / "logs" / "fast_native_dim32_harmonized_v1_20261010"
CONFIGS = (
    ("ds003626", "fast_native_ds003626.yaml"),
    ("russian", "fast_native_russian.yaml"),
    ("spanish", "fast_native_spanish.yaml"),
    ("bimodal", "fast_native_bimodal.yaml"),
)


@dataclass(frozen=True)
class Job:
    dataset: str
    fold_index: int
    config_path: Path

    @property
    def name(self) -> str:
        return f"fast_{self.dataset}_fold{self.fold_index + 1:02d}"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def prepare_jobs() -> list[Job]:
    jobs: list[Job] = []
    for dataset, filename in CONFIGS:
        path = CONFIG_ROOT / filename
        config = load_yaml(path)
        jobs.extend(
            Job(dataset, int(fold), path)
            for fold in config["split"]["fold_indices"]
        )
    return jobs


def command_for(job: Job) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(ADDITION_ROOT / "run_fast_native_all_datasets.py"),
        "--config",
        str(job.config_path),
        "--models",
        "fast",
        "--folds",
        str(job.fold_index + 1),
    ]


def run_worker(
    gpu: int,
    pending: queue.Queue[Job],
    failures: list[str],
    stop: threading.Event,
    lock: threading.Lock,
) -> None:
    while not stop.is_set():
        try:
            job = pending.get_nowait()
        except queue.Empty:
            return
        try:
            log_path = LOG_ROOT / f"{job.name}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            prefix = os.pathsep.join((str(PAPER4_ROOT / "src"), str(PAPER4_ROOT)))
            env["PYTHONPATH"] = prefix + (
                os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
            )
            command = command_for(job)
            with lock:
                print(f"START gpu={gpu} job={job.name} log={log_path}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                log.write("COMMAND " + " ".join(command) + "\n")
                log.flush()
                result = subprocess.run(
                    command,
                    cwd=PAPER4_ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode:
                failures.append(job.name)
                stop.set()
                with lock:
                    print(f"FAIL gpu={gpu} job={job.name} rc={result.returncode}", flush=True)
            else:
                with lock:
                    print(f"DONE gpu={gpu} job={job.name}", flush=True)
        finally:
            pending.task_done()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--plan", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        raise ValueError(f"GPU ids must be unique non-negative integers: {args.gpus}")
    jobs = prepare_jobs()
    print(
        f"PLAN suite=fast gpus={args.gpus} total={len(jobs)} "
        + " ".join(
            f"{name}={sum(job.dataset == name for job in jobs)}"
            for name, _ in CONFIGS
        ),
        flush=True,
    )
    if args.plan:
        for sequence, job in enumerate(jobs, 1):
            print(f"{sequence:02d}/{len(jobs):02d} {job.name}")
        return

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    pending: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        pending.put(job)
    failures: list[str] = []
    stop = threading.Event()
    lock = threading.Lock()
    workers = [
        threading.Thread(
            target=run_worker,
            args=(gpu, pending, failures, stop, lock),
            daemon=False,
        )
        for gpu in args.gpus
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if failures:
        raise SystemExit(
            f"Stopped after failed job(s): {', '.join(failures)}. "
            f"Inspect {LOG_ROOT}."
        )
    print(f"ALL DONE jobs={len(jobs)}", flush=True)


if __name__ == "__main__":
    main()

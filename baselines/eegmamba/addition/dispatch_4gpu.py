"""Distribute all EEGMamba LOSO folds across four GPUs."""

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


ROOT = Path(__file__).resolve().parents[1]
ADDITION = ROOT / "addition"
LOG_ROOT = ROOT / "logs"
MODEL = "eegmamba"
CONFIGS = (
    ADDITION / "configs" / "ds003626_official_baselines.yaml",
    ADDITION / "configs" / "directional_words_russian_official_baselines.yaml",
    ADDITION / "configs" / "directional_words_spanish_official_baselines.yaml",
    ADDITION / "configs" / "bimodal_inner_speech_official_baselines.yaml",
)


@dataclass(frozen=True)
class Job:
    dataset: str
    config: Path
    fold: int


def jobs() -> list[Job]:
    result: list[Job] = []
    for config_path in CONFIGS:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if set(config["models"]) != {MODEL}:
            raise ValueError(f"{config_path} must contain only {MODEL}")
        dataset = str(config["data"]["dataset"])
        result.extend(
            Job(dataset, config_path, int(index) + 1)
            for index in config["split"]["fold_indices"]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if len(args.gpus) != 4 or len(set(args.gpus)) != 4:
        raise ValueError("Exactly four unique GPU indices are required")
    planned = jobs()
    print(f"PLAN model={MODEL} fits={len(planned)} gpus={args.gpus}", flush=True)
    if args.plan:
        for number, job in enumerate(planned, 1):
            print(f"{number:03d}/{len(planned):03d} {job.dataset} fold={job.fold:02d}")
        return

    work: queue.Queue[Job | None] = queue.Queue()
    for job in planned:
        work.put(job)
    for _ in args.gpus:
        work.put(None)
    failures: list[tuple[Job, int, Path]] = []
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            job = work.get()
            if job is None:
                return
            runner = (
                "run_official_baselines.py"
                if job.dataset == "ds003626"
                else "run_multidataset_official_baselines.py"
            )
            log = LOG_ROOT / job.dataset / f"fold_{job.fold:02d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, "-u", str(ADDITION / runner),
                "--config", str(job.config), "--models", MODEL,
                "--folds", str(job.fold),
            ]
            environment = os.environ.copy()
            environment.update({
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTHONPATH": os.pathsep.join(
                    [str(ROOT / "src"), str(ROOT), str(ADDITION),
                     environment.get("PYTHONPATH", "")]
                ),
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
            })
            with lock:
                print(f"START gpu={gpu} dataset={job.dataset} fold={job.fold:02d} log={log}", flush=True)
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(
                    command, cwd=ROOT, env=environment,
                    stdout=stream, stderr=subprocess.STDOUT, check=False,
                )
            with lock:
                if result.returncode:
                    failures.append((job, result.returncode, log))
                    print(f"FAILED gpu={gpu} dataset={job.dataset} fold={job.fold:02d}", flush=True)
                else:
                    print(f"DONE gpu={gpu} dataset={job.dataset} fold={job.fold:02d}", flush=True)

    workers = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    if failures:
        for job, code, log in failures:
            print(f"FAILED_JOB dataset={job.dataset} fold={job.fold} rc={code} log={log}")
        raise SystemExit(1)
    print("ALL JOBS COMPLETE", flush=True)


if __name__ == "__main__":
    main()

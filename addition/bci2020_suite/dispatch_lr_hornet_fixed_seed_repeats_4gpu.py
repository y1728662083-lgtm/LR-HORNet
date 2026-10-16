"""Run LR-HORNet on every BCI2020 fold ten times with one fixed seed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import yaml


PAPER4_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = PAPER4_ROOT / "configs/main/bci2020.yaml"
RUNTIME_ROOT = Path(__file__).resolve().parent / "runtime_configs/lr_hornet_repeat10"
LOG_ROOT = PAPER4_ROOT / "logs/bci2020_lr_hornet_fixed_seed_repeat10"
DONE_ROOT = LOG_ROOT / ".done"
DEFAULT_OUTPUT_ROOT = (
    PAPER4_ROOT / "outputs/bci2020_lr_hornet_fixed_seed_repeat10"
)
N_FOLDS = 15
DEFAULT_REPEATS = 10
PARAMETER_PATTERN = re.compile(
    r"parameters:\s*total=([\d,]+),\s*trainable=([\d,]+)"
)


@dataclass(frozen=True)
class Job:
    repeat: int
    fold: int


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_base_config(config: dict) -> int:
    if config["seed"] != 42:
        raise ValueError(f"Expected the archived fixed seed 42, got {config['seed']}")
    if config["data"].get("dataset") != "bci2020":
        raise ValueError("The repeat runner is restricted to BCI2020")
    if config["data"]["subjects"] != list(range(1, N_FOLDS + 1)):
        raise ValueError("BCI2020 repeat experiment requires subjects 1..15")
    if config["split"]["fold_indices"] != list(range(N_FOLDS)):
        raise ValueError("The base config must contain all 15 LOSO folds")
    if config["split"].get("adapt_held_out_subjects") is not True:
        raise ValueError("Calibrated LOSO must remain enabled")
    if config["model"]["name"] != "group4_patchLstm_deformableLstm":
        raise ValueError("The repeat runner is restricted to LR-HORNet")
    kwargs = config["model"]["kwargs"]
    if (
        kwargs.get("patch_size") != 128
        or kwargs.get("patch_stride") != 64
        or kwargs.get("pad_incomplete_patch") is not True
    ):
        raise ValueError("LR-HORNet must retain the audited 0.5 s/0.25 s patch contract")
    return int(config["seed"])


def _runtime_config(
    job: Job,
    base: dict,
    output_root: Path,
    dataset_root: Path,
) -> Path:
    payload = json.loads(json.dumps(base))
    payload["data"]["root"] = str(dataset_root)
    payload["split"]["fold_indices"] = [job.fold - 1]
    payload["training"]["output_root"] = str(
        output_root / f"repeat_{job.repeat:02d}"
    )
    payload["loader"]["cache_preprocessed_trials"] = True
    payload["loader"]["num_workers"] = 0
    payload["loader"]["persistent_workers"] = False
    payload["loader"]["prefetch_factor"] = 2
    path = (
        RUNTIME_ROOT
        / f"repeat_{job.repeat:02d}_fold_{job.fold:02d}.yaml"
    )
    _write_yaml(path, payload)
    return path


def _result_dir(job: Job, output_root: Path) -> Path:
    return (
        output_root
        / f"repeat_{job.repeat:02d}"
        / "bci2020"
        / "imagined_speech"
        / "group4_patchLstm_deformableLstm"
        / f"fold_{job.fold:02d}"
    )


def _done_marker(job: Job) -> Path:
    return DONE_ROOT / f"repeat_{job.repeat:02d}_fold_{job.fold:02d}.json"


def _is_resumable(
    job: Job,
    *,
    marker: Path,
    result_file: Path,
    seed: int,
    config_sha256: str,
) -> bool:
    if not marker.is_file() or not result_file.is_file():
        return False
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    result_payload = json.loads(result_file.read_text(encoding="utf-8"))
    required = {
        "status": "complete",
        "repeat": job.repeat,
        "fold": job.fold,
        "seed": seed,
        "base_config_sha256": config_sha256,
    }
    for name, expected in required.items():
        if marker_payload.get(name) != expected or result_payload.get(name) != expected:
            raise RuntimeError(
                f"Existing repeat result has a different contract for {job}: "
                f"field={name!r}. Move that exact repeat/fold output before rerunning."
            )
    return True


def _job_env(gpu: str, cpu_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = str(cpu_threads)
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    env["PYTHONPATH"] = os.pathsep.join(
        (str(PAPER4_ROOT / "src"), str(PAPER4_ROOT))
    ) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _best_validation_accuracy(training_log: Path) -> tuple[int, float]:
    with training_log.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty training log: {training_log}")
    selected = [row for row in rows if row.get("is_best", "").casefold() in {"1", "true"}]
    best = selected[-1] if selected else max(
        rows,
        key=lambda row: float(row["val_accuracy"]),
    )
    return int(best["epoch"]), float(best["val_accuracy"])


def _record_result(
    job: Job,
    *,
    output_root: Path,
    log_path: Path,
    seed: int,
    gpu: str,
    elapsed: float,
    config_sha256: str,
) -> dict:
    result_dir = _result_dir(job, output_root)
    metrics_path = result_dir / "test_metrics.json"
    training_log = result_dir / "training_log.csv"
    checkpoint = result_dir / "best_model.pt"
    for path in (metrics_path, training_log, checkpoint):
        if not path.is_file():
            raise RuntimeError(f"Completed process is missing output: {path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    best_epoch, validation_accuracy = _best_validation_accuracy(training_log)
    match = PARAMETER_PATTERN.search(log_path.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError(f"Parameter count not found in {log_path}")
    subjects = tuple(range(1, N_FOLDS + 1))
    record = {
        "status": "complete",
        "model": "lr_hornet",
        "model_name": "group4_patchLstm_deformableLstm",
        "dataset": "bci2020",
        "protocol": "calibrated_loso",
        "seed": seed,
        "repeat": job.repeat,
        "fold": job.fold,
        "validation_subject": subjects[job.fold % N_FOLDS],
        "test_subject": subjects[job.fold - 1],
        "best_epoch": best_epoch,
        "validation_accuracy": validation_accuracy,
        "test_loss": float(metrics["loss"]),
        "test_accuracy": float(metrics["accuracy"]),
        "test_balanced_accuracy": float(metrics["balanced_accuracy"]),
        "test_macro_f1": float(metrics["macro_f1"]),
        "parameters_total": int(match.group(1).replace(",", "")),
        "parameters_trainable": int(match.group(2).replace(",", "")),
        "gpu": gpu,
        "elapsed_seconds": elapsed,
        "base_config_sha256": config_sha256,
        "result_directory": str(result_dir),
        "training_log": str(training_log),
        "test_metrics": str(metrics_path),
        "launcher_log": str(log_path),
    }
    (result_dir / "repeat_result.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return record


def _worker(
    gpu: str,
    cpu_ids: tuple[int, ...],
    jobs: queue.Queue[Job],
    base_config: dict,
    output_root: Path,
    dataset_root: Path,
    seed: int,
    config_sha256: str,
    cpu_threads: int,
    rerun: bool,
    failures: list[dict],
    lock: threading.Lock,
    stop_event: threading.Event,
    max_failures: int,
) -> None:
    while not stop_event.is_set():
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        marker = _done_marker(job)
        result_file = _result_dir(job, output_root) / "repeat_result.json"
        try:
            resumable = not rerun and _is_resumable(
                job,
                marker=marker,
                result_file=result_file,
                seed=seed,
                config_sha256=config_sha256,
            )
        except Exception as error:
            failure = {
                "repeat": job.repeat,
                "fold": job.fold,
                "gpu": gpu,
                "resume_error": f"{type(error).__name__}: {error}",
            }
            with lock:
                failures.append(failure)
                stop_event.set()
            print(
                f"FAILED resume-contract repeat={job.repeat:02d} "
                f"fold={job.fold:02d}: {error}",
                flush=True,
            )
            jobs.task_done()
            return
        if resumable:
            print(
                f"SKIP gpu={gpu} repeat={job.repeat:02d} fold={job.fold:02d}",
                flush=True,
            )
            jobs.task_done()
            continue

        config_path = _runtime_config(
            job,
            base_config,
            output_root,
            dataset_root,
        )
        log_path = LOG_ROOT / f"repeat_{job.repeat:02d}_fold_{job.fold:02d}.log"
        command = [sys.executable, "-u", "main.py", "--config", str(config_path)]
        cpu_spec = ",".join(map(str, cpu_ids))
        run_command = command
        if os.name == "posix" and cpu_ids and shutil.which("taskset"):
            run_command = ["taskset", "-c", cpu_spec, *command]
        print(
            f"START gpu={gpu} repeat={job.repeat:02d} fold={job.fold:02d} "
            f"seed={seed} cpus={cpu_spec} log={log_path}",
            flush=True,
        )
        started = time.time()
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            log.write("COMMAND " + subprocess.list2cmdline(run_command) + "\n")
            log.write(f"CUDA_VISIBLE_DEVICES={gpu}\nCPU_AFFINITY={cpu_spec}\n")
            completed = subprocess.run(
                run_command,
                cwd=PAPER4_ROOT,
                env=_job_env(gpu, cpu_threads),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        elapsed = time.time() - started
        failure: dict | None = None
        if completed.returncode == 0:
            try:
                record = _record_result(
                    job,
                    output_root=output_root,
                    log_path=log_path,
                    seed=seed,
                    gpu=gpu,
                    elapsed=elapsed,
                    config_sha256=config_sha256,
                )
                marker.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(
                    f"DONE gpu={gpu} repeat={job.repeat:02d} fold={job.fold:02d} "
                    f"val={record['validation_accuracy']:.4f} "
                    f"test={record['test_accuracy']:.4f} seconds={elapsed:.1f}",
                    flush=True,
                )
            except Exception as error:
                failure = {
                    "repeat": job.repeat,
                    "fold": job.fold,
                    "gpu": gpu,
                    "returncode": 0,
                    "postprocess_error": f"{type(error).__name__}: {error}",
                    "log": str(log_path),
                }
        else:
            failure = {
                "repeat": job.repeat,
                "fold": job.fold,
                "gpu": gpu,
                "returncode": completed.returncode,
                "log": str(log_path),
            }
        if failure is not None:
            with lock:
                failures.append(failure)
                if len(failures) >= max_failures:
                    stop_event.set()
            print(
                f"FAILED gpu={gpu} repeat={job.repeat:02d} fold={job.fold:02d}",
                flush=True,
            )
        jobs.task_done()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=list(range(1, N_FOLDS + 1)),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PAPER4_ROOT / "data" / "bci2020",
        help="BCI2020 source root written into each generated runtime YAML.",
    )
    parser.add_argument("--cpu-threads-per-process", type=int)
    parser.add_argument("--max-failures", type=int, default=1)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--plan", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    folds = tuple(dict.fromkeys(args.folds))
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("GPU list must contain unique values")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if not folds or any(fold < 1 or fold > N_FOLDS for fold in folds):
        raise ValueError("--folds must contain one-based values in 1..15")
    if args.max_failures < 1:
        raise ValueError("--max-failures must be positive")

    base_config = _read_yaml(BASE_CONFIG)
    seed = _validate_base_config(base_config)
    config_sha256 = _sha256(BASE_CONFIG)
    output_root = args.output_root.resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    try:
        available_cpus = tuple(sorted(os.sched_getaffinity(0)))
    except AttributeError:
        available_cpus = tuple(range(os.cpu_count() or 1))
    if len(available_cpus) < len(gpus):
        raise RuntimeError("There are fewer available CPU cores than GPU workers")
    base_size, remainder = divmod(len(available_cpus), len(gpus))
    groups = []
    cursor = 0
    for index in range(len(gpus)):
        size = base_size + int(index < remainder)
        groups.append(tuple(available_cpus[cursor:cursor + size]))
        cursor += size
    cpu_groups = tuple(groups)
    cpu_threads = args.cpu_threads_per_process or min(map(len, cpu_groups))
    if cpu_threads < 1 or cpu_threads > min(map(len, cpu_groups)):
        raise ValueError("Invalid --cpu-threads-per-process for assigned affinities")

    jobs_list = [
        Job(repeat, fold)
        for repeat in range(1, args.repeats + 1)
        for fold in folds
    ]
    print(
        f"LR-HORNet fixed-seed repeats: seed={seed} repeats={args.repeats} "
        f"folds={list(folds)} fits={len(jobs_list)} gpus={list(gpus)} "
        f"dataset_root={dataset_root} output_root={output_root}",
        flush=True,
    )
    for gpu, cpu_ids in zip(gpus, cpu_groups, strict=True):
        print(f"RESOURCE gpu={gpu} cpu_affinity={','.join(map(str, cpu_ids))}")
    if args.plan:
        for sequence, job in enumerate(jobs_list, 1):
            print(
                f"{sequence:03d}/{len(jobs_list):03d} "
                f"repeat={job.repeat:02d} fold={job.fold:02d} seed={seed}"
            )
        return

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    DONE_ROOT.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    failure_path = LOG_ROOT / "failures.json"
    if failure_path.is_file():
        failure_path.unlink()
    work: queue.Queue[Job] = queue.Queue()
    for job in jobs_list:
        work.put(job)
    failures: list[dict] = []
    lock = threading.Lock()
    stop_event = threading.Event()
    threads = [
        threading.Thread(
            target=_worker,
            args=(
                gpu,
                cpu_ids,
                work,
                base_config,
                output_root,
                dataset_root,
                seed,
                config_sha256,
                cpu_threads,
                args.rerun,
                failures,
                lock,
                stop_event,
                args.max_failures,
            ),
            name=f"gpu-{gpu}",
        )
        for gpu, cpu_ids in zip(gpus, cpu_groups, strict=True)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        failure_path.write_text(
            json.dumps(failures, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise SystemExit(
            f"{len(failures)} fit(s) failed; {work.qsize()} queued fit(s) "
            f"were not started; inspect {failure_path}"
        )
    print(f"ALL DONE fits={len(jobs_list)} seed={seed}", flush=True)


if __name__ == "__main__":
    main()

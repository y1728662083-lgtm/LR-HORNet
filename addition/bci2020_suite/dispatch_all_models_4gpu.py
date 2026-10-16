"""Dispatch the six archived Paper4 models over four physical GPUs."""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import yaml


PAPER4_ROOT = Path(__file__).resolve().parents[2]
SUITE_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = SUITE_ROOT / "runtime_configs"
LOG_ROOT = PAPER4_ROOT / "logs" / "bci2020_all_models"
DONE_ROOT = LOG_ROOT / ".done"
ALL_MODELS = ("lr_hornet", "eegnet", "atcnet", "eegmamba", "fast", "labram")
N_FOLDS = 15


@dataclass(frozen=True)
class Job:
    model: str
    fold: int


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def _resolve_weight(filename: str, baseline: str) -> Path | None:
    candidates = (
        PAPER4_ROOT / "baselines" / baseline / "addition" / "weights" / filename,
        PAPER4_ROOT / "addition" / "weights" / filename,
        PAPER4_ROOT / "weights" / filename,
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def prepare_runtime_configs(
    *,
    require_weights: bool,
    cpu_threads: int,
    loader_workers: int,
    dataset_root: Path,
    output_root: Path,
) -> dict[str, Path]:
    """Create immutable-per-launch configs and resolve existing large weights."""
    configs = {
        "lr_hornet": PAPER4_ROOT / "configs" / "main" / "bci2020.yaml",
        "eegnet": PAPER4_ROOT / "baselines" / "eegnet" / "configs" / "bci2020.yaml",
        "atcnet": PAPER4_ROOT / "baselines" / "atcnet" / "configs" / "bci2020.yaml",
        "eegmamba": PAPER4_ROOT / "baselines" / "eegmamba" / "addition" / "configs" / "bci2020_official_baselines.yaml",
        "fast": PAPER4_ROOT / "baselines" / "fast" / "addition" / "configs" / "fast_native_bci2020.yaml",
        "labram": PAPER4_ROOT / "baselines" / "labram" / "addition" / "configs" / "bci2020_official_baselines.yaml",
    }
    missing = [str(path) for path in configs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model configs:\n" + "\n".join(missing))

    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    runtime: dict[str, Path] = {}
    for model, source in configs.items():
        payload = _read_yaml(source)
        payload["data"]["root"] = str(dataset_root)
        payload["training"]["output_root"] = str(output_root / model)
        if model in {"lr_hornet", "eegnet", "atcnet"}:
            # MNE Epochs keep lazy file readers which are not safe to share with
            # persistent multiprocessing workers.  Cache each standardized
            # split once in RAM, then train from tensors in the main process.
            payload["loader"]["cache_preprocessed_trials"] = True
            payload["loader"]["num_workers"] = 0
            payload["loader"]["persistent_workers"] = False
            payload["loader"]["prefetch_factor"] = 2
        else:
            payload["training"]["cpu_threads_per_process"] = cpu_threads
            # Official baselines first materialize standardized trials in RAM;
            # their tensor-backed loaders do not benefit from worker processes.
            payload["training"]["data_loader_workers"] = 0
        if model == "eegmamba":
            weight = _resolve_weight("pretrained_EEGMamba.pth", model)
            if weight is not None:
                payload["models"][model]["checkpoint"] = str(weight)
            elif require_weights:
                raise FileNotFoundError(
                    "Missing pretrained_EEGMamba.pth. Put it in "
                    "baselines/eegmamba/addition/weights/ or addition/weights/."
                )
        elif model == "labram":
            weight = _resolve_weight("labram-base-official.safetensors", model)
            if weight is not None:
                payload["models"][model]["checkpoint"] = str(weight)
            elif require_weights:
                raise FileNotFoundError(
                    "Missing labram-base-official.safetensors. Put it in "
                    "baselines/labram/addition/weights/ or addition/weights/."
                )
        target = RUNTIME_ROOT / f"{model}.yaml"
        _write_yaml(target, payload)
        runtime[model] = target
    return runtime


def _command(job: Job, configs: dict[str, Path]) -> tuple[Path, list[str]]:
    python = sys.executable
    if job.model == "lr_hornet":
        payload = _read_yaml(configs[job.model])
        payload["split"]["fold_indices"] = [job.fold - 1]
        config = RUNTIME_ROOT / f"lr_hornet_fold_{job.fold:02d}.yaml"
        _write_yaml(config, payload)
        return PAPER4_ROOT, [python, "-u", "main.py", "--config", str(config)]
    if job.model in {"eegnet", "atcnet"}:
        payload = _read_yaml(configs[job.model])
        payload["split"]["fold_indices"] = [job.fold - 1]
        config = RUNTIME_ROOT / f"{job.model}_fold_{job.fold:02d}.yaml"
        _write_yaml(config, payload)
        architecture = "EEGNet" if job.model == "eegnet" else "ATCNet"
        return PAPER4_ROOT, [
            python,
            "-u",
            "src/eeg_pipeline/models/baseline_model.py",
            "--config",
            str(config),
            "--architecture",
            architecture,
        ]
    baseline_root = PAPER4_ROOT / "baselines" / job.model
    runner = (
        "addition/run_fast_native_all_datasets.py"
        if job.model == "fast"
        else "addition/run_multidataset_official_baselines.py"
    )
    return baseline_root, [
        python,
        "-u",
        runner,
        "--config",
        str(configs[job.model]),
        "--models",
        job.model,
        "--folds",
        str(job.fold),
    ]


def _job_env(job: Job, gpu: str, cpu_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(cpu_threads)
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    env["OMP_PROC_BIND"] = "close"
    env["OMP_PLACES"] = "cores"
    if job.model in {"lr_hornet", "eegnet", "atcnet"}:
        paths = [PAPER4_ROOT / "src", PAPER4_ROOT]
    else:
        baseline = PAPER4_ROOT / "baselines" / job.model
        paths = [baseline / "src", baseline, baseline / "addition"]
    old = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(map(str, paths)) + (os.pathsep + old if old else "")
    return env


def _done_marker(job: Job) -> Path:
    return DONE_ROOT / f"{job.model}_fold_{job.fold:02d}.json"


def _worker(
    gpu: str,
    cpu_ids: tuple[int, ...],
    jobs: queue.Queue[Job],
    configs: dict[str, Path],
    cpu_threads: int,
    rerun: bool,
    failures: list[dict],
    lock: threading.Lock,
    stop_event: threading.Event,
    max_failures: int,
) -> None:
    while True:
        if stop_event.is_set():
            return
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        marker = _done_marker(job)
        if marker.is_file() and not rerun:
            print(f"SKIP gpu={gpu} model={job.model} fold={job.fold:02d} (done marker)", flush=True)
            jobs.task_done()
            continue
        cwd, command = _command(job, configs)
        log_path = LOG_ROOT / f"{job.model}_fold_{job.fold:02d}.log"
        cpu_spec = ",".join(str(value) for value in cpu_ids)
        print(
            f"START gpu={gpu} cpus={cpu_spec} model={job.model} "
            f"fold={job.fold:02d} log={log_path}",
            flush=True,
        )
        started = time.time()
        run_command = command
        if os.name == "posix" and cpu_ids and shutil.which("taskset"):
            run_command = ["taskset", "-c", cpu_spec, *command]
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            log.write("COMMAND " + subprocess.list2cmdline(run_command) + "\n")
            log.write(f"CUDA_VISIBLE_DEVICES={gpu}\n")
            log.write(f"CPU_AFFINITY={cpu_spec}\n")
            log.write(f"CPU_THREADS={cpu_threads}\n")
            completed = subprocess.run(
                run_command,
                cwd=cwd,
                env=_job_env(job, gpu, cpu_threads),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        elapsed = time.time() - started
        if completed.returncode == 0:
            marker.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "model": job.model,
                        "fold": job.fold,
                        "gpu": gpu,
                        "elapsed_seconds": elapsed,
                        "log": str(log_path),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"DONE gpu={gpu} model={job.model} fold={job.fold:02d} seconds={elapsed:.1f}", flush=True)
        else:
            failure = {
                "model": job.model,
                "fold": job.fold,
                "gpu": gpu,
                "returncode": completed.returncode,
                "log": str(log_path),
            }
            with lock:
                failures.append(failure)
                if len(failures) >= max_failures:
                    stop_event.set()
            print(f"FAILED gpu={gpu} model={job.model} fold={job.fold:02d} rc={completed.returncode}", flush=True)
        jobs.task_done()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS))
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=list(range(1, N_FOLDS + 1)),
        help="One-based folds to run. Defaults to all 15 folds.",
    )
    parser.add_argument(
        "--cpu-threads-per-process",
        type=int,
        default=None,
        help="Defaults to available CPU cores per GPU minus loader workers.",
    )
    parser.add_argument(
        "--loader-workers-per-process",
        type=int,
        default=None,
        help="Compatibility option; must be 0 because lazy MNE readers are cached in-process.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PAPER4_ROOT / "data" / "bci2020",
        help="BCI2020 source root; runtime YAML files are generated with this path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PAPER4_ROOT / "outputs" / "bci2020_all_models",
        help="Destination shared by the six model families.",
    )
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--max-failures",
        type=int,
        default=1,
        help="Stop assigning new jobs after this many failures (default: 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if not gpus:
        raise ValueError("At least one GPU must be supplied")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"GPU list contains duplicates: {gpus}")
    folds = tuple(dict.fromkeys(args.folds))
    if not folds or any(fold < 1 or fold > N_FOLDS for fold in folds):
        raise ValueError(f"--folds must contain one-based values in 1..{N_FOLDS}")
    if args.max_failures < 1:
        raise ValueError("--max-failures must be positive")
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    try:
        available_cpu_ids = tuple(sorted(os.sched_getaffinity(0)))
    except AttributeError:
        available_cpu_ids = tuple(range(os.cpu_count() or 1))
    if len(available_cpu_ids) < len(gpus):
        raise RuntimeError(
            f"Only {len(available_cpu_ids)} CPUs are available for {len(gpus)} GPU workers"
        )
    base_size, remainder = divmod(len(available_cpu_ids), len(gpus))
    groups: list[tuple[int, ...]] = []
    cursor = 0
    for index in range(len(gpus)):
        size = base_size + int(index < remainder)
        groups.append(tuple(available_cpu_ids[cursor:cursor + size]))
        cursor += size
    cpu_groups = tuple(groups)
    minimum_group_size = min(map(len, cpu_groups))
    loader_workers = args.loader_workers_per_process
    if loader_workers is None:
        loader_workers = 0
    if loader_workers != 0:
        raise ValueError(
            "--loader-workers-per-process must be 0: persistent MNE worker "
            "processes can corrupt lazy epoch reads"
        )
    cpu_threads = args.cpu_threads_per_process
    if cpu_threads is None:
        cpu_threads = max(1, minimum_group_size - loader_workers)
    if cpu_threads < 1:
        raise ValueError("--cpu-threads-per-process must be positive")
    if cpu_threads + loader_workers > minimum_group_size:
        raise ValueError(
            "CPU oversubscription requested: cpu_threads + loader_workers "
            f"= {cpu_threads + loader_workers}, but only {minimum_group_size} cores/GPU are available"
        )
    configs = prepare_runtime_configs(
        require_weights=not args.plan,
        cpu_threads=cpu_threads,
        loader_workers=loader_workers,
        dataset_root=dataset_root,
        output_root=output_root,
    )
    job_list = [Job(model, fold) for fold in folds for model in args.models]
    print(
        f"BCI2020 calibrated-LOSO plan: models={list(args.models)} folds={list(folds)} "
        f"fits={len(job_list)} physical_gpus={list(gpus)} "
        f"available_cpus={len(available_cpu_ids)} cpu_threads/process={cpu_threads} "
        f"loader_workers/process={loader_workers} dataset_root={dataset_root} "
        f"output_root={output_root}",
        flush=True,
    )
    for gpu, cpu_ids in zip(gpus, cpu_groups, strict=True):
        print(f"RESOURCE gpu={gpu} cpu_affinity={','.join(map(str, cpu_ids))}")
    if args.plan:
        for index, job in enumerate(job_list, 1):
            gpu = gpus[(index - 1) % len(gpus)]
            cwd, command = _command(job, configs)
            print(f"{index:03d}/{len(job_list):03d} gpu~{gpu} model={job.model} fold={job.fold:02d} cwd={cwd}")
            print("  " + subprocess.list2cmdline(command))
        return

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    DONE_ROOT.mkdir(parents=True, exist_ok=True)
    failure_path = LOG_ROOT / "failures.json"
    if failure_path.is_file():
        failure_path.unlink()
    work: queue.Queue[Job] = queue.Queue()
    for job in job_list:
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
                configs,
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
        failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        unstarted = work.qsize()
        raise SystemExit(
            f"{len(failures)} fit(s) failed; {unstarted} queued fit(s) were not "
            f"started; inspect {failure_path}"
        )
    print(f"ALL DONE fits={len(job_list)}", flush=True)


if __name__ == "__main__":
    main()

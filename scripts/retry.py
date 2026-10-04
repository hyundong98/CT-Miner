"""Retry errors/interruptions of finished focused sweeps; retain timeouts.

User-run only. Successful jobs (including insufficient-feature results) are
untouched; timeouts are reported but never retried. Source fingerprints must
match; build manifests may differ only in path/serialization details.
Original root config remains original provenance;
each retry has its actual config/fingerprint and a retry_runs audit record.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from ctminer.paths import code_hashes
import queue
import shutil
import signal
import subprocess
import sys
import threading
from time import perf_counter
import traceback
import uuid

from ctminer.study import HERE, read, write, fingerprint, summarize
from ctminer.resources import cpu_slots, TreeMeter
from ctminer.reuse import compatible_code, compatible_build
from ctminer.memory import VERSION as MEMORY_VERSION

RETRY_STATUSES = {"failed", "interrupted"}
TERMINAL_STATUSES = RETRY_STATUSES | {"complete", "timeout"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_changes(saved, current):
    return compatible_code(saved, current)


def inspect(roots):
    states, selected = {}, []
    for root in roots:
        index = read(root / "job_index.json")
        cfg = read(root / "config.json")
        counts = Counter()
        for entry in index:
            out = (root / entry["path"]).resolve()
            if not out.is_relative_to(root) or out == root:
                raise ValueError(f"Invalid job path: {out}")
            result = (
                read(out / "result.json")
                if (out / "result.json").exists()
                else {"status": "pending"}
            )
            status = result.get("status", "unknown")
            counts[status] += 1
            if status == "pending" or status == "running":
                raise ValueError(
                    f"Sweep not finished ({status}): {out}. Finish the original sweep first."
                )
            if status not in TERMINAL_STATUSES:
                raise ValueError(f"Unknown result status {status}: {out}")
            if result.get("job_fingerprint") != entry["fingerprint"]:
                raise ValueError(f"Result fingerprint mismatch: {out}")
            if status not in RETRY_STATUSES:
                continue
            job = read(out / "job.json")
            unsigned = {
                k: v for k, v in job.items() if k not in {"fingerprint", "execution"}
            }
            if (
                job.get("fingerprint") != entry["fingerprint"]
                or fingerprint(unsigned) != entry["fingerprint"]
            ):
                raise ValueError(f"Job fingerprint mismatch: {out}")
            for key in (
                "dataset",
                "engine",
                "budget",
                "study",
                "length_percent",
                "repeat",
                "feature_mode",
                "minsup",
                "minconf",
                "maxsta",
            ):
                if job.get(key) != entry.get(key):
                    raise ValueError(f"Index/job mismatch ({key}): {out}")
            selected.append(
                dict(root=root, out=out, entry=entry, job=job, previous=result)
            )
        if len({entry["path"] for entry in index}) != len(index):
            raise ValueError(f"Duplicate job paths: {root}")
        states[root] = dict(index=index, config=cfg, counts=dict(counts))
        print(f"{root}: {dict(counts)}", flush=True)
    return states, selected


def summarize_with_timeouts(root):
    summarize(root)
    timeouts = []
    for entry in read(root / "job_index.json"):
        path = root / entry["path"] / "result.json"
        result = read(path) if path.exists() else {}
        if result.get("status") == "timeout":
            timeouts.append(dict(entry, reason=result.get("reason", "timeout")))
    write(root / "timeouts.json", timeouts)
    lines = [
        "",
        f"Timeouts: {len(timeouts)} jobs; retained without retry.[^timeout-policy]",
        "",
        "[^timeout-policy]: Timeout jobs have no completed performance/runtime measurement. "
        "Scores are not imputed as zero, and cutoff times are not averaged as completed runtimes. "
        "Observed elapsed time and RSS remain in jobs.csv for audit only. "
        "Macro summaries exclude Wafer and average valid datasets separately for each method and condition. "
        "A dataset contributes only when every planned repetition is valid for that method and condition. "
        "Missing results do not exclude other methods or conditions; unavailable PCA results are shown as dashes. "
        "Dataset counts and membership are recorded in summary.csv; per-dataset results retain Wafer.",
    ]
    if timeouts:
        lines += [
            "",
            "| Dataset | Study | Method | K | Prefix % | Repeat | Timeout |",
            "|---|---|---|---:|---:|---:|---|",
        ]
        for row in timeouts:
            reason = (
                row["reason"].split(";", 1)[0].replace("|", "\\|").replace("\n", " ")
            )
            lines.append(
                f"| {row['dataset']} | {row['study']} | {row['engine']} | {row['budget']} | {row['length_percent']} | {row['repeat']} | {reason} |"
            )
    with (root / "summary.md").open("a") as stream:
        stream.write("\n".join(lines) + "\n")


def execute(item, lane, execution, stopping, run_id):
    out, job = item["out"], item["new_job"]
    cfg = job["config"]
    execution = dict(execution, sample_interval=cfg["sample_interval"])
    history = out / "attempts" / run_id
    job["execution"] = dict(
        execution,
        lane=lane,
        cpus=execution["cpu_slots"][lane],
        threads_per_job=cfg["threads"],
        retry_run_id=run_id,
    )
    try:
        history.mkdir(parents=True, exist_ok=False)
        # The audit was written before changing a job; keep its complete old attempt.
        for old in list(out.iterdir()):
            if old.name != "attempts":
                shutil.move(str(old), history / old.name)
        write(out / "job.json", job)
        write(
            out / "result.json",
            dict(status="running", job_fingerprint=job["fingerprint"]),
        )
    except Exception as exc:
        write(out / "job.json", job)
        write(
            out / "result.json",
            dict(
                status="failed",
                reason=f"Retry setup failed: {exc}",
                traceback=traceback.format_exc(),
                job_fingerprint=job["fingerprint"],
                retry_run_id=run_id,
            ),
        )
        return "failed"
    env = dict(os.environ)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env[key] = str(cfg["threads"])
    started = perf_counter()
    meter = proc = None
    timed_out = interrupted = False
    error = None
    try:
        with (out / "worker.log").open("w") as log:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "ctminer.study",
                    "_worker",
                    str(out / "job.json"),
                ],
                cwd=HERE.parent,
                env=env,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            meter = TreeMeter(proc.pid, execution["sample_interval"])
            try:
                while proc.poll() is None:
                    phase = (
                        read(out / "phase.json")["phase"]
                        if (out / "phase.json").exists()
                        else "startup"
                    )
                    meter.sample(phase)
                    if (
                        stopping.is_set()
                        or perf_counter() - started >= cfg["job_timeout"]
                    ):
                        interrupted = stopping.is_set()
                        timed_out = not interrupted
                        meter.signal()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            pass
                        meter.signal(signal.SIGKILL)
                        proc.wait()
                        break
                    stopping.wait(execution["sample_interval"])
            finally:
                if proc.poll() is None:
                    meter.signal(signal.SIGKILL)
                    proc.wait()
                meter.signal(signal.SIGKILL)
    except Exception as exc:
        error = dict(reason=str(exc), traceback=traceback.format_exc())
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    resources = dict(
        meter.result() if meter else {},
        elapsed_wall_seconds=perf_counter() - started,
        returncode=proc.returncode if proc else None,
        execution=job["execution"],
    )
    if (out / "worker_usage.json").exists():
        resources.update(read(out / "worker_usage.json"))
    if (out / "feature_memory.json").exists():
        resources["feature_memory"] = read(out / "feature_memory.json")
    write(out / "resources.json", resources)
    result = (
        read(out / "result.json")
        if (out / "result.json").exists()
        else dict(status="failed", reason="Worker result missing")
    )
    if error:
        result.update(status="failed", **error)
    elif timed_out or interrupted:
        result.update(
            status="interrupted" if interrupted else "timeout",
            reason="retry supervisor interrupted"
            if interrupted
            else f"whole job exceeded {cfg['job_timeout']} seconds",
        )
    elif result.get("status") == "running" or (
        proc.returncode != 0 and result.get("status") == "complete"
    ):
        result.update(
            status="failed", reason=f"worker exited with code {proc.returncode}"
        )
    elif (
        result.get("status") == "failed"
        and "exceeded" in result.get("reason", "")
        and "seconds" in result.get("reason", "")
    ):
        result["status"] = "timeout"
    result.update(
        resources=resources,
        study=job["study"],
        job_fingerprint=job["fingerprint"],
        retry_run_id=run_id,
    )
    write(out / "result.json", result)
    return result["status"]


def run(args):
    if sys.platform != "linux":
        raise RuntimeError("Retry execution requires Linux /proc")
    import fcntl

    roots = sorted({p.resolve() for p in args.outputs})
    if min(args.jobs, args.cores_per_job) < 1:
        raise ValueError("Positive jobs and cores-per-job required")
    with ExitStack() as stack:
        # Hold all original locks, even for --list-only: never race a live sweep.
        for root in roots:
            lock = stack.enter_context((root / "run.lock").open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(
                    f"A supervisor is still using {root}; wait for both sweeps to finish"
                )
        states, items = inspect(roots)
        skipped_timeouts = sum(
            state["counts"].get("timeout", 0) for state in states.values()
        )
        print(
            f"Retry candidates: {len(items)}; timeouts skipped: {skipped_timeouts}; successful jobs are untouched.",
            flush=True,
        )
        for item in items:
            job = item["job"]
            print(
                f"  {job['dataset']} / {job['engine']} / K{job['budget']} / L{job['length_percent']} / {item['previous']['status']}"
            )
        if args.list_only:
            return
        if args.summarize_only or not items:
            for root in roots:
                summarize_with_timeouts(root)
            return
        from ctminer.engines import _java_ready, CLASSES
        from ctminer.worker import package_versions

        _java_ready()
        slots = cpu_slots(args.jobs, args.cores_per_job, "physical")
        current_code = code_hashes()
        packages = package_versions()
        build = digest(CLASSES / "manifest.json")
        checked_inputs = {}
        for item in items:
            cfg = item["job"]["config"]
            if cfg.get("memory_protocol") != MEMORY_VERSION:
                raise ValueError("Memory protocol changed; use a new output directory")
            item["source_changes"] = source_changes(cfg["code"], current_code)
            if cfg["packages"] != packages:
                raise ValueError(
                    "Package versions changed; cannot merge this retry into old results"
                )
            build_change = compatible_build(
                cfg, dict(cfg, build_manifest_sha256=build), item["job"]["input"]
            )
            if build_change:
                item["source_changes"]["build_manifest_sha256"] = build_change
            if cfg["threads"] > args.cores_per_job:
                raise ValueError("Stored thread count exceeds allocated physical cores")
            inp = item["job"]["input"]
            if inp not in checked_inputs:
                checked_inputs[inp] = digest(inp)
            if checked_inputs[inp] != item["job"]["input_sha256"]:
                raise ValueError(f"Input data changed: {inp}")
            new_job = deepcopy(item["job"])
            new_job.pop("execution", None)
            new_job.pop("fingerprint", None)
            new_job["config"]["code"] = current_code
            new_job["config"]["build_manifest_sha256"] = build
            new_job["fingerprint"] = fingerprint(new_job)
            item["new_job"] = new_job
        run_id = (
            datetime.now(timezone.utc).strftime("retry-%Y%m%dT%H%M%SZ-")
            + uuid.uuid4().hex[:8]
        )
        execution = dict(
            started_utc=datetime.now(timezone.utc).isoformat(),
            concurrent_jobs=args.jobs,
            cores_per_job=args.cores_per_job,
            affinity="physical",
            cpu_slots=slots,
            pending_jobs=len(items),
            hostname="anonymous",
        )
        # Record both versions before changing index entries or attempt files.
        for root, state in states.items():
            audit_items = [
                dict(
                    path=i["entry"]["path"],
                    previous_job=i["job"],
                    previous_status=i["previous"]["status"],
                    new_job=i["new_job"],
                    source_changes=i["source_changes"],
                )
                for i in items
                if i["root"] == root
            ]
            if audit_items:
                write(
                    root / "retry_runs" / (run_id + ".json"),
                    dict(
                        execution=execution,
                        jobs=audit_items,
                        policy="Only errors/interruptions retried; timeouts retained without retry. Original config retained; per-job config records actual source. Source fingerprints and scientific settings must match; only equivalent build paths/serialization may differ.",
                    ),
                )
                (root / "RETRY_NOTES.md").write_text(
                    "# Retry provenance\n\n"
                    "config.json describes the original sweep. Successful jobs are untouched. "
                    "Retried jobs record their actual code in job.json; job_index.json tracks the latest fingerprints. "
                    "See retry_runs/*.json for prior/current jobs and source changes, and attempts/ for old outputs. "
                    "Source fingerprints, scientific settings, input data and package versions are unchanged. "
                    "Java build content must match; only recorded paths/serialization may differ.\n\n"
                    "Timeouts are never retried. They are retained in timeouts.json and summary footnotes, "
                    "and excluded from completed-result averages without zero imputation.\n\n"
                    "Timing summaries use the latest attempt per job, not cumulative compute cost. "
                    "Retries occurred separately from original successes; consult execution_history.jsonl "
                    "and per-job execution metadata when reporting timing. Use `python -m scripts.retry OUTPUT` for further retries; "
                    "do not resume this retry-managed directory through the regular experiment runner.\n"
                )
                with (root / "execution_history.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            dict(
                                execution,
                                retry_run_id=run_id,
                                retry_jobs_in_root=len(audit_items),
                            )
                        )
                        + "\n"
                    )
        pending = queue.Queue()
        for item in items:
            pending.put(item)
        stopping, mutex = threading.Event(), threading.Lock()
        completed = [0]

        def lane_worker(lane):
            while not stopping.is_set():
                try:
                    item = pending.get_nowait()
                except queue.Empty:
                    return
                try:
                    # Update one entry immediately before its attempt, not queued jobs.
                    with mutex:
                        item["entry"]["fingerprint"] = item["new_job"]["fingerprint"]
                        write(
                            item["root"] / "job_index.json",
                            states[item["root"]]["index"],
                        )
                    status = execute(item, lane, execution, stopping, run_id)
                    with mutex:
                        completed[0] += 1
                        print(
                            f"[{completed[0]}/{len(items)}] {item['out']} / {status}",
                            flush=True,
                        )
                except BaseException:
                    stopping.set()
                    raise

        def terminate(signum, frame):
            stopping.set()
            raise KeyboardInterrupt("Retry interrupted")

        signal.signal(signal.SIGTERM, terminate)
        pool = ThreadPoolExecutor(max_workers=args.jobs)
        futures = [pool.submit(lane_worker, lane) for lane in range(args.jobs)]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            stopping.set()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        for root in roots:
            summarize_with_timeouts(root)
        print(
            "Retry finished; summaries refreshed. Further retries must use scripts.retry."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "outputs",
        nargs="+",
        type=Path,
        help="Finished quality and/or length output directories",
    )
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--cores-per-job", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--list-only",
        action="store_true",
        help="List retryable errors only and count skipped timeouts; no mining",
    )
    mode.add_argument(
        "--summarize-only",
        action="store_true",
        help="Refresh summaries with timeout footnotes; no retries or mining",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()

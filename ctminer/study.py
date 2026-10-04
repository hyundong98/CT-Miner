"""Independent feature extraction and clustering sweeps."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
from time import perf_counter
import traceback

from ctminer.paths import code_hashes
from ctminer.memory import VERSION as MEMORY_VERSION, sampled_peak_bytes

HERE = Path(__file__).resolve().parent
DATASETS = ["Car", "Beef", "ElectricDevices", "MoteStrain", "PigCVP", "Wafer"]
ALLOWED_DATASETS = set(DATASETS) | {
    "PigAirwayPressure",
    "PigArtPressure",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "TwoLeadECG",
    "CinCECGTorso",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
}
MINERS = ["CT"]
METHODS = ["Raw", "PCA", "CT"]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def specifications(args):
    for dataset in args.datasets:
        for study in ["quality", "length"] if args.study == "both" else [args.study]:
            for percent in [100] if study == "quality" else args.length_percents:
                for engine in args.methods:
                    if study == "length" and engine not in MINERS:
                        continue
                    # References have no support/confidence/stability threshold.
                    # Never repeat them for irrelevant grid dimensions.
                    thresholds = (
                        args.minsups
                        if args.mining_protocol == "native" and engine in MINERS
                        else [None]
                    )
                    confidences = [None]
                    stabilities = [None]
                    budgets = (
                        [0]
                        if engine == "Raw"
                        else [args.pca_components]
                        if args.feature_mode == "threshold" and engine == "PCA"
                        else [0]
                        if args.feature_mode == "threshold"
                        else args.budgets
                    )
                    for minsup in thresholds:
                        for minconf in confidences:
                            for maxsta in stabilities:
                                for budget in budgets:
                                    for repeat in range(
                                        args.repeat_start,
                                        args.repeat_start + args.repeats,
                                    ):
                                        yield dict(
                                            dataset=dataset,
                                            engine=engine,
                                            budget=budget,
                                            study=study,
                                            length_percent=percent,
                                            repeat=repeat,
                                            feature_mode=args.feature_mode,
                                            minsup=minsup,
                                            minconf=minconf,
                                            maxsta=maxsta,
                                        )


def validate_args(args):
    if args.repeat_start < 0:
        raise ValueError("--repeat-start must be nonnegative")
    if (
        min(
            args.jobs,
            args.cores_per_job,
            args.threads,
            args.repeats,
            args.n_init,
            args.pattern_cap,
        )
        < 1
    ):
        raise ValueError(
            "Positive job/core/thread/repetition/init/cap settings required"
        )
    if args.threads > args.cores_per_job and args.affinity == "physical":
        raise ValueError("threads exceeds allocated physical cores")
    if not (
        math.isfinite(args.java_timeout)
        and math.isfinite(args.job_timeout)
        and 0 < args.java_timeout < args.job_timeout
    ):
        raise ValueError("Require 0 < java-timeout < job-timeout")
    if not math.isfinite(args.sample_interval) or args.sample_interval < 0.02:
        raise ValueError("RSS sampling interval must be finite and >= 0.02 seconds")
    if not re.fullmatch(r"[1-9][0-9]*[kKmMgG]", args.java_heap):
        raise ValueError("Invalid Java heap size")
    for name in ("datasets", "methods", "budgets", "seeds", "length_percents"):
        values = getattr(args, name)
        if not values or len(set(values)) != len(values):
            raise ValueError(f"Nonempty unique {name} required")
    if not set(args.datasets) <= ALLOWED_DATASETS:
        raise ValueError("Dataset is outside the current or legacy focused-study list")
    if not set(args.require_reuse_datasets) <= set(args.datasets):
        raise ValueError(
            "Required reuse datasets must belong to the requested dataset list"
        )
    if any(b not in (10, 20, 30, 40, 50) for b in args.budgets) or min(args.seeds) < 0:
        raise ValueError("Use budgets 10/20/30/40/50 and nonnegative seeds")
    if args.mining_protocol == "legacy" and args.lengths is None:
        args.lengths = list(range(2, 52))
    if args.lengths is not None:
        limit = 1_000_000 if args.mining_protocol == "native" else 63
        if (
            not args.lengths
            or len(set(args.lengths)) != len(args.lengths)
            or not 2 <= min(args.lengths) <= max(args.lengths) <= limit
        ):
            raise ValueError(f"Unique pattern lengths in 2..{limit} required")
    if not 1 <= min(args.length_percents) <= max(args.length_percents) <= 100:
        raise ValueError("Prefix percentages must lie in 1..100")
    if (
        not args.minsups
        or len(set(args.minsups)) != len(args.minsups)
        or any(not math.isfinite(v) or v <= 0 for v in args.minsups)
    ):
        raise ValueError("Unique finite positive --minsups required")
    if args.mining_protocol != "native" or not set(args.methods) <= set(METHODS):
        raise ValueError("Only native CT, Raw and PCA are supported")
    if min(args.pca_components, args.max_feature_cells) < 1:
        raise ValueError("Positive PCA components and feature matrix cell cap required")
    if args.study in {"length", "both"} and not set(args.methods) & set(MINERS):
        raise ValueError("Length study requires at least one miner")


def plan(args):
    validate_args(args)
    specs = list(specifications(args))
    counts = Counter(s["study"] for s in specs)
    manifest_file = args.prepared / "manifest.json"
    manifest = read(manifest_file) if manifest_file.exists() else {}
    data = manifest.get("datasets", {})
    print(
        json.dumps(
            dict(
                datasets=args.datasets,
                dataset_count=len(args.datasets),
                methods=args.methods,
                independent_budgets=args.budgets,
                feature_mode=args.feature_mode,
                minsups=args.minsups,
                mining_protocol=args.mining_protocol,
                jobs_by_study=dict(counts),
                total_jobs=len(specs),
                quality_clustering_evaluations=0
                if args.features_only
                else (counts["quality"] + counts["prefix-quality"]) * len(args.seeds),
                repetitions=args.repeats,
                concurrent_jobs=args.jobs,
                cores_per_job=args.cores_per_job,
                java_timeout_seconds=args.java_timeout,
                job_timeout_seconds=args.job_timeout,
                length_policy="fixed sample count; prefix floor(original_length * percent / 100), minimum 2; no concatenation/resampling",
                length_percents=args.length_percents,
                output_lengths=args.lengths or "all input lengths",
                not_ready=[
                    name
                    for name in args.datasets
                    if data.get(name, {}).get("status") != "ready"
                ],
                reuse_from=[str(p) for p in args.reuse_from],
                require_reuse_datasets=args.require_reuse_datasets,
                note="Counts are before reuse. run validates reusable jobs before launching any workers. No mining/build/check was executed.",
            ),
            indent=2,
        )
    )


def worker(path):
    # Set affinity before importing NumPy/sklearn and before creating Java.
    import resource

    job = read(path)
    try:
        cpus = job["execution"]["cpus"]
        if cpus is not None:
            os.sched_setaffinity(0, set(cpus))
        from ctminer.worker import worker as archive_worker

        archive_worker(path)
    except BaseException as exc:
        result_path = Path(path).parent / "result.json"
        old = read(result_path) if result_path.exists() else {}
        if old.get("status") not in {"failed", "complete"}:
            write(
                result_path,
                dict(
                    status="failed",
                    job_fingerprint=job["fingerprint"],
                    reason=str(exc),
                    traceback=traceback.format_exc(),
                ),
            )
        raise
    finally:
        own, child = (
            resource.getrusage(resource.RUSAGE_SELF),
            resource.getrusage(resource.RUSAGE_CHILDREN),
        )
        write(
            Path(path).parent / "worker_usage.json",
            dict(
                worker_cpu_seconds=own.ru_utime
                + own.ru_stime
                + child.ru_utime
                + child.ru_stime,
                worker_user_seconds=own.ru_utime + child.ru_utime,
                worker_system_seconds=own.ru_stime + child.ru_stime,
                actual_affinity=sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None,
                scope="getrusage self + reaped children (includes Java); not a simultaneous RSS measurement",
            ),
        )


def run(args):
    validate_args(args)
    if sys.platform != "linux":
        raise RuntimeError(
            "Measured run requires Linux /proc. plan and summarize are portable."
        )
    from ctminer.resources import cpu_slots, TreeMeter
    from ctminer.engines import _java_ready, CLASSES
    from ctminer.worker import package_versions
    from ctminer.reuse import ReuseCatalog, existing_reuse
    from scripts.build import verify_java

    slots = cpu_slots(args.jobs, args.cores_per_job, args.affinity)
    # Read-only checks; never auto-build or run a miner as preflight.
    _java_ready()
    verify_java()
    manifest = read(args.prepared / "manifest.json")
    for name in args.datasets:
        record = manifest["datasets"].get(name, {})
        if record.get("status") != "ready":
            raise ValueError(f"Dataset not ready: {name}")
        if digest(args.prepared / record["npz"]) != record["sha256"]:
            raise ValueError(f"Prepared data changed: {name}")
    cfg = dict(
        budgets=sorted(args.budgets),
        seeds=args.seeds,
        n_init=args.n_init,
        mining_protocol=args.mining_protocol,
        feature_mode=args.feature_mode,
        minsups=args.minsups,
        ct_execution_version="ct_native_strict_id_v2"
        if args.mining_protocol == "native" and set(args.methods) & {"CT", "CT-ID"}
        else None,
        protocol_version="native_objectives_v1"
        if args.mining_protocol == "native"
        else "legacy_frequency_v1",
        design_version="method_specific_grid_v2",
        memory_protocol=MEMORY_VERSION,
        sample_interval=args.sample_interval,
        max_feature_cells=args.max_feature_cells,
        pca_components=args.pca_components,
        scaling="standard",
        lengths=sorted(args.lengths) if args.lengths is not None else None,
        threads=args.threads,
        java_timeout=args.java_timeout,
        job_timeout=args.job_timeout,
        java_heap=args.java_heap,
        java_warmups=0,
        java_repeats=1,
        pattern_cap=args.pattern_cap,
        independent_budgets=True,
        methods=args.methods,
        datasets=args.datasets,
        study=args.study,
        repeats=args.repeats,
        repeat_start=args.repeat_start,
        features_only=args.features_only,
        length_percents=args.length_percents,
        packages=package_versions(),
        input_manifest_sha256=digest(args.prepared / "manifest.json"),
        build_manifest_sha256=digest(CLASSES / "manifest.json"),
        code=code_hashes(),
        scope="unsupervised TRAIN+TEST clustering; labels used only for class-count K and evaluation",
        length_policy="nested within-sample prefixes; fixed sample count; not synthetic long-series concatenation",
    )
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (
        (root / "run.lock").open("a") as lockfile,
        ReuseCatalog(args.reuse_from, root, cfg) as reuse,
    ):
        import fcntl

        try:
            fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another supervisor is using this output directory")
        if (root / "config.json").exists() and read(root / "config.json") != cfg:
            raise ValueError(
                "Configuration, data or code changed: use a new output directory"
            )
        write(root / "config.json", cfg)
        pending, index = queue.Queue(), []
        imported = retained = 0
        for spec in specifications(args):
            data = manifest["datasets"][spec["dataset"]]
            condition = f"{spec['feature_mode']}_S{spec['minsup']}_C{spec['minconf']}_V{spec['maxsta']}"
            relative = (
                Path("jobs")
                / spec["study"]
                / spec["dataset"]
                / spec["engine"]
                / condition
                / f"K{spec['budget']}"
                / f"L{spec['length_percent']}"
                / f"R{spec['repeat']}"
            )
            job = dict(
                **spec,
                input=str((args.prepared / data["npz"]).resolve()),
                input_sha256=data["sha256"],
                config=dict(
                    cfg,
                    budgets=[spec["budget"]],
                    native_minsup=spec["minsup"],
                ),
            )
            job["fingerprint"] = fingerprint(job)
            entry = dict(**spec, path=str(relative), fingerprint=job["fingerprint"])
            index.append(entry)
            out = root / relative
            if (out / "result.json").exists():
                previous = read(out / "result.json")
                reused = existing_reuse(out, job)
                expected = (
                    reused["source_fingerprint"] if reused else job["fingerprint"]
                )
                if previous.get("job_fingerprint") != expected:
                    raise ValueError(f"Job fingerprint mismatch: {out}")
                if reused:
                    entry.update(
                        fingerprint=expected, reused_from=reused["source_job_path"]
                    )
                if previous.get("status") in {"complete", "timeout"} or (
                    previous.get("status") in {"failed", "interrupted"}
                    and not args.retry_failed
                ):
                    if spec["dataset"] in args.require_reuse_datasets and previous.get(
                        "status"
                    ) not in {"complete", "timeout"}:
                        raise ValueError(
                            f"Required reuse is not complete or timeout: {out}"
                        )
                    retained += 1
                    continue
                if reused:
                    raise ValueError(
                        f"Imported terminal result changed: {out}; do not rerun it in place"
                    )
            else:
                reused = reuse.copy(job, out)
                if reused:
                    entry.update(
                        fingerprint=reused["source_fingerprint"],
                        reused_from=reused["source_job_path"],
                    )
                    imported += 1
                    continue
            if spec["dataset"] in args.require_reuse_datasets:
                raise ValueError(
                    f"No reusable completed/timeout result for {spec}; refusing to remine this dataset"
                )
            pending.put((out, job))
        write(root / "job_index.json", index)
        execution = dict(
            started_utc=datetime.now(timezone.utc).isoformat(),
            concurrent_jobs=args.jobs,
            cores_per_job=args.cores_per_job,
            affinity=args.affinity,
            cpu_slots=slots,
            threads_per_job=args.threads,
            sample_interval=args.sample_interval,
            pending_jobs=pending.qsize(),
            reused_jobs=imported,
            retained_jobs=retained,
            planned_jobs=len(index),
            hostname="anonymous",
        )
        print(
            f"Planned {len(index)}; reused {imported}; retained {retained}; new execution {pending.qsize()}",
            flush=True,
        )
        with (root / "execution_history.jsonl").open("a") as stream:
            stream.write(json.dumps(execution) + "\n")
        stopping, print_lock = threading.Event(), threading.Lock()
        total = pending.qsize()
        done = [0]

        def execute(out, job, lane):
            out.mkdir(parents=True, exist_ok=True)
            if (out / "result.json").exists():
                # Keep all evidence of timeouts/failed attempts; never reuse a JVM or features.
                history = out / "attempts"
                history.mkdir(exist_ok=True)
                attempt = history / str(len(list(history.iterdir())) + 1)
                attempt.mkdir()
                for old in list(out.iterdir()):
                    if old.name not in {"attempts", "job.json"}:
                        shutil.move(str(old), attempt / old.name)
            job["execution"] = dict(execution, lane=lane, cpus=slots[lane])
            write(out / "job.json", job)
            write(
                out / "result.json",
                dict(status="running", job_fingerprint=job["fingerprint"]),
            )
            env = dict(os.environ)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS",
            ):
                env[key] = str(args.threads)
            tick = perf_counter()
            timed_out = interrupted = False
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
                meter = TreeMeter(proc.pid, args.sample_interval)
                try:
                    while proc.poll() is None:
                        phase_path = out / "phase.json"
                        phase = (
                            read(phase_path)["phase"]
                            if phase_path.exists()
                            else "startup"
                        )
                        meter.sample(phase)
                        if (
                            stopping.is_set()
                            or perf_counter() - tick >= args.job_timeout
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
                        stopping.wait(args.sample_interval)
                finally:
                    if proc.poll() is None:
                        meter.signal(signal.SIGKILL)
                        proc.wait()
                    # Also clean up tracked descendants if a worker crashed.
                    meter.signal(signal.SIGKILL)
                code = proc.returncode
            resources = dict(
                meter.result(),
                elapsed_wall_seconds=perf_counter() - tick,
                returncode=code,
                execution=job["execution"],
            )
            if (out / "worker_usage.json").exists():
                resources.update(read(out / "worker_usage.json"))
            if (out / "feature_memory.json").exists():
                resources["feature_memory"] = read(out / "feature_memory.json")
            write(out / "resources.json", resources)
            result = read(out / "result.json")
            if timed_out or interrupted:
                result.update(
                    status="interrupted" if interrupted else "timeout",
                    reason="supervisor interrupted"
                    if interrupted
                    else f"whole job exceeded {args.job_timeout} seconds",
                )
            elif result.get("status") == "running" or (
                code != 0 and result.get("status") == "complete"
            ):
                result.update(status="failed", reason=f"worker exited with code {code}")
            elif (
                result.get("status") == "failed"
                and "exceeded" in result.get("reason", "")
                and "seconds" in result.get("reason", "")
            ):
                result["status"] = "timeout"
            result["resources"] = resources
            result["study"] = job["study"]
            result["job_fingerprint"] = job["fingerprint"]
            write(out / "result.json", result)
            with print_lock:
                done[0] += 1
                print(
                    f"[{done[0]}/{total}] {job['study']} / {job['dataset']} / {job['engine']} / {job.get('feature_mode', 'topk')} S{job.get('minsup')} K{job['budget']} / C{job.get('minconf')} V{job.get('maxsta')} / L{job['length_percent']} / {result['status']}",
                    flush=True,
                )

        def lane_worker(lane):
            while not stopping.is_set():
                try:
                    out, job = pending.get_nowait()
                except queue.Empty:
                    return
                try:
                    execute(out, job, lane)
                except BaseException:
                    stopping.set()
                    raise

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
    summarize(root)


def save_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(root):
    root = Path(root)
    cfg = read(root / "config.json")
    jobs, metrics, summaries = [], [], []
    complete = set(cfg["datasets"])
    for entry in read(root / "job_index.json"):
        path = root / entry["path"] / "result.json"
        result = read(path) if path.exists() else dict(status="pending")
        if result.get("job_fingerprint") != entry["fingerprint"] and path.exists():
            raise ValueError(f"Result fingerprint mismatch: {path}")
        resources = result.get("resources", {})
        phase_name = (
            "pca_fit_transform"
            if entry["engine"] == "PCA"
            else "mining_and_feature_construction"
        )
        phase_stats = resources.get("phases", {}).get(phase_name, {})
        memory = resources.get("feature_memory", {})
        peak = sampled_peak_bytes(memory)
        record = dict(
            **entry,
            status=result["status"],
            reason=result.get("reason"),
            wall_seconds=resources.get("elapsed_wall_seconds"),
            cpu_seconds=resources.get("worker_cpu_seconds"),
            sampled_cpu_seconds=resources.get("sampled_cpu_seconds"),
            peak_rss_mib=(
                resources["sampled_peak_rss_bytes"] / 2**20
                if "sampled_peak_rss_bytes" in resources
                else None
            ),
            java_peak_rss_mib=(
                resources["sampled_java_peak_rss_bytes"] / 2**20
                if entry["engine"] in MINERS
                and resources.get("sampled_java_peak_rss_bytes") is not None
                else None
            ),
            feature_phase_peak_rss_mib=(
                phase_stats["sampled_peak_rss_bytes"] / 2**20
                if "sampled_peak_rss_bytes" in phase_stats
                else None
            ),
            feature_peak_rss_mib=peak / 2**20 if peak is not None else None,
            memory_protocol=memory.get("version"),
            memory_status=memory.get("status", "missing"),
        )
        mining_path = path.parent / "mining.json"
        mining = read(mining_path) if mining_path.exists() else {}
        rows = result.get("rows", [])
        first = rows[0] if rows else {}
        record.update(
            {
                k: first.get(k)
                for k in (
                    "sequence_length",
                    "original_length",
                    "total_points",
                    "n_samples",
                )
            }
        )
        record.update(
            native_mining_seconds=mining.get("native_mining_seconds"),
            feature_transform_selection_seconds=mining.get(
                "feature_transform_selection_seconds"
            ),
            feature_seconds=mining.get("invocation_seconds", mining.get("seconds")),
            eligible_union_patterns=mining.get("eligible_union_patterns"),
        )
        if first.get("mining_invocation_seconds") is not None:
            record["feature_seconds"] = first["mining_invocation_seconds"]
        record["algorithm_seconds"] = first.get(
            "algorithm_seconds",
            mining.get("native_mining_seconds", mining.get("seconds")),
        )
        jobs.append(record)
        if rows:
            record["Dim"] = rows[0].get("Dim")
            record["feature_status"] = (
                "ok"
                if all(r.get("status") == "ok" for r in rows)
                else ",".join(sorted({r.get("status", "unknown") for r in rows}))
            )
        if cfg.get("features_only"):
            record["matrix_sha256"] = first.get("diagnostics", {}).get("matrix_sha256")
            continue
        if entry["study"] in {"quality", "prefix-quality"}:
            valid = (
                result["status"] == "complete"
                and len(rows) == len(cfg["seeds"])
                and {r.get("seed") for r in rows} == set(cfg["seeds"])
            )
            valid = valid and all(
                r.get("status") == "ok"
                and (
                    entry["engine"] == "Raw"
                    or (
                        entry.get("feature_mode") == "threshold"
                        and entry["engine"] != "PCA"
                        and r.get("Dim", 0) > 0
                    )
                    or r.get("Dim") == entry["budget"]
                )
                and all(
                    isinstance(r.get(k), (int, float)) and math.isfinite(r[k])
                    for k in ("NMI", "h", "ARI")
                )
                for r in rows
            )
            if not valid:
                complete.discard(entry["dataset"])
            else:
                mean = {
                    k: statistics.mean(r[k] for r in rows) for k in ("NMI", "h", "ARI")
                }
                summaries.append(
                    dict(
                        record,
                        **mean,
                        Dim=rows[0]["Dim"],
                        feature_seconds=rows[0]["mining_invocation_seconds"],
                        algorithm_seconds=record.get("algorithm_seconds"),
                        native_mining_seconds=rows[0].get("native_mining_seconds"),
                        feature_transform_selection_seconds=rows[0].get(
                            "feature_transform_selection_seconds"
                        ),
                        eligible_union_patterns=rows[0].get("eligible_union_patterns"),
                        strong_rule_occurrences=rows[0].get("strong_rule_occurrences"),
                        cluster_seconds=statistics.mean(
                            r["clustering_seconds"] for r in rows
                        ),
                    )
                )
                for row in rows:
                    metrics.append(
                        dict(
                            dataset=entry["dataset"],
                            engine=entry["engine"],
                            budget=entry["budget"],
                            repeat=entry["repeat"],
                            feature_mode=entry.get("feature_mode", "topk"),
                            minsup=entry.get("minsup"),
                            minconf=entry.get("minconf"),
                            maxsta=entry.get("maxsta"),
                            study=entry["study"],
                            length_percent=entry["length_percent"],
                            seed=row["seed"],
                            **{k: row[k] for k in ("NMI", "h", "ARI", "Dim")},
                        )
                    )
        elif result["status"] == "complete" and rows:
            record.update(
                {
                    k: rows[0].get(k)
                    for k in (
                        "sequence_length",
                        "total_points",
                        "n_samples",
                        "Dim",
                        "mining_seconds",
                        "mining_invocation_seconds",
                        "native_mining_seconds",
                        "feature_transform_selection_seconds",
                        "eligible_union_patterns",
                        "strong_rule_occurrences",
                    )
                }
            )
            record["feature_status"] = rows[0].get("status")
    if cfg.get("features_only"):
        save_csv(root / "jobs.csv", jobs)
        save_csv(root / "feature_repeats.csv", jobs)
        (root / "summary.md").write_text(
            "# Independent feature extraction repeats\n\n"
            + f"Job statuses: {dict(Counter(row['status'] for row in jobs))}\n\n"
            + "Per-condition observations: feature_repeats.csv. No clustering was run.\n"
            + "Use feature_seconds/algorithm_seconds and feature_peak_rss_mib for feature resources. Memory is the peak sampled simultaneous RSS sum of the worker and descendants during feature construction. "
            + "Whole-job time and peak RSS have a different scope from jobs that included clustering. "
            + "Missing phase samples are NA, never zero. Failed/timeout jobs are not successful repetitions.\n"
        )
        print(f"Feature repeats: {root / 'feature_repeats.csv'}")
        return
    from ctminer.tables import aggregate, cell, summary_datasets, write_tables

    write_tables(root, cfg, jobs, summaries)
    save_csv(root / "jobs.csv", jobs)
    save_csv(root / "quality_seeds.csv", metrics)
    save_csv(root / "quality_by_dataset.csv", summaries)
    quality_by_path = {row["path"]: row for row in summaries}
    length_rows = [
        dict(
            row,
            **{
                key: quality_by_path.get(row["path"], {}).get(key)
                for key in ("NMI", "h", "ARI", "cluster_seconds")
            },
        )
        for row in jobs
        if row["study"] in {"length", "prefix-quality"}
    ]
    save_csv(root / "length_resources.csv", length_rows)
    write(
        root / "failures.json",
        [
            r
            for r in jobs
            if r["status"] != "complete" or r.get("feature_status") != "ok"
        ],
    )
    write(
        root / "coverage.json",
        dict(
            aggregate_datasets=summary_datasets(cfg),
            aggregate_policy="per-method, per-condition means over available main datasets; Wafer excluded",
            complete_quality_datasets=sorted(complete)
            if cfg["study"] != "length"
            else [],
            excluded_quality_datasets=sorted(set(cfg["datasets"]) - complete),
            statuses=dict(Counter(r["status"] for r in jobs)),
        ),
    )
    lines = [
        "# Focused feature study",
        "",
        f"Protocol: {cfg.get('mining_protocol', 'legacy')}; feature mode: {cfg.get('feature_mode', 'topk')}; minsups: {cfg.get('minsups', [])}.",
        "CT protocol: qualify patterns by per-series support, rank by total corpus support, and reuse trie counts for features. Top-K uses safe support-bound early stopping; threshold mode returns all qualifying patterns.",
        "Each budget is mined/fitted from scratch in a fresh process. No K50 prefix reuse.",
        "Time/memory are observed concurrent-run measurements; CPU affinity does not isolate memory bandwidth.",
        "Memory is the peak sampled simultaneous RSS sum of the worker and descendants during feature construction, excluding clustering. Whole-job RSS is a separate diagnostic in CSV.",
        "Whole-job time includes all clustering seeds and output. Mining is counted once per job.",
        "",
        f"Job statuses: {dict(Counter(r['status'] for r in jobs))}",
        "Wafer is excluded from aggregates and retained in the per-dataset tables. Each method and condition uses its own valid datasets; an unavailable PCA condition does not exclude another method's results.",
        "Means average seeds, then repetitions, then available datasets equally. Every planned repetition must be valid for a dataset to contribute. Dataset counts and membership are listed in summary.csv. Missing values are dashes; no imputation is used.",
        "",
        "| Method / minsup | Datasets | Dim | NMI | h | ARI | Native mine s | Transform/select s | Feature s | Whole-job s | Sampled peak RSS MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups = defaultdict(list)
    for row in jobs:
        if row["dataset"] in summary_datasets(cfg) and row["study"] in {
            "quality",
            "prefix-quality",
        }:
            groups[
                (
                    row["engine"],
                    row["budget"],
                    row.get("minsup"),
                    row.get("feature_mode", "topk"),
                    row.get("minconf"),
                    row.get("maxsta"),
                    row["study"],
                    row["length_percent"],
                )
            ].append(
                dict(
                    quality_by_path.get(row["path"], row),
                    quality_valid=row["path"] in quality_by_path,
                )
            )
    macro = []
    for (engine, budget, minsup, mode, minconf, maxsta, study, percent), rows in sorted(
        groups.items(), key=lambda item: str(item[0])
    ):
        result = dict(
            engine=engine,
            budget=budget,
            minsup=minsup,
            feature_mode=mode,
            minconf=minconf,
            maxsta=maxsta,
            study=study,
            length_percent=percent,
            **aggregate(rows, cfg),
        )
        macro.append(result)
        cells = [
            cell(result[key])
            for key in (
                "Dim",
                "NMI",
                "h",
                "ARI",
                "native_mining_seconds",
                "feature_transform_selection_seconds",
                "feature_seconds",
                "wall_seconds",
                "feature_peak_rss_mib",
            )
        ]
        label = (
            engine
            if engine == "Raw"
            else f"{engine}-all"
            if mode == "threshold" and engine != "PCA"
            else f"{engine}-K{budget}"
        )
        if minsup is not None:
            label += f" / S{minsup:g}"
        if minconf is not None:
            label += f" / minconf={minconf:g}"
        if maxsta is not None:
            label += f" / maxsta={maxsta:g}"
        if study == "prefix-quality":
            label += f" / input={percent}%"
        lines.append(f"| {label} | {result['datasets']} | {' | '.join(cells)} |")
    save_csv(root / "summary.csv", macro)
    lines += [
        "",
        "All attempted per-dataset conditions: all_conditions.csv / dataset_tables.md. Main-dataset comparison slices: matched_slices.csv / matched_slices.md. Summary means and dataset membership: summary.csv.",
        "Successful per-dataset quality/time/dimension: quality_by_dataset.csv. Length curves: length_resources.csv.",
        "All attempted jobs, including timeouts with elapsed time and peak RSS: jobs.csv. Timeouts are not completed-runtime measurements.",
        "Each job retains resources.json (phase peak RSS, CPU allocation), mining.json, dictionary.json and logs.",
        f"Reused jobs: {sum('reused_from' in row for row in jobs)}. Reused jobs retain original source fingerprints, timings and execution settings; see reused_from.json. Reuse/copy time is not mining time.",
    ]
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Summary: {root / 'summary.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "run"):
        p = sub.add_parser(command)
        p.add_argument("--prepared", type=Path, default=Path("data/prepared"))
        p.add_argument("--output", type=Path, default=Path("results/clustering"))
        p.add_argument(
            "--mining-protocol", choices=["native"], default="native"
        )
        p.add_argument("--feature-mode", choices=["topk", "threshold"], default="topk")
        p.add_argument(
            "--minsups",
            nargs="+",
            type=float,
            default=[2.0],
            help="Absolute per-series occurrence support thresholds",
        )
        p.add_argument("--max-feature-cells", type=int, default=100_000_000)
        p.add_argument(
            "--pca-components",
            type=int,
            default=30,
            help="PCA reference dimension in threshold mode; has no mining threshold",
        )
        p.add_argument("--datasets", nargs="+", default=DATASETS)
        p.add_argument(
            "--reuse-from",
            nargs="+",
            type=Path,
            default=[],
            help="Copy compatible complete/timeout jobs from finished focused sweeps; preserve provenance",
        )
        p.add_argument(
            "--require-reuse-datasets",
            nargs="+",
            default=[],
            help="Refuse to run any missing job for these already-completed datasets",
        )
        p.add_argument(
            "--methods",
            nargs="+",
            choices=METHODS,
            default=["CT"],
        )
        p.add_argument(
            "--study",
            choices=["quality", "length", "prefix-quality", "both"],
            default="both",
            help="prefix-quality grows input prefixes and also runs clustering; length retains legacy mining-only behavior",
        )
        p.add_argument("--budgets", nargs="+", type=int, default=[10, 20, 30, 40, 50])
        p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
        p.add_argument("--n-init", type=int, default=10)
        p.add_argument(
            "--lengths",
            nargs="+",
            type=int,
            default=None,
            help="Default: all lengths through the input length. Explicit lengths restrict reported features.",
        )
        p.add_argument(
            "--length-percents", nargs="+", type=int, default=list(range(10, 101, 10))
        )
        p.add_argument(
            "--repeats",
            type=int,
            default=1,
            help="Independent whole-job repetitions, never shared JVM repeats",
        )
        p.add_argument(
            "--repeat-start",
            type=int,
            default=0,
            help="First repeat label; does not change the input or mining",
        )
        p.add_argument(
            "--features-only",
            action="store_true",
            help="Extract/save features and resources, skip K-means and feature scaling for clustering",
        )
        p.add_argument("--jobs", type=int, default=6)
        p.add_argument("--cores-per-job", type=int, default=4)
        p.add_argument("--threads", type=int, default=1)
        p.add_argument("--affinity", choices=["physical", "none"], default="physical")
        p.add_argument("--sample-interval", type=float, default=0.05)
        p.add_argument("--java-heap", default="16g")
        p.add_argument("--java-timeout", type=float, default=7200)
        p.add_argument("--job-timeout", type=float, default=10800)
        p.add_argument("--pattern-cap", type=int, default=2000000)
        p.add_argument("--retry-failed", action="store_true")
    p = sub.add_parser("summarize")
    p.add_argument("output", type=Path)
    p = sub.add_parser("_worker")
    p.add_argument("job", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        plan(args)
    elif args.command == "run":

        def terminate(signum, frame):
            raise KeyboardInterrupt("Supervisor terminated")

        signal.signal(signal.SIGTERM, terminate)
        run(args)
    elif args.command == "summarize":
        summarize(args.output)
    else:
        worker(args.job)


if __name__ == "__main__":
    main()

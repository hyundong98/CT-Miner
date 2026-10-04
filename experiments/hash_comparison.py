"""CT versus CT-Hash under matched n and L settings."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
from time import perf_counter, sleep
import traceback

from ctminer.paths import code_hashes
from ctminer.memory import VERSION as MEMORY_VERSION, sampled_peak_bytes

HERE = Path(__file__).resolve().parent
METHODS = ("CT", "CT-Hash")
DATASETS = ["Car", "Beef", "ElectricDevices", "MoteStrain", "PigCVP", "Wafer"]
MAX_LENGTH = 1000000  # Matches the Java collection input format.


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def matching_outputs(left, right):
    """Compare already completed jobs; never rerun either algorithm."""
    import numpy as np

    a, b = read(left / "result.json"), read(right / "result.json")
    fields = (
        "input_sha256",
        "output_sha256",
        "stage_signature",
        "Dim",
        "final_length",
        "valid_windows_at_L",
    )
    if any(a[k] != b[k] for k in fields):
        return False
    if read(left / "dictionary.json") != read(right / "dictionary.json"):
        return False
    if not np.array_equal(
        np.load(left / "features.npy", allow_pickle=False, mmap_mode="r"),
        np.load(right / "features.npy", allow_pickle=False, mmap_mode="r"),
    ):
        return False
    fields = (
        "length",
        "observed_patterns",
        "eligible_patterns",
        "maximum_unfiltered_support",
        "cutoff",
    )
    stages = [
        [{k: s[k] for k in fields} for s in read(folder / "mining.json")["steps"]]
        for folder in (left, right)
    ]
    return stages[0] == stages[1]


def specifications(args):
    """Read manifest metadata only. Never load data, build Java or execute a job."""
    for name in ("l_values", "n_values", "n_percents", "datasets", "families"):
        values = getattr(args, name)
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Nonempty unique {name} required")
    if (
        args.repeats < 1
        or args.repeat_start < 0
        or args.seed < 0
        or args.samples < 1
        or args.sample_limit < 0
    ):
        raise ValueError("Invalid repetitions, seed or sample count")
    if not 1 <= args.budget <= 2147483647 or not 1 <= args.pattern_cap <= 2147483647:
        raise ValueError("Positive int budget and pattern cap required")
    if args.jobs != 1 or args.threads != 1:
        raise ValueError("This controlled scaling runner requires --jobs 1 --threads 1")
    if args.cores_per_job < 1 or not re.fullmatch(
        r"[1-9][0-9]*[kKmMgG]", args.java_heap
    ):
        raise ValueError("Invalid cores or Java heap")
    if not all(
        math.isfinite(t) and t > 0
        for t in (args.java_timeout, args.job_timeout, args.sample_interval)
    ):
        raise ValueError("Finite positive timeouts and sampling interval required")
    if args.java_timeout >= args.job_timeout or args.sample_interval < 0.02:
        raise ValueError(
            "Require java-timeout < job-timeout and sample-interval >= .02"
        )
    if (
        not 2
        <= min(args.l_values + [args.l_fixed])
        <= max(args.l_values + [args.l_fixed])
        <= MAX_LENGTH
    ):
        raise ValueError(f"Pattern lengths must be within 2..{MAX_LENGTH}")
    if args.n_fixed is not None and not 2 <= args.n_fixed < 999999999:
        raise ValueError("n-fixed outside the CT input range")
    if (
        min(args.n_values) < 2
        or max(args.n_values) >= 999999999
        or min(args.n_percents) < 1
        or max(args.n_percents) > 100
    ):
        raise ValueError("Invalid n-values or n-percents")
    sources = []
    if args.source == "prepared":
        manifest = read(args.prepared / "manifest.json")
        for name in args.datasets:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError("Invalid dataset name")
            data = manifest["datasets"].get(name, {})
            if data.get("status") != "ready" or len(data.get("lengths", [])) != 1:
                raise ValueError(f"Ready equal-length dataset required: {name}")
            sources.append(
                dict(
                    name=name,
                    kind="prepared",
                    n=data["lengths"][0],
                    samples=min(args.sample_limit, data["n_samples"])
                    if args.sample_limit
                    else data["n_samples"],
                    input=str((args.prepared / data["npz"]).resolve()),
                    sha256=data["sha256"],
                )
            )
    else:
        size = max(args.n_fixed or 4096, max(args.n_values))
        sources = [
            dict(
                name=f"synthetic-{family}",
                kind="synthetic",
                family=family,
                n=size,
                samples=args.samples,
                seed=args.seed,
            )
            for family in args.families
        ]
    conditions, skipped = [], []
    for source in sources:
        proposals = []
        if args.study in {"vary-l", "both"}:
            n = (
                args.n_fixed
                if args.n_fixed is not None
                else (4096 if args.source == "synthetic" else source["n"])
            )
            proposals += [("vary-l", n, L, None) for L in sorted(args.l_values)]
        if args.study in {"vary-n", "both"}:
            if args.source == "synthetic":
                proposals += [
                    ("vary-n", n, args.l_fixed, None) for n in sorted(args.n_values)
                ]
            else:
                # Do not clamp short prefixes to L: that would change the design.
                proposals += [
                    ("vary-n", source["n"] * p // 100, args.l_fixed, p)
                    for p in sorted(args.n_percents)
                ]
        seen = set()
        for study, n, L, percent in proposals:
            c = dict(
                source=source["name"],
                study=study,
                n=n,
                L=L,
                samples=source["samples"],
                percent=percent,
            )
            reason = (
                "n exceeds the source length"
                if n > source["n"]
                else "n < L"
                if n < L
                else None
            )
            if (study, n, L) in seen:
                reason = "duplicate prefix size after rounding"
            seen.add((study, n, L))
            if reason:
                skipped.append(dict(c, reason=reason))
            else:
                conditions.append(c)
    if not conditions:
        raise ValueError("No feasible conditions; revise n/L settings")
    return sources, conditions, skipped


def plan(args):
    sources, conditions, skipped = specifications(args)
    print(
        json.dumps(
            dict(
                sources=sources,
                conditions=conditions,
                skipped=skipped,
                total_jobs=len(conditions) * 2 * args.repeats,
                planned_pairs=len(conditions) * args.repeats,
                methods=METHODS,
                budget=args.budget,
                repeats=args.repeats,
                policy="all lengths 2..L; no early stop; cold JVM per job; no clustering",
            ),
            indent=2,
        )
    )


def array_hash(x):
    import numpy as np

    h = hashlib.sha256(json.dumps(list(x.shape)).encode())
    h.update(np.asarray(x, dtype=">f8", order="C").tobytes())
    return h.hexdigest()


def prepare_source(source, args, target):
    """Input preparation is outside every measured worker; labels are unused."""
    import numpy as np

    name_seed = int(hashlib.sha256(source["name"].encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, name_seed]))
    if source["kind"] == "prepared":
        if digest(source["input"]) != source["sha256"]:
            raise ValueError(f"Prepared input hash changed: {source['name']}")
        with np.load(source["input"], allow_pickle=False) as data:
            x = np.array(data["X"], dtype=np.float64, copy=True)
        if x.ndim != 2 or x.shape[1] != source["n"] or x.shape[0] < source["samples"]:
            raise ValueError("Prepared manifest shape mismatch")
        indices = (
            np.sort(rng.choice(len(x), source["samples"], replace=False))
            if source["samples"] < len(x)
            else np.arange(len(x))
        )
        x = x[indices]
    else:
        indices = np.arange(source["samples"])
        x = np.empty((source["samples"], source["n"]), dtype=np.float64)
        for s in range(len(x)):
            n = source["n"]
            if source["family"] == "random":
                x[s] = rng.permutation(n)  # Exact integers, no ties after .9g.
            elif source["family"] == "monotone":
                x[s] = np.arange(n)
            else:
                # Repeated ordinal shape with distinct values across periods.
                period = min(32, n)
                shape = rng.permutation(period)
                i = np.arange(n)
                blocks = (n + period - 1) // period
                x[s] = shape[i % period] * blocks + i // period
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite source input")
    x = np.asarray([[float(format(float(v), ".9g")) for v in row] for row in x])
    np.save(target, x, allow_pickle=False)
    return dict(
        file=target.name,
        sha256=digest(target),
        shape=list(x.shape),
        array_sha256=array_hash(x),
        sample_indices=indices.tolist(),
        quantization=".9g float64; strict distinct-valued windows; no jitter/resampling",
    )


def valid_windows(x, L):
    """Input diagnostics only, outside the timed Java invocation."""
    import numpy as np

    histogram = np.zeros(L + 1, dtype=np.int64)
    for row in x:
        end = len(row)
        next_position = {}
        limits = np.empty(len(row), dtype=np.int64)
        for i in range(len(row) - 1, -1, -1):
            end = min(end, next_position.get(float(row[i]), len(row)))
            limits[i] = min(L, end - i)
            next_position[float(row[i])] = i
        histogram += np.bincount(limits, minlength=L + 1)
    counts = np.cumsum(histogram[::-1])[::-1]
    return [int(v) for v in counts[2:]]


def worker(path):
    import resource

    job = read(path)
    out = Path(path).parent
    cfg = job["config"]
    try:
        if job["cpus"] is not None:
            os.sched_setaffinity(0, set(job["cpus"]))
        import numpy as np
        from ctminer.collections import mine_hash as mine
        from ctminer.memory import FeatureMemory

        write(out / "phase.json", dict(phase="input"))
        if digest(job["input"]) != job["input_file_sha256"]:
            raise ValueError("Benchmark source changed")
        x = np.load(job["input"], allow_pickle=False, mmap_mode="r")[:, : job["n"]]
        if x.shape != (job["samples"], job["n"]):
            raise ValueError("Benchmark prefix shape mismatch")
        input_hash = array_hash(x)
        write(out / "phase.json", dict(phase="mining"))
        with FeatureMemory(
            out / "feature_memory.json", interval=cfg["sample_interval"]
        ):
            patterns, values, mining = mine(
                x,
                job["engine"],
                cfg["budget"],
                out / "mining",
                lengths=range(2, job["L"] + 1),
                timeout=cfg["java_timeout"],
                cap=cfg["pattern_cap"],
                heap=cfg["java_heap"],
                threads=1,
                warmups=0,
                repeats=1,
                all_patterns=False,
                stop_early=False,
                allow_long_ct=True,
            )
        write(out / "phase.json", dict(phase="output_and_diagnostics"))
        if [s["length"] for s in mining["steps"]] != list(
            range(2, job["L"] + 1)
        ) or mining["stop_reason"] != "all_requested_lengths":
            raise ValueError(
                "Incomplete length exploration; no scaling result accepted"
            )
        dictionary = [
            dict(codes=list(p), support=float(s))
            for p, s in zip(patterns, mining["scores"])
        ]
        write(out / "dictionary.json", dictionary)
        np.save(out / "features.npy", values, allow_pickle=False)
        write(out / "mining.json", mining)
        diagnostics = valid_windows(x, job["L"])
        write(
            out / "input_diagnostics.json",
            dict(
                valid_windows_by_length=diagnostics,
                lengths=list(range(2, job["L"] + 1)),
                total_points=int(x.size),
                raw_windows_by_length=[
                    len(x) * (job["n"] - ell + 1) for ell in range(2, job["L"] + 1)
                ],
            ),
        )
        stage_fields = (
            "length",
            "observed_patterns",
            "eligible_patterns",
            "maximum_unfiltered_support",
            "cutoff",
        )
        stages = [{k: s[k] for k in stage_fields} for s in mining["steps"]]
        index = sum(s["index_seconds"] for s in mining["steps"])
        write(
            out / "result.json",
            dict(
                status="complete",
                fingerprint=job["fingerprint"],
                input_sha256=input_hash,
                output_sha256=fingerprint(
                    dict(dictionary=dictionary, matrix=array_hash(values))
                ),
                stage_signature=fingerprint(stages),
                dictionary_sha256=digest(out / "dictionary.json"),
                features_sha256=digest(out / "features.npy"),
                Dim=len(patterns),
                budget_filled=len(patterns) == cfg["budget"],
                mining_seconds=mining["seconds"],
                feature_seconds=mining["invocation_seconds"],
                index_seconds=index,
                post_index_seconds=mining["seconds"] - index,
                count_seconds=sum(s["count_seconds"] for s in mining["steps"]),
                selection_seconds=sum(s["selection_seconds"] for s in mining["steps"]),
                observed_patterns=sum(s["observed_patterns"] for s in mining["steps"]),
                final_length=job["L"],
                valid_windows_at_L=diagnostics[-1],
                canonical_id_runs=mining.get("canonical_id_runs"),
            ),
        )
    except Exception as exc:
        write(
            out / "result.json",
            dict(
                status="timeout" if isinstance(exc, TimeoutError) else "failed",
                fingerprint=job["fingerprint"],
                reason=str(exc),
                traceback=traceback.format_exc(),
            ),
        )
        return 1
    finally:
        own = resource.getrusage(resource.RUSAGE_SELF)
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        write(
            out / "worker_usage.json",
            dict(
                worker_cpu_seconds=own.ru_utime
                + own.ru_stime
                + child.ru_utime
                + child.ru_stime
            ),
        )
    return 0


def execute(job, out):
    from ctminer.resources import TreeMeter

    cfg = job["config"]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        history = out / "attempts"
        history.mkdir(exist_ok=True)
        attempt = history / str(len(list(history.iterdir())) + 1)
        attempt.mkdir()
        for old in list(out.iterdir()):
            if old.name != "attempts":
                shutil.move(str(old), attempt / old.name)
    write(out / "job.json", job)
    write(out / "result.json", dict(status="running", fingerprint=job["fingerprint"]))
    env = dict(os.environ)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env[name] = "1"
    tick = perf_counter()
    timed_out = interrupted = False
    with (out / "worker.log").open("w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "experiments.hash_comparison",
                "_worker",
                str(out / "job.json"),
            ],
            cwd=HERE.parent,
            env=env,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        meter = TreeMeter(proc.pid, cfg["sample_interval"])
        try:
            while proc.poll() is None:
                phase_path = out / "phase.json"
                meter.sample(
                    read(phase_path)["phase"] if phase_path.exists() else "startup"
                )
                if perf_counter() - tick >= cfg["job_timeout"]:
                    timed_out = True
                    break
                sleep(cfg["sample_interval"])
        except BaseException:
            interrupted = True
            raise
        finally:
            # Java uses its own process group; kill tracked descendants as well.
            meter.signal(signal.SIGKILL)
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            resources = dict(
                meter.result(),
                elapsed_wall_seconds=perf_counter() - tick,
                returncode=proc.returncode,
                cpus=job["cpus"],
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
                    else "whole-job timeout",
                )
            elif proc.returncode != 0 and result["status"] not in {"failed", "timeout"}:
                result.update(status="failed", reason=f"worker exit {proc.returncode}")
            elif result["status"] == "running":
                result.update(status="failed", reason="worker produced no result")
            result["resources"] = resources
            write(out / "result.json", result)
    return result["status"]


def run(args):
    sources, conditions, skipped = specifications(args)
    if sys.platform != "linux":
        raise RuntimeError(
            "Measured runs require Linux /proc; plan and summarize are portable"
        )
    from ctminer.resources import cpu_slots
    from ctminer.engines import _java_ready, CLASSES
    from scripts.build import verify_java
    import fcntl

    _java_ready()  # Read-only stale-build check, never auto-compiles.
    verify_java()
    cpus = cpu_slots(1, args.cores_per_job, args.affinity)[0]
    cfg = {
        k: v
        for k, v in vars(args).items()
        if k not in {"command", "output", "retry_failed", "prepared"}
    }
    cfg.update(
        memory_protocol=MEMORY_VERSION,
        sources=sources,
        conditions=conditions,
        skipped=skipped,
        cpus=cpus,
        hostname="anonymous",
        numpy_version=importlib.metadata.version("numpy"),
        python_version=sys.version,
        build_sha256=digest(CLASSES / "manifest.json"),
        code=code_hashes(),
        stop_early=False,
        min_pattern_length=2,
        timing="fresh process and cold JVM; no warmup; mining only",
    )
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "config.json").exists() and read(root / "config.json") != cfg:
            raise ValueError("Settings/code/build changed: use a new output folder")
        write(root / "config.json", cfg)
        inputs = root / "inputs"
        inputs.mkdir(exist_ok=True)
        input_manifest = (
            read(inputs / "manifest.json")
            if (inputs / "manifest.json").exists()
            else {}
        )
        for source in sources:
            name = source["name"]
            target = inputs / f"{name}.npy"
            if name in input_manifest:
                if (
                    not target.exists()
                    or digest(target) != input_manifest[name]["sha256"]
                ):
                    raise ValueError(f"Cached benchmark input changed: {name}")
            else:
                input_manifest[name] = prepare_source(source, args, target)
                write(inputs / "manifest.json", input_manifest)
        index = []
        pending = []
        for cindex, condition in enumerate(conditions):
            for repeat in range(args.repeat_start, args.repeat_start + args.repeats):
                order = METHODS if (repeat + cindex) % 2 == 0 else METHODS[::-1]
                for engine in order:
                    source = input_manifest[condition["source"]]
                    job = dict(
                        condition,
                        engine=engine,
                        repeat=repeat,
                        config=cfg,
                        cpus=cpus,
                        input=str(inputs / source["file"]),
                        input_file_sha256=source["sha256"],
                    )
                    job["fingerprint"] = fingerprint(job)
                    relative = (
                        Path("jobs")
                        / condition["study"]
                        / condition["source"]
                        / f"n{condition['n']}_L{condition['L']}"
                        / f"R{repeat}"
                        / engine
                    )
                    entry = dict(
                        condition,
                        engine=engine,
                        repeat=repeat,
                        path=str(relative),
                        fingerprint=job["fingerprint"],
                    )
                    index.append(entry)
                    result_path = root / relative / "result.json"
                    if result_path.exists():
                        result = read(result_path)
                        if result.get("fingerprint") != job["fingerprint"]:
                            raise ValueError(f"Job fingerprint mismatch: {relative}")
                        if result["status"] == "complete" or (
                            result["status"] in {"failed", "timeout", "interrupted"}
                            and not args.retry_failed
                        ):
                            continue
                    pending.append((job, root / relative))
        write(root / "job_index.json", index)
        print(
            f"Planned {len(index)} jobs; retained {len(index) - len(pending)}; pending {len(pending)}; skipped {len(skipped)} conditions",
            flush=True,
        )
        for i, (job, out) in enumerate(pending, 1):
            status = execute(job, out)
            print(
                f"[{i}/{len(pending)}] {job['source']} / {job['study']} / n={job['n']} / L={job['L']} / R{job['repeat']} / {job['engine']} / {status}",
                flush=True,
            )
            left, right = out.parent / "CT", out.parent / "CT-Hash"
            if all(
                (p / "result.json").exists()
                and read(p / "result.json")["status"] == "complete"
                for p in (left, right)
            ):
                matched = matching_outputs(left, right)
                write(
                    out.parent / "parity.json",
                    dict(status="matched" if matched else "mismatch"),
                )
                if not matched:
                    summarize(root)
                    raise RuntimeError(
                        f"Output mismatch: {out.parent}; remaining jobs were not started"
                    )
    summarize(root)


def csv_write(path, rows):
    with Path(path).open("w", newline="") as stream:
        if rows:
            writer = csv.DictWriter(
                stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r))
            )
            writer.writeheader()
            writer.writerows(rows)


def summarize(root):
    root = Path(root)
    cfg = read(root / "config.json")
    grouped = defaultdict(dict)
    jobs = []
    pairs = []
    metrics = (
        "mining_seconds",
        "feature_seconds",
        "index_seconds",
        "post_index_seconds",
        "count_seconds",
        "selection_seconds",
        "peak_rss_mib",
        "mining_peak_rss_mib",
        "java_peak_rss_mib",
        "feature_peak_rss_mib",
    )
    for entry in read(root / "job_index.json"):
        folder = root / entry["path"]
        path = folder / "result.json"
        r = read(path) if path.exists() else dict(status="pending")
        if path.exists() and r.get("fingerprint") != entry["fingerprint"]:
            raise ValueError(f"Result fingerprint mismatch: {path}")
        row = dict(
            entry,
            status=r["status"],
            reason=r.get("reason"),
            Dim=r.get("Dim"),
            budget_filled=r.get("budget_filled"),
        )
        res = r.get("resources", {})
        for key in metrics:
            row[key] = r.get(key)
        row["whole_job_seconds"] = res.get("elapsed_wall_seconds")
        row["worker_cpu_seconds"] = res.get("worker_cpu_seconds")
        for label, value in (
            ("peak_rss_mib", res.get("sampled_peak_rss_bytes")),
            (
                "mining_peak_rss_mib",
                res.get("phases", {}).get("mining", {}).get("sampled_peak_rss_bytes"),
            ),
            ("java_peak_rss_mib", res.get("sampled_java_peak_rss_bytes")),
        ):
            row[label] = value / 2**20 if value is not None else None
        memory = res.get("feature_memory", {})
        peak = sampled_peak_bytes(memory)
        row["feature_peak_rss_mib"] = peak / 2**20 if peak is not None else None
        row["memory_protocol"] = memory.get("version")
        row["memory_status"] = memory.get("status", "missing")
        if r["status"] == "complete":
            for filename, field in (
                ("dictionary.json", "dictionary_sha256"),
                ("features.npy", "features_sha256"),
            ):
                if (
                    not (folder / filename).exists()
                    or digest(folder / filename) != r[field]
                ):
                    raise ValueError(f"Saved output changed: {folder / filename}")
        key = (entry["source"], entry["study"], entry["n"], entry["L"], entry["repeat"])
        if entry["engine"] in grouped[key]:
            raise ValueError("Duplicate planned job")
        grouped[key][entry["engine"]] = (r, row)
        jobs.append(row)
    for key, group in sorted(grouped.items()):
        pair = dict(zip(("source", "study", "n", "L", "repeat"), key))
        if set(group) != set(METHODS) or any(
            r[0]["status"] != "complete" for r in group.values()
        ):
            pair.update(
                status="unverified",
                reason="missing, failed, timed out or pending member",
            )
        else:
            a, ar = group["CT"]
            b, br = group["CT-Hash"]
            same = matching_outputs(root / ar["path"], root / br["path"])
            pair.update(
                status="matched" if same else "mismatch",
                Dim=a["Dim"],
                budget_filled=a["budget_filled"],
                valid_windows_at_L=a["valid_windows_at_L"],
                input_sha256=a["input_sha256"],
                output_sha256=a["output_sha256"],
            )
            if same:
                for metric in metrics:
                    av, bv = ar[metric], br[metric]
                    pair["ct_" + metric] = av
                    pair["hash_" + metric] = bv
                    pair["hash_over_ct_" + metric] = (
                        bv / av
                        if av is not None and bv is not None and av > 0
                        else None
                    )
        pairs.append(pair)
    csv_write(root / "jobs.csv", jobs)
    csv_write(root / "pairs.csv", pairs)
    summary = []
    buckets = defaultdict(list)
    for p in pairs:
        buckets[(p["source"], p["study"], p["n"], p["L"])].append(p)
    for key, ps in sorted(buckets.items()):
        matched = [p for p in ps if p["status"] == "matched"]
        stable = len({(p["input_sha256"], p["output_sha256"]) for p in matched}) <= 1
        complete = len(matched) == cfg["repeats"] and stable
        row = dict(
            zip(("source", "study", "n", "L"), key),
            matched_repeats=len(matched),
            planned_repeats=cfg["repeats"],
            status="complete"
            if complete
            else "unstable_outputs"
            if not stable
            else "incomplete",
        )
        row.update(
            Dim=matched[0]["Dim"] if matched else None,
            budget_filled=matched[0]["budget_filled"] if matched else None,
            valid_windows_at_L=matched[0]["valid_windows_at_L"] if matched else None,
        )
        # Incomplete/timeout conditions stay visible and have no headline speedup.
        for metric in metrics:
            for prefix in ("ct_", "hash_"):
                field = prefix + metric
                values = [p[field] for p in matched if p.get(field) is not None]
                row[field] = (
                    statistics.mean(values)
                    if complete and len(values) == len(matched)
                    else None
                )
            ct_mean, hash_mean = row["ct_" + metric], row["hash_" + metric]
            row["hash_over_ct_" + metric] = (
                hash_mean / ct_mean
                if ct_mean is not None and hash_mean is not None and ct_mean > 0
                else None
            )
        summary.append(row)
    csv_write(root / "scaling.csv", summary)
    write(
        root / "audit.json",
        dict(
            job_statuses=dict(Counter(r["status"] for r in jobs)),
            pair_statuses=dict(Counter(p["status"] for p in pairs)),
            skipped_conditions=cfg["skipped"],
            incomplete_conditions=[r for r in summary if r["status"] != "complete"],
        ),
    )
    lines = [
        "# CT vs CT-Hash scaling",
        "",
        "L is maximum pattern length. Every length 2..L is mined with early stopping disabled. Fixed K, no clustering; cold JVM and fresh index per job. Inputs and sample count are shared. Execution order alternates. All jobs are serial.",
        "",
        "Arithmetic means require every planned repetition to complete with exact ordered dictionary/support/features and stage signatures matching. Speedups are ratios of mean Hash time to mean CT time. Feature and mining speedups are reported separately. Missing/timeout values are not zero. K is an upper limit; Dim may be smaller on simple inputs.",
        "",
        "Feature time includes JVM startup and input/output. Index time includes trie construction and compact-state preprocessing. Post-index is Java mining minus index time. Reported memory is the peak sampled simultaneous RSS sum of the worker and descendants during feature construction. Java-only and whole-job RSS remain separate CSV diagnostics. Fixed overhead, JIT/GC, ties and changing candidate counts can affect slopes; timing alone does not prove a worst-case bound.",
        "",
        f"Job statuses: {dict(Counter(r['status'] for r in jobs))}. Pair statuses: {dict(Counter(p['status'] for p in pairs))}.",
        "",
        "| Source | Study | n | L | Matched repeats | Status | CT feature s | Hash feature s | Feature speedup | CT mining s | Hash mining s | Mining speedup | CT sampled peak RSS MiB | Hash sampled peak RSS MiB |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        cells = [
            r["source"],
            r["study"],
            str(r["n"]),
            str(r["L"]),
            f"{r['matched_repeats']}/{r['planned_repeats']}",
            r["status"],
        ]
        cells += [
            "NA" if r[k] is None else f"{r[k]:.4f}"
            for k in (
                "ct_feature_seconds",
                "hash_feature_seconds",
                "hash_over_ct_feature_seconds",
                "ct_mining_seconds",
                "hash_mining_seconds",
                "hash_over_ct_mining_seconds",
                "ct_feature_peak_rss_mib",
                "hash_feature_peak_rss_mib",
            )
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Skipped conditions (no clamping, padding or extrapolation):", ""]
    lines += [
        f"- {s['source']} / {s['study']} / n={s['n']} / L={s['L']}: {s['reason']}"
        for s in cfg["skipped"]
    ] or ["None."]
    lines += [
        "",
        "Files: scaling.csv (condition means and ratios of means); pairs.csv (per-repeat parity/ratios); jobs.csv (all jobs including failures/timeouts); audit.json; per-job mining.json, dictionary.json, features.npy, input_diagnostics.json, resources.json and logs.",
        "",
    ]
    (root / "summary.md").write_text("\n".join(lines))
    print(f"Summary: {root / 'summary.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "run"):
        p = sub.add_parser(command)
        p.add_argument(
            "--source", choices=["prepared", "synthetic"], default="prepared"
        )
        p.add_argument("--prepared", type=Path, default=Path("data/prepared"))
        p.add_argument("--datasets", nargs="+", default=DATASETS)
        p.add_argument(
            "--families",
            nargs="+",
            choices=["random", "monotone", "periodic"],
            default=["random", "monotone", "periodic"],
        )
        p.add_argument(
            "--samples",
            type=int,
            default=1,
            help="Synthetic series count, fixed across conditions",
        )
        p.add_argument(
            "--sample-limit",
            type=int,
            default=0,
            help="Prepared data only: 0 uses all samples; otherwise a seeded fixed subset",
        )
        p.add_argument(
            "--seed",
            type=int,
            default=17,
            help="Input seed, shared across measurement repeats",
        )
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--study", choices=["vary-l", "vary-n", "both"], default="both")
        p.add_argument(
            "--n-fixed",
            type=int,
            help="vary-l prefix length; default full real series / 4096 synthetic",
        )
        p.add_argument(
            "--l-values", nargs="+", type=int, default=[8, 16, 32, 64, 128, 256]
        )
        p.add_argument("--l-fixed", type=int, default=16)
        p.add_argument(
            "--n-values",
            nargs="+",
            type=int,
            default=[256, 512, 1024, 2048, 4096],
            help="Synthetic vary-n sizes",
        )
        p.add_argument(
            "--n-percents",
            nargs="+",
            type=int,
            default=[20, 40, 60, 80, 100],
            help="Prepared vary-n prefixes",
        )
        p.add_argument("--budget", type=int, default=30)
        p.add_argument("--repeats", type=int, default=5)
        p.add_argument(
            "--repeat-start",
            type=int,
            default=0,
            help="First measurement repeat label; input seed remains unchanged",
        )
        p.add_argument("--jobs", type=int, choices=[1], default=1)
        p.add_argument("--threads", type=int, choices=[1], default=1)
        p.add_argument("--cores-per-job", type=int, default=1)
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
    elif args.command == "summarize":
        summarize(args.output)
    elif args.command == "_worker":
        raise SystemExit(worker(args.job))
    else:
        if sys.platform != "linux":
            parser.error(
                "Resource measurements require Linux. Use python run.py validate for a local functional check."
            )

        def stop(signum, frame):
            raise KeyboardInterrupt("Scaling supervisor terminated")

        signal.signal(signal.SIGTERM, stop)
        run(args)


if __name__ == "__main__":
    main()

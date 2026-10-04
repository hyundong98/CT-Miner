"""One independent feature extraction per condition for sampled peak RSS."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from ctminer.memory import VERSION as MEMORY_VERSION, sampled_peak_bytes

PROTOCOL = "one_run_sampled_feature_rss_v3"

HERE = Path(__file__).resolve().parent
DATASETS = ["Car", "Beef", "ElectricDevices", "MoteStrain", "PigCVP", "Wafer"]
METHODS = ("Raw", "PCA", "CT")
NATIVE = {
    "topk": (
        "01_topk",
        [
            "--study",
            "quality",
            "--feature-mode",
            "topk",
            "--budgets",
            "10",
            "20",
            "30",
            "40",
            "50",
            "--minsups",
            "2",
        ],
    ),
    "threshold": (
        "02_threshold",
        [
            "--study",
            "quality",
            "--feature-mode",
            "threshold",
            "--minsups",
            "5",
            "10",
            "25",
            "--pca-components",
            "30",
        ],
    ),
    "prefix": (
        "03_input_length",
        [
            "--study",
            "prefix-quality",
            "--feature-mode",
            "topk",
            "--budgets",
            "30",
            "--minsups",
            "2",
            "--length-percents",
            "20",
            "40",
            "60",
            "80",
            "100",
        ],
    ),
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def commands(args):
    action = "plan" if args.command == "plan" else "run"
    common = [
        "--prepared",
        str(args.prepared.resolve()),
        "--datasets",
        *DATASETS,
        "--cores-per-job",
        "1",
        "--threads",
        "1",
        "--sample-interval",
        str(args.sample_interval),
        "--java-heap",
        args.java_heap,
        "--java-timeout",
        str(args.java_timeout),
        "--job-timeout",
        str(args.job_timeout),
        "--repeats",
        "1",
        "--repeat-start",
        "0",
        "--pattern-cap",
        "2000000",
    ]
    for name in args.only:
        folder, preset = NATIVE[name]
        command = [
            sys.executable,
            "-m",
            "ctminer.study",
            action,
            *common,
            "--output",
            str((args.output / folder).resolve()),
            "--methods",
            *args.methods,
            "--mining-protocol",
            "native",
            "--jobs",
            str(args.jobs),
            "--features-only",
            "--max-feature-cells",
            "100000000",
            *preset,
        ]
        if args.retry_failed:
            command.append("--retry-failed")
        yield folder, command

    if args.include_hash:
        for study in ["vary-l", "vary-n"]:
            folder = "04_ct_hash/" + study
            command = [
                sys.executable,
                "-m",
                "experiments.hash_comparison",
                action,
                *common,
                "--output",
                str((args.output / folder).resolve()),
                "--source",
                "prepared",
                "--study",
                study,
                "--sample-limit",
                "0",
                "--seed",
                "17",
                "--budget",
                "30",
                "--l-values",
                "8",
                "16",
                "32",
                "64",
                "128",
                "256",
                "--l-fixed",
                "16",
                "--n-percents",
                "20",
                "40",
                "60",
                "80",
                "100",
                "--jobs",
                "1",
            ]
            if args.retry_failed:
                command.append("--retry-failed")
            yield folder, command


def collect(root):
    manifest = read(root / "memory_run.json")
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError(
            "Expected sampled feature RSS results; collect legacy measurements with their original code"
        )
    rows, pending_suites = [], []
    for suite in manifest["suites"]:
        folder = root / suite
        if not (folder / "job_index.json").exists():
            pending_suites.append(suite)
            continue
        cfg = read(folder / "config.json")
        scaling = suite.startswith("04_ct_hash/")
        if (
            cfg.get("memory_protocol") != MEMORY_VERSION
            or cfg.get("repeats") != 1
            or cfg.get("repeat_start") != 0
            or (not scaling and not cfg.get("features_only"))
        ):
            raise ValueError(f"Expected a single feature-only repetition: {folder}")
        for entry in read(folder / "job_index.json"):
            jobdir = folder / entry["path"]
            result = (
                read(jobdir / "result.json")
                if (jobdir / "result.json").exists()
                else {}
            )
            job = read(jobdir / "job.json") if (jobdir / "job.json").exists() else {}
            resources = result.get("resources", {})
            memory = resources.get("feature_memory", {})
            first = (result.get("rows") or [{}])[0]
            complete = result.get("status") == "complete"
            identity_ok = (
                result.get("fingerprint" if scaling else "job_fingerprint")
                == entry["fingerprint"]
            )
            feature_status = "ok" if scaling and complete else first.get("status")
            observed = sampled_peak_bytes(memory)
            status = (
                result.get("status", "pending")
                if not complete
                else "fingerprint_mismatch"
                if not identity_ok
                else feature_status
                if feature_status != "ok"
                else "memory_missing"
                if memory.get("version") != MEMORY_VERSION
                or memory.get("status") != "ok"
                or not observed
                else "ok"
            )
            row = dict(
                suite=suite,
                dataset=entry.get("dataset", entry.get("source")),
                method=entry["engine"],
                budget=job.get("config", {}).get("budget")
                if scaling
                else entry.get("budget"),
                feature_mode="topk" if scaling else entry.get("feature_mode"),
                length_percent=entry.get("percent")
                if scaling
                else entry.get("length_percent"),
                n=entry.get("n") if scaling else first.get("sequence_length"),
                L=entry.get("L"),
                minsup=entry.get("minsup"),
                minconf=entry.get("minconf"),
                maxsta=entry.get("maxsta"),
                repeat=entry["repeat"],
                status=status,
                job_status=result.get("status", "pending"),
                feature_status=feature_status,
                fingerprint_valid=identity_ok,
                Dim=result.get("Dim") if scaling else first.get("Dim"),
                memory_observations=1 if status == "ok" else 0,
                feature_peak_rss_mib=observed / 2**20 if status == "ok" else None,
                observed_feature_peak_rss_mib=observed / 2**20 if observed else None,
                rss_samples=memory.get("samples"),
                java_rss_samples=memory.get("java_rss_samples"),
                sample_interval_seconds=memory.get("sample_interval_seconds"),
                memory_status=memory.get("status", "missing"),
                memory_boundary=memory.get("boundary"),
                memory_scope=memory.get("scope"),
                input_sha256=result.get("input_sha256")
                if scaling
                else job.get("input_sha256"),
                matrix_sha256=result.get("output_sha256")
                if scaling
                else first.get("diagnostics", {}).get("matrix_sha256"),
                dictionary_sha256=sha(jobdir / "dictionary.json")
                if (jobdir / "dictionary.json").exists()
                else None,
                job_fingerprint=entry["fingerprint"],
                build_sha256=cfg.get(
                    "build_sha256" if scaling else "build_manifest_sha256"
                ),
                path=str(jobdir.relative_to(root)),
                reason=result.get("reason", first.get("reason", "")),
            )
            rows.append(row)
    target = root / "memory_summary"
    target.mkdir(parents=True, exist_ok=True)
    with (target / "memory_by_condition.csv").open("w", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    issues = [r for r in rows if r["status"] != "ok"]
    write(
        target / "audit.json",
        dict(
            statuses=dict(Counter(r["status"] for r in rows)),
            pending_suites=pending_suites,
            issues=issues,
            memory_repetitions=1,
            metric="maximum sampled simultaneous RSS sum of worker and descendants during feature construction",
            old_quality_and_time_results_modified=False,
        ),
    )
    lines = [
        "# Single-run memory measurements",
        "",
        "One fresh feature construction per condition. No clustering. No mean or standard deviation over repetitions.",
        "Memory is the maximum sampled simultaneous RSS sum of the worker and descendants during feature construction.",
        "Includes mining invocation and feature generation, or input standardization and PCA fitting; excludes later feature archive writing, diagnostics, and clustering.",
        "Samples are taken at both feature boundaries and periodically in between. Missing worker or Java samples are invalid, never zero.",
        "Existing repeated runtime and clustering results remain separate. Raw reports its input-ready footprint.",
        "Invalid/insufficient-feature conditions have no headline memory value; their observed footprint is retained separately in CSV.",
        "",
        "| Study | Method | Valid memory | Listed jobs |",
        "|---|---|---:|---:|",
    ]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["suite"], row["method"])].append(row)
    for (suite, method), values in sorted(groups.items()):
        lines.append(
            f"| {suite} | {method} | {sum(r['status'] == 'ok' for r in values)} | {len(values)} |"
        )
    lines += [
        "",
        f"Not yet started suites: {', '.join(pending_suites) or 'none'}.",
        "",
        "Full data: memory_by_condition.csv. Missing/failed/invalid jobs: audit.json.",
        "Merge by suite, dataset, method, feature mode, K, prefix/n/L, minsup, minconf, maxsta; verify input/build/output identity before attaching to old statistics.",
    ]
    (target / "summary.md").write_text("\n".join(lines) + "\n")
    print(
        f"Memory summary: {target}; statuses={dict(Counter(r['status'] for r in rows))}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run", "resume", "collect"])
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/memory_sampled_rss")
    )
    parser.add_argument("--only", nargs="+", choices=list(NATIVE), default=list(NATIVE))
    parser.add_argument(
        "--jobs", type=int, default=12, help="Number of concurrent feature jobs"
    )
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--java-heap", default="16g")
    parser.add_argument("--java-timeout", type=float, default=7200.0)
    parser.add_argument("--job-timeout", type=float, default=10800.0)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="On resume, retry failed/interrupted native jobs; native timeouts remain terminal",
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["CT"])
    parser.add_argument(
        "--include-hash", action="store_true",
        help="Also remeasure CT/CT-Hash sweeps with the same sampled RSS metric",
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.command == "collect":
        collect(args.output)
        return
    if not math.isfinite(args.sample_interval) or args.sample_interval < 0.02:
        parser.error("RSS sampling interval must be finite and >= 0.02 seconds")
    if args.jobs < 1 or len(set(args.only)) != len(args.only):
        parser.error("Positive jobs and unique studies required")
    if args.retry_failed and args.command != "resume":
        parser.error("--retry-failed requires resume")
    if not (args.prepared / "manifest.json").is_file():
        parser.error(
            f"Missing {args.prepared / 'manifest.json'}; pass the existing prepared-data directory"
        )
    steps = list(commands(args))
    if args.command != "plan":
        if sys.platform != "linux":
            parser.error("Measured runs require Linux; no experiments were started")
        marker = dict(
            protocol=PROTOCOL,
            suites=[name for name, _ in steps],
            prepared=str(args.prepared.resolve()),
            jobs=args.jobs,
            methods=args.methods,
            java_heap=args.java_heap,
            java_timeout=args.java_timeout,
            job_timeout=args.job_timeout,
            repeats=1,
            memory_metric=MEMORY_VERSION,
            sample_interval=args.sample_interval,
            features_only=True,
            script_sha256=sha(__file__),
        )
        if args.command == "run":
            if args.output.exists() and (
                not args.output.is_dir() or any(args.output.iterdir())
            ):
                parser.error(
                    "Output is not empty; choose a new directory or use resume"
                )
            write(args.output / "memory_run.json", marker)
        elif (
            not (args.output / "memory_run.json").is_file()
            or read(args.output / "memory_run.json") != marker
        ):
            parser.error(
                "Resume requires the same memory protocol/settings and output root"
            )
    for name, command in steps:
        print(
            f"Memory-only: {name}; {args.command}; one fresh construction per condition",
            flush=True,
        )
        subprocess.run(command, cwd=HERE.parent, check=True)
    if args.command != "plan":
        collect(args.output)


if __name__ == "__main__":
    main()

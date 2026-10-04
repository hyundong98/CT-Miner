"""Six-dataset clustering and input-prefix experiments."""

import argparse
from pathlib import Path
import subprocess
import sys

DATASETS = ("Car", "Beef", "ElectricDevices", "MoteStrain", "PigCVP", "Wafer")
METHODS = ("Raw", "PCA", "CT")
PRESETS = {
    "1": (
        "topk",
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
    "2": (
        "threshold",
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
    "3": (
        "input_length",
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "summarize"))
    parser.add_argument(
        "--only", nargs="+", choices=tuple(PRESETS), default=list(PRESETS)
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=["CT"],
        help="Defaults to CT; optionally include Raw and PCA",
    )
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output-base", type=Path, default=Path("results/clustering"))
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--cores-per-job", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--java-heap", default="16g")
    parser.add_argument("--java-timeout", type=float, default=7200.0)
    parser.add_argument("--job-timeout", type=float, default=10800.0)
    parser.add_argument("--pattern-cap", type=int, default=2_000_000)
    parser.add_argument("--max-feature-cells", type=int, default=100_000_000)
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Fresh feature extraction only; skip clustering",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--repeat-start", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Allow matching results of this new protocol to resume; never import older results",
    )
    args = parser.parse_args()
    if len(set(args.only)) != len(args.only):
        parser.error("Each study may be selected only once")
    if len(set(args.methods)) != len(args.methods):
        parser.error("Each method may be selected only once")
    # Preflight every destination before starting the first requested study.
    # Never delete existing output as a side effect of requesting a fresh run.
    if args.command == "run" and not args.resume:
        for number in args.only:
            output = args.output_base / PRESETS[number][0]
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise ValueError(
                    f"Output already exists: {output}. Choose a new --output-base, or --resume for identical new-protocol jobs."
                )
    for number in args.only:
        folder, preset = PRESETS[number]
        output = args.output_base / folder
        command = [sys.executable, "-m", "ctminer.study", args.command]
        if args.command == "summarize":
            command += [str(output)]
        else:
            command += [
                "--prepared",
                str(args.prepared),
                "--output",
                str(output),
                "--datasets",
                *DATASETS,
                "--methods",
                *args.methods,
                "--mining-protocol",
                "native",
                "--jobs",
                str(args.jobs),
                "--cores-per-job",
                str(args.cores_per_job),
                "--sample-interval",
                str(args.sample_interval),
                "--threads",
                str(args.threads),
                "--java-heap",
                args.java_heap,
                "--java-timeout",
                str(args.java_timeout),
                "--job-timeout",
                str(args.job_timeout),
                "--pattern-cap",
                str(args.pattern_cap),
                "--max-feature-cells",
                str(args.max_feature_cells),
                "--repeats",
                str(args.repeats),
                "--repeat-start",
                str(args.repeat_start),
                *preset,
            ]
            if args.features_only:
                command += ["--features-only"]
        print(f"Study {number}: {folder}; action={args.command}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

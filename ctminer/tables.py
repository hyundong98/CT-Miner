"""Read-only reports for focused grids. Never launch miners or choose best scores."""

from collections import defaultdict
import csv
import itertools
import json
import math
import statistics

METRICS = (
    "Dim",
    "NMI",
    "h",
    "ARI",
    "algorithm_seconds",
    "native_mining_seconds",
    "feature_transform_selection_seconds",
    "feature_seconds",
    "cluster_seconds",
    "wall_seconds",
    "feature_peak_rss_mib",
)


def save_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        if fields:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)


def cell(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "/").replace("\n", " ")


def summary_datasets(cfg):
    return [name for name in cfg["datasets"] if name != "Wafer"]


def aggregate(rows, cfg):
    datasets = summary_datasets(cfg)
    first = cfg.get("repeat_start", 0)
    repeats = set(range(first, first + cfg["repeats"]))
    grouped = defaultdict(list)
    for row in rows:
        if row["dataset"] in datasets and row.get("quality_valid", True):
            grouped[row["dataset"]].append(row)
    included = [
        name
        for name in datasets
        if len(grouped[name]) == len(repeats)
        and {row["repeat"] for row in grouped[name]} == repeats
    ]
    result = dict(
        datasets=len(included),
        dataset_names=",".join(included),
        excluded_datasets=",".join(name for name in datasets if name not in included),
    )
    for metric in METRICS:
        means = []
        for name in included:
            values = [row.get(metric) for row in grouped[name]]
            if all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                means.append(statistics.mean(values))
        result[metric] = (
            statistics.mean(means) if included and len(means) == len(included) else None
        )
    return result


def _matches(
    row, engine, study, percent, budget, minsup, minconf, maxsta, mode, pca_components
):
    expected_budget = (
        0
        if engine == "Raw"
        else pca_components
        if engine == "PCA" and mode == "threshold"
        else budget
    )
    return (
        row["engine"] == engine
        and row["study"] == study
        and row["length_percent"] == percent
        and row["budget"] == expected_budget
        and (
            row.get("minsup") in {None, minsup}
            if engine in {"Raw", "PCA"}
            else row.get("minsup") == minsup
        )
        and (engine != "OPR" or row.get("minconf") == minconf)
        and (engine != "SOPP" or row.get("maxsta") == maxsta)
    )


def write_tables(root, cfg, jobs, summaries):
    quality = {row["path"]: row for row in summaries}
    attempts = []
    for job in jobs:
        row = dict(job)
        row.update(quality.get(job["path"], {}))
        row["quality_valid"] = job["path"] in quality
        row["analysis_status"] = (
            "ok"
            if row["quality_valid"]
            else row.get("feature_status", row["status"])
            if row["status"] == "complete"
            else row["status"]
        )
        if (
            row["status"] == "complete"
            and row["analysis_status"] == "ok"
            and not row["quality_valid"]
        ):
            row["analysis_status"] = (
                "mining_only" if row["study"] == "length" else "invalid_quality"
            )
        for key in METRICS:
            row.setdefault(key, None)
        attempts.append(row)
    save_csv(
        root / "all_conditions.csv",
        [
            dict(row, **{key: None for key in METRICS})
            if not row["quality_valid"]
            and row["study"] in {"quality", "prefix-quality"}
            else row
            for row in attempts
        ],
    )
    lines = [
        "# All per-dataset conditions",
        "",
        "No hyperparameter selection or averaging across settings. Each row averages clustering seeds only; repetitions stay separate.",
        "Missing/insufficient results remain visible. Runtime for failed jobs is not a completed algorithm runtime.",
        "Memory is the peak sampled simultaneous RSS sum of the worker and descendants during feature construction, excluding clustering. Whole-job RSS is a separate CSV diagnostic.",
        "",
    ]
    fields = (
        "engine",
        "budget",
        "length_percent",
        "minsup",
        "minconf",
        "maxsta",
        "repeat",
        "analysis_status",
        "Dim",
        "NMI",
        "h",
        "ARI",
        "algorithm_seconds",
        "feature_transform_selection_seconds",
        "feature_seconds",
        "wall_seconds",
        "feature_peak_rss_mib",
    )
    labels = (
        "Method",
        "K (0=all/Raw)",
        "Input %",
        "minsup",
        "minconf",
        "maxsta",
        "R",
        "Status",
        "Dim",
        "NMI",
        "h",
        "ARI",
        "Algorithm s",
        "Transform s",
        "Feature s",
        "Job s",
        "Sampled peak RSS MiB",
    )
    # Preserve the saved-result schema, but show legacy parameters only when
    # the report actually contains the corresponding method.
    visible = [
        (field, label)
        for field, label in zip(fields, labels)
        if (field != "minconf" or "OPR" in cfg["methods"])
        and (field != "maxsta" or "SOPP" in cfg["methods"])
    ]
    fields, labels = zip(*visible)
    for dataset in cfg["datasets"]:
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| " + " | ".join(labels) + " |",
                "|" + "---|" * len(fields),
            ]
        )
        for row in attempts:
            if row["dataset"] == dataset:
                lines.append(
                    "| "
                    + " | ".join(
                        cell(
                            None
                            if not row["quality_valid"] and k in METRICS
                            else row.get(k)
                        )
                        for k in fields
                    )
                    + " |"
                )
        lines.append("")
    (root / "dataset_tables.md").write_text("\n".join(lines) + "\n")

    # Average each method independently; an infeasible PCA fit does not remove
    # valid mining results. References are displayed in each relevant slice.
    mode = cfg.get("feature_mode", "topk")
    thresholds = (
        cfg.get("minsups", [2.0]) if cfg.get("mining_protocol") == "native" else [None]
    )
    confidences = (
        (cfg.get("opr_minconfs") or [cfg.get("opr_minconf", 0.65)])
        if "OPR" in cfg["methods"]
        else [None]
    )
    stabilities = (
        (cfg.get("sopp_maxstas") or [cfg.get("sopp_max_cv", 0.5)])
        if "SOPP" in cfg["methods"]
        else [None]
    )
    # Older saved jobs have no minconf/maxsta key; retain their historic protocol
    # without pretending they were newly executed grid conditions.
    if not any("minconf" in j for j in jobs):
        confidences = [None]
    if not any("maxsta" in j for j in jobs):
        stabilities = [None]
    studies = sorted(
        {
            (r["study"], r["length_percent"])
            for r in jobs
            if r["study"] in {"quality", "prefix-quality"}
        }
    )
    slices = []
    coverage = []
    for (study, percent), budget, minsup, minconf, maxsta in itertools.product(
        studies,
        [0] if mode == "threshold" else cfg["budgets"],
        thresholds,
        confidences,
        stabilities,
    ):
        condition = dict(
            study=study,
            length_percent=percent,
            budget=budget,
            minsup=minsup,
            opr_minconf=minconf,
            sopp_maxsta=maxsta,
            feature_mode=mode,
        )
        by_engine = {
            e: [
                r
                for r in attempts
                if r["quality_valid"]
                and _matches(
                    r,
                    e,
                    study,
                    percent,
                    budget,
                    minsup,
                    minconf,
                    maxsta,
                    mode,
                    cfg.get("pca_components", 30),
                )
            ]
            for e in cfg["methods"]
        }
        for engine, rows in by_engine.items():
            result = dict(condition, engine=engine, **aggregate(rows, cfg))
            coverage.append(
                dict(
                    condition,
                    engine=engine,
                    included=result["dataset_names"].split(",")
                    if result["dataset_names"]
                    else [],
                    excluded=result["excluded_datasets"].split(",")
                    if result["excluded_datasets"]
                    else [],
                )
            )
            slices.append(result)
    save_csv(root / "matched_slices.csv", slices)
    (root / "matched_slice_coverage.json").write_text(
        json.dumps(coverage, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Main-dataset comparisons for each experimental condition",
        "",
        "Wafer is excluded from aggregates and retained in the per-dataset tables. Means average seeds, then repetitions, then available datasets equally for each method and condition. Dataset counts and membership are recorded in CSV.",
        "Unavailable experiments are shown as dashes in the per-dataset table. Missing PCA results do not exclude another method's results. PCA means use only feasible datasets; no value is imputed.",
        "Algorithm time is native mining for miners, scaling plus PCA fit/transform for PCA, and zero for Raw. Feature time additionally includes common transform and transport for miners.",
        "Hyperparameters are never averaged or selected using NMI/h/ARI. Reference and parameter-independent rows can reappear across slices; their times must not be summed as new runs.",
        "",
    ]
    previous = None
    for row in slices:
        key = tuple(
            row[k]
            for k in (
                "study",
                "length_percent",
                "budget",
                "minsup",
                "opr_minconf",
                "sopp_maxsta",
            )
        )
        if key != previous:
            study, percent, budget, support, conf, sta = key
            heading = (
                f"## {study}: input {percent}%, "
                f"{'all features' if mode == 'threshold' else 'K=' + str(budget)}, "
                f"minsup={support}"
            )
            if "OPR" in cfg["methods"] and conf is not None:
                heading += f", OPR minconf={conf}"
            if "SOPP" in cfg["methods"] and sta is not None:
                heading += f", SOPP maxsta={sta}"
            lines += [
                heading,
                "",
                "| Method | Datasets | Dim | NMI | h | ARI | Algorithm s | Transform s | Feature s | Job s | Sampled peak RSS MiB |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            previous = key
        keys = (
            "engine",
            "datasets",
            "Dim",
            "NMI",
            "h",
            "ARI",
            "algorithm_seconds",
            "feature_transform_selection_seconds",
            "feature_seconds",
            "wall_seconds",
            "feature_peak_rss_mib",
        )
        lines.append("| " + " | ".join(cell(row.get(k)) for k in keys) + " |")
    (root / "matched_slices.md").write_text("\n".join(lines) + "\n")

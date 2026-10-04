"""Post-hoc interpretation of saved focused-study CT dictionaries.

Does not mine, fit, cluster, build Java, or modify experiment results. Labels
are used only for explicitly marked descriptive analysis, never new evaluation.

"""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def array_hash(x):
    return hashlib.sha256(
        str(x.shape).encode() + np.ascontiguousarray(x, dtype="<f8").tobytes()
    ).hexdigest()


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def code_id(code):
    return "ct_" + hashlib.sha256(json.dumps(list(code)).encode()).hexdigest()[:20]


def ct_parents(code):
    """Decode the saved previous-smaller-distance code into a min-Cartesian tree."""
    left, right, stack = [-1] * len(code), [-1] * len(code), []
    for i, distance in enumerate(code):
        previous = i - distance if distance else -1
        if distance < 0 or distance > i or (previous != -1 and previous not in stack):
            raise ValueError(f"Invalid CT code: {code}")
        last = -1
        while stack and stack[-1] != previous:
            last = stack.pop()
        if stack:
            right[stack[-1]] = i
        if last >= 0:
            left[i] = last
        stack.append(i)
    parent = [-1] * len(code)
    for i in range(len(code)):
        for j in (left[i], right[i]):
            if j >= 0:
                parent[j] = i
    return stack[0], parent


def pd(values):
    stack, code = [], []
    for i, value in enumerate(values):
        while stack and values[stack[-1]] > value:
            stack.pop()
        code.append(i - stack[-1] if stack else 0)
        stack.append(i)
    return tuple(code)


def tree_svg(code):
    _, parents = ct_parents(code)
    depths = []
    for i in range(len(code)):
        depth, at = 0, i
        while parents[at] >= 0:
            depth += 1
            at = parents[at]
        depths.append(depth)
    width, height = max(280, len(code) * 28), 60 + 32 * max(depths)
    xy = [
        (20 + i * (width - 40) / max(1, len(code) - 1), 25 + 32 * d)
        for i, d in enumerate(depths)
    ]
    lines = [f'<svg viewBox="0 0 {width} {height}" aria-label="Min-Cartesian tree">']
    for i, p in enumerate(parents):
        if p >= 0:
            x, y = xy[i]
            a, b = xy[p]
            lines.append(f'<path d="M{x},{y} L{a},{b}" stroke="#64748b"/>')
    for i, (x, y) in enumerate(xy):
        lines.append(
            f'<circle cx="{x}" cy="{y}" r="10" fill="#dbeafe"/><text x="{x}" y="{y + 4}" text-anchor="middle" font-size="10">{i}</text>'
        )
    return "".join(lines) + "</svg>"


def waveform_svg(values):
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    points = " ".join(
        f"{15 + 570 * i / max(1, len(values) - 1):.2f},{130 - 110 * (v - lo) / span:.2f}"
        for i, v in enumerate(values)
    )
    return f'<svg viewBox="0 0 600 150" aria-label="Observed pattern occurrence"><polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2"/></svg>'


def table(headers, rows):
    esc = lambda v: html.escape(str(v))
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{esc(v)}</th>" for v in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table>"
    )


def load_saved(root, entry):
    work = (root / entry["path"]).resolve()
    if not work.is_relative_to(root):
        raise ValueError("Job path leaves results directory")
    job, result = read(work / "job.json"), read(work / "result.json")
    if (
        job["engine"] != "CT"
        or job["study"] != "quality"
        or job.get("length_percent", 100) != 100
    ):
        raise ValueError("Only full-length CT quality jobs are supported")
    for key in ("dataset", "engine", "budget", "repeat", "study", "length_percent"):
        if job[key] != entry[key]:
            raise ValueError(f"Job/index mismatch: {key}")
    if (
        result.get("status") != "complete"
        or result.get("job_fingerprint") != entry["fingerprint"]
        or job["fingerprint"] != entry["fingerprint"]
    ):
        raise ValueError("Job is incomplete or saved fingerprints disagree")
    core = {k: v for k, v in job.items() if k not in {"fingerprint", "execution"}}
    calculated = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
    if calculated != job["fingerprint"]:
        raise ValueError("Saved job fingerprint does not match job contents")
    dictionary = read(work / "dictionary.json")
    if dictionary.get("engine") != "CT" or dictionary.get("feature_value") != "support":
        raise ValueError("Expected an unweighted CT dictionary")
    with np.load(work / "selected_features.npz", allow_pickle=False) as archive:
        values, labels = np.asarray(archive["X"], dtype=float), archive["y"]
    pats = dictionary["patterns"]
    if (
        values.ndim != 2
        or labels.ndim != 1
        or len(labels) != len(values)
        or len(values) == 0
    ):
        raise ValueError("Invalid feature/label dimensions")
    if values.shape[1] != job["budget"] or len(pats) != values.shape[1]:
        raise ValueError("Incomplete dictionary; no feature padding permitted")
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.equal(values, np.floor(values)).all()
    ):
        raise ValueError(
            "CT features must contain finite nonnegative occurrence counts"
        )
    codes = []
    for col, item in enumerate(pats):
        if (
            item["column"] != col
            or any(type(v) is not int for v in item["code"])
            or len(item["code"]) < 2
        ):
            raise ValueError("Invalid pattern-column mapping")
        code = tuple(item["code"])
        ct_parents(code)
        if not np.isclose(values[:, col].sum(), item["support"], rtol=0, atol=1e-8):
            raise ValueError(f"Saved support does not equal column sum: column {col}")
        codes.append(code)
    if len(set(codes)) != len(codes):
        raise ValueError("Duplicate dictionary patterns")
    result_rows = result.get("rows", [])
    seeds = job["config"]["seeds"]
    if len(result_rows) != len(seeds) or {r.get("seed") for r in result_rows} != set(
        seeds
    ):
        raise ValueError("Missing or duplicate clustering seed results")
    for row in result_rows:
        if (
            row.get("status") != "ok"
            or row.get("Dim") != values.shape[1]
            or row.get("budget") != job["budget"]
        ):
            raise ValueError("Invalid saved quality result")
        expected_hash = row.get("diagnostics", {}).get("matrix_sha256")
        if expected_hash is None or expected_hash != array_hash(values):
            raise ValueError("Feature matrix differs from evaluated matrix")
    return work, job, result, dictionary, values, labels, codes


def prepared_input(job, labels, result, prepared):
    if prepared is None:
        return None, None
    manifest = read(prepared / "manifest.json")
    record = manifest["datasets"][job["dataset"]]
    path = (prepared / record["npz"]).resolve()
    if not path.is_relative_to(prepared) or record.get("status") != "ready":
        raise ValueError("Invalid prepared dataset path/status")
    actual = digest(path)
    if actual != job["input_sha256"] or actual != record["sha256"]:
        raise ValueError("Prepared NPZ differs from original experiment input")
    with np.load(path, allow_pickle=False) as archive:
        x, y = archive["X"], archive["y"]
    if (
        not np.array_equal(y, labels)
        or x.ndim != 2
        or len(x) != len(labels)
        or not np.isfinite(x).all()
    ):
        raise ValueError("Prepared input/label mismatch")
    # Reproduce the .9g numerical input convention used in the actual experiment.
    x = np.asarray([[float(format(float(v), ".9g")) for v in row] for row in x])
    expected = {row.get("common_input_sha256") for row in result["rows"]}
    if expected != {array_hash(x)}:
        raise ValueError("Quantized input differs from evaluated input")
    return x, dict(
        path=str(path),
        sha256=actual,
        numerical_input=".9g",
        matching="strict distinct-valued contiguous windows",
    )


def analyze_columns(dataset, budget, values, labels, codes):
    classes, inverse = np.unique(labels, return_inverse=True)
    groups = [np.flatnonzero(inverse == c) for c in range(len(classes))]
    summaries, distributions = [], []
    for col, code in enumerate(codes):
        counts = values[:, col]
        average = float(counts.mean())
        total_variation = float(np.sum((counts - average) ** 2))
        between = 0.0
        class_rows = []
        for label, ids in zip(classes, groups):
            local = counts[ids]
            mean = float(local.mean())
            between += len(ids) * (mean - average) ** 2
            class_rows.append(
                dict(
                    dataset=dataset,
                    budget=budget,
                    pattern_id=code_id(code),
                    column=col,
                    class_label=str(label),
                    n_samples=len(ids),
                    support=int(local.sum()),
                    present_samples=int(np.count_nonzero(local)),
                    presence_fraction=float(np.mean(local > 0)),
                    mean_count=mean,
                    std_count=float(local.std()),
                    median_count=float(np.median(local)),
                    mean_count_ratio_to_overall=mean / average if average else None,
                )
            )
        distributions.extend(class_rows)
        strongest = max(class_rows, key=lambda r: r["mean_count"])
        summaries.append(
            dict(
                dataset=dataset,
                budget=budget,
                pattern_id=code_id(code),
                column=col,
                frequency_rank=col + 1,
                length=len(code),
                code=json.dumps(code),
                support=int(counts.sum()),
                present_samples=int(np.count_nonzero(counts)),
                presence_fraction=float(np.mean(counts > 0)),
                mean_count=average,
                std_count=float(counts.std()),
                constant_column=total_variation == 0,
                eta_squared=min(1.0, max(0.0, between / total_variation))
                if total_variation
                else 0.0,
                highest_mean_class=strongest["class_label"],
                highest_class_mean=strongest["mean_count"],
                highest_class_presence=strongest["presence_fraction"],
            )
        )
    # Validate that the persisted column order is the advertised global frequency ranking.
    order = sorted(
        range(len(codes)),
        key=lambda i: (-summaries[i]["support"], len(codes[i]), codes[i]),
    )
    if order != list(range(len(codes))):
        raise ValueError(
            "Dictionary order differs from frequency/length/lexicographic ranking"
        )
    contrast = sorted(
        range(len(codes)), key=lambda i: (-summaries[i]["eta_squared"], i)
    )
    for rank, col in enumerate(contrast, 1):
        summaries[col]["posthoc_contrast_rank"] = rank
    return summaries, distributions, contrast


def occurrence_examples(x, labels, counts, code, limit):
    """Deterministic count-based sample selection; labels never rank examples."""
    examples = []
    for sample in sorted(
        np.flatnonzero(counts > 0), key=lambda i: (-counts[i], int(i))
    ):
        sequence = x[sample]
        for start in range(len(sequence) - len(code) + 1):
            window = sequence[start : start + len(code)]
            if len(set(window)) != len(code) or pd(window) != code:
                continue
            examples.append(
                dict(
                    sample_index=int(sample),
                    class_label=str(labels[sample]),
                    start=start,
                    end_exclusive=start + len(code),
                    sample_pattern_count=int(counts[sample]),
                    values=window.tolist(),
                    ordinal_ranks=(np.argsort(np.argsort(window)) + 1).tolist(),
                )
            )
            break
        else:
            raise ValueError(
                f"Positive saved count but no strict CT occurrence in sample {sample}"
            )
        if len(examples) >= limit:
            break
    return examples


def render_report(dataset, budget, summaries, class_rows, selected, examples, quality):
    esc = html.escape
    page = [
        '<!doctype html><html lang="en"><meta charset="utf-8">',
        f"<title>{esc(dataset)} CT K{budget}</title>",
        "<style>body{max-width:1150px;margin:30px auto;padding:0 20px;font:15px system-ui;color:#1e293b}table{border-collapse:collapse;width:100%;margin:14px 0}th,td{border-bottom:1px solid #ddd;padding:7px;text-align:right}th:first-child,td:first-child{text-align:left}section{border-top:2px solid #ddd;margin-top:28px;padding-top:12px}svg{display:block;max-width:680px;max-height:350px}code{overflow-wrap:anywhere}small{color:#475569}</style>",
        f"<h1>{esc(dataset)} — CT K{budget}</h1>",
        "<p>Saved dictionary and evaluated count features; no mining or clustering rerun.</p>",
        "<p>Frequency representatives preserve label-free dictionary order. Contrast representatives use labels post hoc: η² is the fraction of count variance explained by class means, not held-out accuracy or a significance test.</p>",
        "<p>Code is previous-smaller distance, with zero for no predecessor. Tree node labels are zero-based window positions, not amplitudes. Example windows are real .9g-quantized observations; each plot uses its own vertical range.</p>",
        table(
            ["NMI (seed mean)", "h (seed mean)", "ARI (seed mean)"],
            [[f"{quality[k]:.4f}" for k in ("NMI", "h", "ARI")]],
        ),
        table(
            [
                "Rank",
                "Length",
                "CT code",
                "Support",
                "Presence",
                "η² (post hoc)",
                "Representative",
            ],
            [
                [
                    r["frequency_rank"],
                    r["length"],
                    r["code"],
                    r["support"],
                    f"{r['presence_fraction']:.3f}",
                    f"{r['eta_squared']:.4f}",
                    r.get("representative_reason", ""),
                ]
                for r in summaries
            ],
        ),
    ]
    for col in selected:
        row = summaries[col]
        code = json.loads(row["code"])
        page += [
            f"<section><h2>Rank {row['frequency_rank']}: {esc(row['representative_reason'])}</h2>",
            f"<code>{esc(row['code'])}</code>",
            tree_svg(code),
        ]
        local = [r for r in class_rows if r["column"] == col]
        local.sort(key=lambda r: (-r["mean_count"], r["class_label"]))
        page += [
            "<details open><summary>Class distributions — all classes; raw totals and per-sample means</summary>",
            table(
                [
                    "Class",
                    "Samples",
                    "Support",
                    "Mean count",
                    "Presence",
                    "Mean / overall",
                ],
                [
                    [
                        r["class_label"],
                        r["n_samples"],
                        r["support"],
                        f"{r['mean_count']:.4f}",
                        f"{r['presence_fraction']:.3f}",
                        f"{r['mean_count_ratio_to_overall']:.3f}"
                        if r["mean_count_ratio_to_overall"] is not None
                        else "NA",
                    ]
                    for r in local
                ],
            ),
            "</details>",
        ]
        for example in examples.get(row["pattern_id"], []):
            page += [
                f"<p>Sample {example['sample_index']}; class {esc(example['class_label'])}; window [{example['start']}, {example['end_exclusive']}).</p>",
                waveform_svg(example["values"]),
            ]
        page.append("</section>")
    return "\n".join(page) + "</html>\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument(
        "--prepared",
        type=Path,
        help="Optional prepared NPZ directory for real occurrence examples",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        help="Default: all datasets in the saved run configuration",
    )
    ap.add_argument("--budgets", nargs="+", type=int, default=[10, 20, 30, 40, 50])
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument(
        "--top", type=int, default=3, help="Frequency representatives per dictionary"
    )
    ap.add_argument(
        "--contrast-top",
        type=int,
        default=3,
        help="Additional label-aware descriptive representatives; 0 disables",
    )
    ap.add_argument(
        "--examples",
        type=int,
        default=2,
        help="At most this many real windows per representative, one per sample",
    )
    args = ap.parse_args()
    if (
        args.top < 1
        or args.contrast_top < 0
        or args.examples < 1
        or args.repeat < 0
        or not args.budgets
        or min(args.budgets) < 1
    ):
        ap.error("Invalid representative counts, budgets or repeat")
    root, output = args.results.resolve(), args.output.resolve()
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        ap.error("Output must be separate from the experiment results directory")
    if output.exists():
        ap.error("Output already exists; choose a new output directory")
    prepared = args.prepared.resolve() if args.prepared else None
    cfg = read(root / "config.json")
    datasets = args.datasets or cfg["datasets"]
    if any(
        not isinstance(d, str) or d in {"", ".", ".."} or "/" in d or "\\" in d
        for d in datasets
    ):
        ap.error("Dataset names must be single directory names")
    if len(set(datasets)) != len(datasets) or len(set(args.budgets)) != len(
        args.budgets
    ):
        ap.error("Duplicate datasets or budgets")
    indexed = {}
    for entry in read(root / "job_index.json"):
        if (
            entry["engine"] == "CT"
            and entry["study"] == "quality"
            and entry["length_percent"] == 100
            and entry["repeat"] == args.repeat
        ):
            key = entry["dataset"], entry["budget"]
            if key in indexed:
                raise ValueError(f"Duplicate CT job: {key}")
            indexed[key] = entry
    missing = [(d, k) for d in datasets for k in args.budgets if (d, k) not in indexed]
    if missing:
        ap.error(f"Requested CT conditions absent from job index: {missing}")
    output.mkdir(parents=True)
    all_patterns, all_classes, representatives, changes, jobs, errors = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    summary = [
        "# Representative CT patterns",
        "",
        "Post-hoc analysis of saved full-length CT dictionaries. No new mining, fitting or clustering.",
        "Class-aware representatives are descriptive selections using all labels, not independently validated predictive features.",
        "Frequency and contrast representatives are reported separately. Tree structure does not define a unique waveform.",
        "Cross-budget changes compare successive requested budgets; reported additions are not necessarily contiguous frequency ranks.",
        "",
        "| Dataset | K | NMI | h | ARI | Representatives | Report |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for dataset in datasets:
        previous = None
        cached_input = None
        exemplar_cache = {}
        seen_counts = {}
        for budget in sorted(args.budgets):
            entry = indexed[dataset, budget]
            try:
                work, job, result, dictionary, values, labels, codes = load_saved(
                    root, entry
                )
                if cached_input is None:
                    x, provenance = prepared_input(job, labels, result, prepared)
                    cached_input = (job["input_sha256"], labels.copy(), x, provenance)
                else:
                    sha, old_labels, x, provenance = cached_input
                    if sha != job["input_sha256"] or not np.array_equal(
                        labels, old_labels
                    ):
                        raise ValueError("Different input/labels across budgets")
                    if x is not None and {
                        r.get("common_input_sha256") for r in result["rows"]
                    } != {array_hash(x)}:
                        raise ValueError("Different quantized inputs across budgets")
                patterns, distributions, contrast = analyze_columns(
                    dataset, budget, values, labels, codes
                )
                for col, code in enumerate(codes):
                    if code in seen_counts and not np.array_equal(
                        values[:, col], seen_counts[code]
                    ):
                        raise ValueError(
                            "Same CT pattern has different counts across requested budgets"
                        )
                frequency_cols = set(range(min(args.top, len(codes))))
                contrast_cols = set(contrast[: args.contrast_top])
                selected = sorted(frequency_cols | contrast_cols)
                for col in selected:
                    reasons = []
                    if col in frequency_cols:
                        reasons.append("frequency")
                    if col in contrast_cols:
                        reasons.append("posthoc_label_contrast")
                    patterns[col]["representative_reason"] = "+".join(reasons)
                change = None
                if previous is not None:
                    old_budget, old_codes, old_values = previous
                    old_map = {c: i for i, c in enumerate(old_codes)}
                    for col, code in enumerate(codes):
                        if code in old_map and not np.array_equal(
                            values[:, col], old_values[:, old_map[code]]
                        ):
                            raise ValueError(
                                "Same CT pattern has different counts across budgets"
                            )
                    added, removed = (
                        set(codes) - set(old_codes),
                        set(old_codes) - set(codes),
                    )
                    change = dict(
                        dataset=dataset,
                        from_budget=old_budget,
                        to_budget=budget,
                        smaller_is_subset=not removed,
                        exact_prefix=codes[: len(old_codes)] == old_codes,
                        added_pattern_ids=json.dumps(
                            [code_id(c) for c in codes if c in added]
                        ),
                        removed_pattern_ids=json.dumps(
                            [code_id(c) for c in old_codes if c in removed]
                        ),
                        added_codes=json.dumps([c for c in codes if c in added]),
                        removed_codes=json.dumps(
                            [c for c in old_codes if c in removed]
                        ),
                    )
                examples = {}
                if x is not None:
                    for col in selected:
                        code = codes[col]
                        if code not in exemplar_cache:
                            exemplar_cache[code] = occurrence_examples(
                                x, labels, values[:, col], code, args.examples
                            )
                        examples[code_id(code)] = exemplar_cache[code]
                quality = {
                    k: float(np.mean([r[k] for r in result["rows"]]))
                    for k in ("NMI", "h", "ARI")
                }
                if not all(np.isfinite(v) for v in quality.values()):
                    raise ValueError("Nonfinite saved quality metric")
                target = output / dataset / f"K{budget}"
                target.mkdir(parents=True)
                write_json(
                    target / "examples.json",
                    dict(
                        input=provenance,
                        indexing="zero-based, end-exclusive",
                        selection="samples ordered by saved pattern count descending, then sample index; first matching strict window per sample; labels do not rank examples",
                        patterns=[
                            dict(
                                pattern_id=code_id(codes[c]),
                                code=list(codes[c]),
                                tree_root=ct_parents(codes[c])[0],
                                tree_parents=ct_parents(codes[c])[1],
                                examples=examples.get(code_id(codes[c]), []),
                            )
                            for c in selected
                        ],
                    ),
                )
                (target / "report.html").write_text(
                    render_report(
                        dataset,
                        budget,
                        patterns,
                        distributions,
                        selected,
                        examples,
                        quality,
                    )
                )
                all_patterns.extend(patterns)
                all_classes.extend(distributions)
                representatives.extend(patterns[c] for c in selected)
                if change is not None:
                    changes.append(change)
                for col, code in enumerate(codes):
                    if code not in seen_counts:
                        seen_counts[code] = values[:, col].copy()
                jobs.append(
                    dict(
                        dataset=dataset,
                        budget=budget,
                        status="ok",
                        source=str(work),
                        fingerprint=job["fingerprint"],
                        input_sha256=job["input_sha256"],
                        dictionary_sha256=digest(work / "dictionary.json"),
                        selected_features_sha256=digest(work / "selected_features.npz"),
                        result_sha256=digest(work / "result.json"),
                        quality=quality,
                        representatives=len(selected),
                        examples_enabled=x is not None,
                    )
                )
                report = (target / "report.html").relative_to(output).as_posix()
                summary.append(
                    f"| {dataset} | {budget} | {quality['NMI']:.4f} | {quality['h']:.4f} | {quality['ARI']:.4f} | {len(selected)} | [report]({report}) |"
                )
                previous = budget, codes, values
                print(
                    f"{dataset} / K{budget}: {len(codes)} patterns, {len(selected)} representatives",
                    flush=True,
                )
            except Exception as exc:
                errors.append(
                    dict(
                        dataset=dataset,
                        budget=budget,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                )
                previous = None
                print(
                    f"{dataset} / K{budget}: ERROR {exc}", file=sys.stderr, flush=True
                )
    write_csv(output / "patterns.csv", all_patterns)
    write_csv(output / "class_stats.csv", all_classes)
    write_csv(output / "representatives.csv", representatives)
    write_csv(output / "budget_changes.csv", changes)
    catalog = {}
    for row in all_patterns:
        key = row["dataset"], row["pattern_id"]
        if key not in catalog:
            catalog[key] = {
                k: v
                for k, v in row.items()
                if k
                not in {
                    "budget",
                    "column",
                    "frequency_rank",
                    "posthoc_contrast_rank",
                    "representative_reason",
                }
            }
            catalog[key].update(selected_budgets=[], representative_budgets=[])
        catalog[key]["selected_budgets"].append(row["budget"])
        if row.get("representative_reason"):
            catalog[key]["representative_budgets"].append(row["budget"])
    catalog_rows = []
    for row in catalog.values():
        catalog_rows.append(
            dict(
                row,
                first_observed_budget=min(row["selected_budgets"]),
                selected_budgets=json.dumps(row["selected_budgets"]),
                representative_budgets=json.dumps(row["representative_budgets"]),
            )
        )
    write_csv(output / "pattern_catalog.csv", catalog_rows)
    write_json(
        output / "audit.json",
        dict(
            source=str(root),
            datasets=datasets,
            budgets=sorted(args.budgets),
            top=args.top,
            contrast_top=args.contrast_top,
            examples=args.examples,
            jobs=jobs,
            definitions=dict(
                eta_squared="sum_class n_class*(class_mean-overall_mean)^2 / sum_sample (count-overall_mean)^2; constant columns get 0",
                frequency_rank="1-based original dictionary column order; ties: shorter length then lexicographic code",
                feature="raw per-sample occurrence count; stability/label not used to remine",
                presence_fraction="fraction of sequences with positive occurrence count",
                highest_mean_class="descriptive post-hoc class; not a pattern label or predictive claim",
            ),
            no_experiments_executed=True,
        ),
    )
    write_json(output / "errors.json", errors)
    summary += [
        "",
        f"Successful conditions: {len(jobs)}. Errors: {len(errors)}. See errors.json.",
        "patterns.csv: all selected patterns. class_stats.csv: all class distributions. representatives.csv: displayed patterns.",
        "pattern_catalog.csv: one row per dataset and unique CT code; budgets retained without summing repeated supports.",
        "budget_changes.csv: saved dictionary nesting, added/removed patterns and verified shared-column counts.",
        "Real windows require --prepared. Example lookup is reporting work and is excluded from experimental timing.",
    ]
    (output / "summary.md").write_text("\n".join(summary) + "\n")
    links = [
        [
            j["dataset"],
            j["budget"],
            j["representatives"],
            f"{j['dataset']}/K{j['budget']}/report.html",
        ]
        for j in jobs
    ]
    page = [
        '<!doctype html><meta charset="utf-8"><title>CT pattern reports</title><h1>CT pattern reports</h1>',
        "<p>Saved dictionaries; label-aware contrast is post-hoc interpretation. See summary.md and audit.json.</p>",
    ]
    for d, k, n, path in links:
        page.append(
            f'<p><a href="{html.escape(path, quote=True)}">{html.escape(d)} K{k}</a> ({n} representatives)</p>'
        )
    page.append(f"<p>Errors: {len(errors)}; see errors.json.</p>")
    (output / "index.html").write_text("\n".join(page))
    print(f"Report: {output / 'index.html'}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

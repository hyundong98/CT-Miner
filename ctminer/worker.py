"""Dataset preparation and clustering workers."""

from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import signal
from time import perf_counter
import traceback

import numpy as np


HERE = Path(__file__).resolve().parent
ENGINES = ["Raw", "PCA", "CT"]
LANGUAGES = {"Raw": "Python/sklearn", "PCA": "Python/sklearn", "CT": "Java"}


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        + "\n"
    )
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def package_versions():
    result = {}
    for name in ("aeon", "numpy", "scipy", "scikit-learn", "threadpoolctl"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def prepare(args):
    from aeon.datasets import tsc_datasets
    from ctminer.data import load as load_archive

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    catalogues = {
        "UCR": list(
            getattr(tsc_datasets, "UCR2019", getattr(tsc_datasets, "univariate", []))
        )
    }
    if args.catalogue == "all-univariate":
        catalogues["redux"] = list(getattr(tsc_datasets, "redux", []))
    names = sorted({n for values in catalogues.values() for n in values})
    if args.datasets:
        unknown = set(args.datasets) - set(names)
        if unknown:
            raise ValueError(
                f"Datasets outside installed aeon catalogue: {sorted(unknown)}"
            )
        names = sorted(set(args.datasets))
    if not names or any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names):
        raise ValueError("Empty or invalid catalogue")
    spec = dict(
        catalogue=args.catalogue,
        names=names,
        available_catalogues=catalogues,
        archive_variants=args.archive_variants,
        packages=package_versions(),
    )
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest["spec"] != spec:
            raise ValueError(
                "Preparation configuration/version changed; use a new output directory"
            )
    else:
        manifest = dict(spec=spec, datasets={})
    for number, name in enumerate(names, 1):
        old = manifest["datasets"].get(name)
        if old and not (args.retry_failed and old["status"] == "failed"):
            if old["status"] == "ready" and (
                not (root / old["npz"]).is_file()
                or digest(root / old["npz"]) != old["sha256"]
            ):
                raise ValueError(f"Prepared file missing/changed for {name}")
            print(
                f"[{number}/{len(names)}] {name}: retained {old['status']}", flush=True
            )
            continue
        load_audit = dict(loader_sha256=digest(HERE / "data.py"))
        try:
            data, labels, meta = load_archive(
                name,
                args.cache,
                root / "downloads/zenodo",
                args.archive_variants == "compatible",
                load_audit,
            )
            series = []
            for row in data:
                row = np.asarray(row, dtype=np.float64)
                if row.ndim == 2 and row.shape[0] == 1:
                    row = row[0]
                if row.ndim != 1:
                    raise ValueError("Not univariate")
                series.append(row)
            y = np.asarray(labels).astype(str)
            lengths = sorted({len(x) for x in series})
            record = dict(
                name=name,
                metadata=meta,
                n_samples=len(y),
                lengths=lengths,
                n_classes=len(set(y)),
                class_counts=dict(Counter(y)),
                source_files=load_audit.get("source_files")
                or [
                    dict(path=str(p), sha256=digest(p))
                    for p in sorted((args.cache.resolve() / name).glob("*.ts"))
                ],
                download=load_audit,
                scope="official TRAIN and TEST combined; no label used for dictionary or scaling",
            )
            if len(series) != len(y) or not len(y):
                raise ValueError("Invalid sample/label count")
            reasons = []
            if len(lengths) != 1:
                reasons.append("unequal length: no common Raw vector space")
            if not lengths or min(lengths) < 2:
                reasons.append("series shorter than 2")
            if any(not np.isfinite(x).all() for x in series):
                reasons.append("missing/nonfinite values remain")
            if not 1 < len(set(y)) < len(y):
                reasons.append("need 2..n_samples-1 classes")
            if reasons:
                record.update(status="excluded", reason="; ".join(reasons))
            else:
                x = np.asarray(series, dtype=np.float64)
                filename = f"data/{name}.npz"
                target = root / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target, X=x, y=y)
                record.update(
                    status="ready",
                    npz=filename,
                    sha256=digest(target),
                    raw_Dim=x.shape[1],
                )
            manifest["datasets"][name] = record
        except Exception as exc:
            manifest["datasets"][name] = dict(
                name=name,
                status="failed",
                reason=str(exc),
                traceback=traceback.format_exc(),
                download=load_audit,
            )
        if old and old["status"] == "failed":
            history = old.get("previous_failures", []) + [
                {k: v for k, v in old.items() if k != "previous_failures"}
            ]
            manifest["datasets"][name]["previous_failures"] = history
        write(manifest_path, manifest)
        row = manifest["datasets"][name]
        detail = f" — {row['reason']}" if row.get("reason") else ""
        print(f"[{number}/{len(names)}] {name}: {row['status']}{detail}", flush=True)
    print(
        f"Prepared {dict(Counter(d['status'] for d in manifest['datasets'].values()))}; no mining or fitting was run."
    )


def worker(path):
    # SIGTERM lets the existing Java _run handler terminate its own process group.
    def terminate(signum, frame):
        raise TimeoutError("Archive job terminated by supervisor")

    signal.signal(signal.SIGTERM, terminate)
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        normalized_mutual_info_score,
        homogeneity_score,
        adjusted_rand_score,
    )
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    from ctminer.features import describe, array_hash
    from ctminer.collections import mine
    from ctminer.memory import FeatureMemory

    job = read(path)
    out = Path(path).parent
    cfg = job["config"]
    started = perf_counter()

    def phase(name):
        if cfg.get("independent_budgets"):
            write(out / "phase.json", dict(phase=name))
            with (out / "phase_events.jsonl").open("a") as events:
                events.write(
                    json.dumps(
                        dict(phase=name, elapsed_seconds=perf_counter() - started)
                    )
                    + "\n"
                )

    try:
        phase("input")
        if digest(job["input"]) != job["input_sha256"]:
            raise ValueError("Prepared data hash changed")
        with np.load(job["input"], allow_pickle=False) as f:
            x, y = f["X"], f["y"]
        original_dim = x.shape[1]
        if job.get("length_percent", 100) != 100:
            x = x[:, : max(2, original_dim * job["length_percent"] // 100)]
        # One numerical input convention for Raw, PCA and every miner. This
        # matches the existing Java collection's legacy .9g transport exactly.
        quantize_start = perf_counter()
        x = np.asarray([[float(format(float(v), ".9g")) for v in row] for row in x])
        common_input_sha256 = array_hash(x)
        quantize_seconds = perf_counter() - quantize_start
        raw_dim = x.shape[1]
        engine = job["engine"]
        if engine not in ENGINES:
            raise ValueError("Only CT, Raw and PCA are included in this release")
        budgets = cfg["budgets"]
        threshold = cfg.get("feature_mode") == "threshold" and engine not in {
            "Raw",
            "PCA",
        }
        with threadpool_limits(limits=cfg["threads"]):
            patterns, mining = [], dict(seconds=0.0)
            if engine in {"Raw", "PCA"}:
                values = x
                if engine == "Raw":
                    with FeatureMemory(
                        out / "feature_memory.json",
                        interval=cfg["sample_interval"],
                        expect_children=False,
                    ):
                        values = x
            else:
                phase("mining_and_feature_construction")
                with FeatureMemory(
                    out / "feature_memory.json", interval=cfg["sample_interval"]
                ):
                    patterns, values, mining = mine(
                        list(x),
                        engine,
                        max(budgets),
                        out / "mining",
                        lengths=cfg["lengths"],
                        timeout=cfg["java_timeout"],
                        cap=cfg["pattern_cap"],
                        heap=cfg["java_heap"],
                        threads=cfg["threads"],
                        warmups=cfg["java_warmups"],
                        repeats=cfg["java_repeats"],
                        native_protocol=cfg.get("mining_protocol") == "native",
                        native_minsup=cfg.get("native_minsup", 2.0),
                        max_feature_cells=cfg.get("max_feature_cells", 100_000_000),
                        all_patterns=threshold,
                    )
                phase("feature_output")
                write(out / "mining.json", mining)
                write(
                    out / "dictionary.json",
                    dict(
                        engine=engine,
                        feature_value="support",
                        selection=mining["selection"],
                        patterns=[
                            dict(column=i, code=p, **mining["selected_details"][i])
                            for i, p in enumerate(patterns)
                        ],
                    ),
                )
                np.savez_compressed(out / "selected_features.npz", X=values, y=y)
            if job.get("study") == "length":
                budget = budgets[0]
                write(
                    out / "result.json",
                    dict(
                        status="complete",
                        job_fingerprint=job["fingerprint"],
                        elapsed_seconds=perf_counter() - started,
                        rows=[
                            dict(
                                dataset=job["dataset"],
                                engine=engine,
                                method=f"{engine}-all"
                                if threshold
                                else f"{engine}-K{budget}",
                                budget=budget,
                                feature_mode=cfg.get("feature_mode", "topk"),
                                minsup=cfg.get("native_minsup"),
                                status=(
                                    "no_features"
                                    if not patterns
                                    else "ok"
                                    if threshold or len(patterns) >= budget
                                    else "insufficient_features"
                                ),
                                Dim=len(patterns),
                                requested_Dim=budget,
                                original_length=original_dim,
                                sequence_length=raw_dim,
                                length_percent=job["length_percent"],
                                n_samples=len(y),
                                total_points=int(x.size),
                                mining_seconds=mining["seconds"],
                                mining_invocation_seconds=mining.get(
                                    "invocation_seconds", mining["seconds"]
                                ),
                                native_mining_seconds=mining.get(
                                    "native_mining_seconds"
                                ),
                                feature_transform_selection_seconds=mining.get(
                                    "feature_transform_selection_seconds"
                                ),
                                eligible_union_patterns=mining.get(
                                    "eligible_union_patterns"
                                ),
                                strong_rule_occurrences=mining.get(
                                    "strong_rule_occurrences"
                                ),
                                study="length",
                                scope="fixed sample count, nested time prefixes; mining and features only",
                            )
                        ],
                    ),
                )
                phase("done")
                return
            pca = None
            if engine == "PCA":
                phase("pca_fit_transform")
                with FeatureMemory(
                    out / "feature_memory.json",
                    interval=cfg["sample_interval"],
                    expect_children=False,
                ):
                    start = perf_counter()
                    pca_input = (
                        StandardScaler().fit_transform(values)
                        if cfg["scaling"] == "standard"
                        else values
                    )
                    pca = PCA(
                        n_components=min(max(budgets), len(x) - 1, raw_dim),
                        svd_solver="randomized",
                        random_state=0,
                    )
                    values = pca.fit_transform(pca_input)
                    pca_seconds = perf_counter() - start
                mining = dict(
                    seconds=pca_seconds,
                    explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
                )
                write(out / "pca.json", mining)
                np.savez_compressed(
                    out / "selected_features.npz",
                    X=values,
                    y=y,
                    components=pca.components_,
                    mean=pca.mean_,
                )
            rows = []
            common_row = dict(
                study=job.get("study", "quality"),
                original_length=original_dim,
                sequence_length=raw_dim,
                length_percent=job.get("length_percent", 100),
                total_points=int(x.size),
                n_samples=len(y),
                minconf=job.get("minconf"),
                maxsta=job.get("maxsta"),
                feature_mode=cfg.get("feature_mode", "topk"),
                minsup=cfg.get("native_minsup"),
                mining_invocation_seconds=mining.get(
                    "invocation_seconds", mining["seconds"]
                ),
                algorithm_seconds=mining["seconds"]
                if engine in {"Raw", "PCA"}
                else mining.get("native_mining_seconds", mining["seconds"]),
                native_mining_seconds=mining.get("native_mining_seconds"),
                feature_transform_selection_seconds=mining.get(
                    "feature_transform_selection_seconds"
                ),
                eligible_union_patterns=mining.get("eligible_union_patterns"),
            )
            phase("feature_output" if cfg.get("features_only") else "feature_scaling")
            for budget in [0] if engine == "Raw" else budgets:
                method = (
                    "Raw"
                    if engine == "Raw"
                    else f"{engine}-all"
                    if threshold
                    else f"{engine}-K{budget}"
                )
                if threshold and values.shape[1] == 0:
                    rows.append(
                        dict(
                            dataset=job["dataset"],
                            method=method,
                            engine=engine,
                            budget=budget,
                            status="no_features",
                            Dim=0,
                            reason="No native outputs qualify; no Raw fallback",
                        )
                    )
                    continue
                if budget > values.shape[1]:
                    rows.append(
                        dict(
                            dataset=job["dataset"],
                            method=method,
                            engine=engine,
                            budget=budget,
                            status="insufficient_features",
                            Dim=values.shape[1],
                            requested_Dim=budget,
                            reason="No padding, duplication or Raw replacement",
                        )
                    )
                    continue
                a = values if engine == "Raw" or threshold else values[:, :budget]
                info = (
                    describe(patterns if threshold else patterns[:budget], engine, a)
                    if patterns
                    else dict(matrix_sha256=array_hash(a), column_sha256=[])
                )
                if cfg.get("features_only"):
                    rows.append(
                        dict(
                            dataset=job["dataset"],
                            method=method,
                            engine=engine,
                            budget=budget,
                            status="ok",
                            Dim=a.shape[1],
                            raw_Dim=raw_dim,
                            diagnostics=info,
                            mining_seconds=mining["seconds"],
                            scope="features_only; no clustering",
                        )
                    )
                    continue
                prep = perf_counter()
                # PCA preserves its variance ordering; whitening is not applied.
                a = (
                    StandardScaler().fit_transform(a)
                    if cfg["scaling"] == "standard" and engine != "PCA"
                    else a
                )
                scaling_s = perf_counter() - prep
                assigned = []
                for seed in cfg["seeds"]:
                    phase(f"clustering_seed_{seed}")
                    estimator = KMeans(
                        n_clusters=len(set(y)),
                        n_init=cfg["n_init"],
                        random_state=seed,
                        algorithm="lloyd",
                    )
                    start = perf_counter()
                    pred = estimator.fit_predict(a)
                    fit_s = perf_counter() - start
                    assigned.append(pred)
                    rows.append(
                        dict(
                            dataset=job["dataset"],
                            method=method,
                            engine=engine,
                            budget=budget,
                            status="ok",
                            seed=seed,
                            Dim=a.shape[1],
                            raw_Dim=raw_dim,
                            implementation_language=LANGUAGES[engine],
                            input_quantization_seconds=quantize_seconds,
                            feature_mode=cfg.get("feature_mode", "topk"),
                            minsup=cfg.get("native_minsup"),
                            native_mining_seconds=mining.get("native_mining_seconds"),
                            feature_transform_selection_seconds=mining.get(
                                "feature_transform_selection_seconds"
                            ),
                            eligible_union_patterns=mining.get(
                                "eligible_union_patterns"
                            ),
                            strong_rule_occurrences=mining.get(
                                "strong_rule_occurrences"
                            ),
                            common_input_sha256=common_input_sha256,
                            dimension_ratio=a.shape[1] / raw_dim,
                            n_samples=len(y),
                            n_classes=len(set(y)),
                            NMI=float(
                                normalized_mutual_info_score(
                                    y, pred, average_method="geometric"
                                )
                            ),
                            h=float(homogeneity_score(y, pred)),
                            ARI=float(adjusted_rand_score(y, pred)),
                            actual_clusters=len(set(pred)),
                            mining_shared_seconds=mining["seconds"],
                            mining_invocation_seconds=mining.get(
                                "invocation_seconds", mining["seconds"]
                            ),
                            mining_warmups=cfg["java_warmups"] if patterns else 0,
                            mining_repeats=cfg["java_repeats"] if patterns else 1,
                            scaling_seconds=scaling_s,
                            clustering_seconds=fit_s,
                            pipeline_shared_seconds=quantize_seconds
                            + mining["seconds"]
                            + scaling_s
                            + fit_s,
                            timing_scope=(
                                "independent budget: fresh process and full mining/PCA fit; mining shared only across clustering seeds"
                                if cfg.get("independent_budgets")
                                else "maximum-budget discovery shared across budgets; invocation including warmups/transport separate"
                            ),
                            diagnostics=info,
                        )
                    )
                np.savez_compressed(
                    out / f"clusters_K{budget}.npz",
                    labels=y,
                    seeds=cfg["seeds"],
                    clusters=np.asarray(assigned),
                )
            rows = [dict(common_row, **row) for row in rows]
            phase("result_output")
            write(
                out / "result.json",
                dict(
                    status="complete",
                    job_fingerprint=job["fingerprint"],
                    elapsed_seconds=perf_counter() - started,
                    rows=rows,
                ),
            )
            phase("done")
    except Exception as exc:
        phase("failed")
        write(
            out / "result.json",
            dict(
                status="failed",
                job_fingerprint=job["fingerprint"],
                reason=str(exc),
                traceback=traceback.format_exc(),
                elapsed_seconds=perf_counter() - started,
            ),
        )
        raise

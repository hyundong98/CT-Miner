"""Check the complete feature-to-clustering path on bounded inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ctminer.collections import mine_hash as hash_mine
from ctminer.engines import ct_counts
from ctminer.native import mine
from ctminer.paths import sha256
from scripts.build import verify_build, verify_java

METHODS = ["CT", "Raw", "PCA"]
DATASETS = ["Car", "Beef", "ElectricDevices", "MoteStrain", "PigCVP", "Wafer"]


def fixtures(prepared, names):
    if prepared is None:
        rng = np.random.default_rng(17)
        x = np.array([rng.permutation(48) for _ in range(24)], dtype=float)
        yield "synthetic", x, np.array([i % 3 for i in range(len(x))])
        # Ties, constant sequences and signed zero exercise strict-window exclusion.
        tied = np.array([[0, 0, 1, 2, 1, 3, 4, 5],
                         [1, 2, 3, 1, 2, 4, 3, 5],
                         [-0., 0., -1, 2, -1, 3, 2, 4],
                         [3, 3, 3, 3, 3, 3, 3, 3]], dtype=float)
        yield "ties", tied, np.array([0, 1, 0, 1])
        return
    manifest = json.loads((prepared / "manifest.json").read_text())
    for name in names:
        entry = manifest["datasets"][name]
        path = prepared / entry["npz"]
        if entry["status"] != "ready" or sha256(path) != entry["sha256"]:
            raise ValueError(f"Prepared data changed: {name}")
        with np.load(path, allow_pickle=False) as data:
            # A small, reproducible input for validation, not a paper measurement.
            indices = np.linspace(
                0, len(data["y"]) - 1, min(24, len(data["y"])), dtype=int
            )
            yield name, data["X"][indices, :48], data["y"][indices]


def matrix_hash(values):
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype="<f8").tobytes()
    ).hexdigest()


def check_dataset(name, x, y, work, heap):
    results = []
    for percent in (50, 100):
        samples = x[:, : max(2, x.shape[1] * percent // 100)]
        samples = np.array(
            [[float(format(float(v), ".9g")) for v in row] for row in samples]
        )
        for method in METHODS:
            patterns = []
            if method == "Raw":
                values = samples
            elif method == "PCA":
                values = PCA(
                    n_components=3, svd_solver="randomized", random_state=0
                ).fit_transform(StandardScaler().fit_transform(samples))
            else:
                patterns, values, _ = mine(
                    samples,
                    method,
                    3,
                    work / f"{percent}/{method}",
                    lengths=range(2, 9),
                    minsup=2,
                    heap=heap,
                    timeout=120,
                )
            if not np.isfinite(values).all() or values.shape[0] != len(y):
                raise AssertionError(f"Invalid features: {name}/{method}")
            entry = dict(
                dataset=name,
                method=method,
                prefix_percent=percent,
                Dim=values.shape[1],
                patterns=patterns,
                matrix_sha256=matrix_hash(values),
            )
            if values.shape[1] == 0:
                entry["status"] = "no_features"
            else:
                scaled = (
                    values
                    if method == "PCA"
                    else StandardScaler().fit_transform(values)
                )
                pred = KMeans(
                    n_clusters=len(set(y)), n_init=10, random_state=0, algorithm="lloyd"
                ).fit_predict(scaled)
                entry.update(
                    status="ok",
                    NMI=float(
                        normalized_mutual_info_score(
                            y, pred, average_method="geometric"
                        )
                    ),
                    h=float(homogeneity_score(y, pred)),
                    ARI=float(adjusted_rand_score(y, pred)),
                )
            results.append(entry)
        # Include threshold output and sparse length selections, not only top-K.
        for lengths in (list(range(2, 9)), [3, 6, 8]):
            rows = [ct_counts(row, lengths, ties="drop_windows")[0] for row in samples]
            eligible = {p for row in rows for p, counts in row.items() if counts[0] >= 2}
            ranked = sorted(eligible, key=lambda p: (
                -sum(row.get(p, [0.0])[0] for row in rows), len(p), p))
            for budget in (3, 0):
                expected = ranked[:budget] if budget else ranked
                expected_values = np.asarray([
                    [row.get(p, [0.0])[0] for p in expected] for row in rows
                ], dtype=float).reshape(len(samples), len(expected))
                patterns, values, stats = mine(
                    samples, "CT", budget,
                    work / f"{percent}/reference-{lengths[0]}-{len(lengths)}-{budget}",
                    lengths=lengths, minsup=2, heap=heap, timeout=120)
                if patterns != expected or not np.array_equal(values, expected_values):
                    raise AssertionError(f"CT/Python reference mismatch: {name}/{percent}/{budget}")
                if stats["scores"] != expected_values.sum(axis=0).tolist():
                    raise AssertionError("CT support mismatch")
                results.append(dict(dataset=name, method="CT/Python-reference",
                                    prefix_percent=percent, budget=budget,
                                    lengths=lengths, status="matched", Dim=len(patterns),
                                    matrix_sha256=matrix_hash(values)))
        a = hash_mine(
            samples,
            "CT",
            3,
            work / f"{percent}/CT",
            lengths=range(2, 17),
            heap=heap,
            stop_early=False,
            allow_long_ct=True,
            timeout=120,
        )
        b = hash_mine(
            samples,
            "CT-Hash",
            3,
            work / f"{percent}/CT-Hash",
            lengths=range(2, 17),
            heap=heap,
            stop_early=False,
            allow_long_ct=True,
            timeout=120,
        )
        if (
            a[0] != b[0]
            or not np.array_equal(a[1], b[1])
            or a[2]["scores"] != b[2]["scores"]
        ):
            raise AssertionError(f"CT/CT-Hash mismatch: {name}/{percent}")
        results.append(
            dict(
                dataset=name,
                method="CT/CT-Hash",
                prefix_percent=percent,
                status="matched",
                Dim=len(a[0]),
                matrix_sha256=matrix_hash(a[1]),
            )
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared", type=Path, help="Use small excerpts from prepared real datasets"
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument(
        "--output", type=Path, default=Path("results/validation/smoke.json")
    )
    parser.add_argument("--java-heap", default="1g")
    args = parser.parse_args()
    verify_java()
    verify_build()
    if args.output.exists():
        parser.error("Output already exists; choose a new report path")
    report = dict(
        scope="Functional checks on bounded inputs; not a full performance reproduction",
        cases=[],
    )
    with (
        tempfile.TemporaryDirectory(prefix="ctminer-check-") as directory,
        threadpool_limits(limits=1),
    ):
        for name, x, y in fixtures(args.prepared, args.datasets):
            report["cases"].extend(
                check_dataset(name, x, y, Path(directory) / name, args.java_heap)
            )
            print(f"{name}: clustering, CT/Python reference and CT/CT-Hash checks passed", flush=True)
    report["status"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

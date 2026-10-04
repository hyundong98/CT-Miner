"""Pattern and feature-matrix summaries."""

from pathlib import Path
from time import perf_counter
import hashlib
import numpy as np

from ctminer.io import collection_batch
from ctminer.engines import CLASSES, _java_ready, _run


def array_hash(x):
    return hashlib.sha256(
        str(x.shape).encode() + np.ascontiguousarray(x, dtype="<f8").tobytes()
    ).hexdigest()


def pd(p):
    stack, result = [], []
    for i, value in enumerate(p):
        while stack and p[stack[-1]] > value:
            stack.pop()
        result.append(i - stack[-1] if stack else 0)
        stack.append(i)
    return tuple(result)


def describe(patterns, engine, matrix):
    def is_monotone(p):
        if engine in {
            "CT",
            "CT-Naive",
            "CT-Hash",
            "CT-Pairwise",
            "CT-Compact",
            "CT-ID",
        }:
            return p == (0,) * len(p) or p == (0,) + (1,) * (len(p) - 1)
        return p == tuple(range(1, len(p) + 1)) or p == tuple(range(len(p), 0, -1))

    columns = [array_hash(matrix[:, i : i + 1]) for i in range(matrix.shape[1])]
    return dict(
        matrix_sha256=array_hash(matrix),
        column_sha256=columns,
        unique_feature_columns=len(set(columns)),
        zero_row_fraction=float(np.mean(np.all(matrix == 0, axis=1))),
        constant_columns=int(np.sum(np.ptp(matrix, axis=0) == 0)),
        monotone_patterns=sum(is_monotone(p) for p in patterns),
        monotone_fraction=sum(is_monotone(p) for p in patterns) / len(patterns),
        pattern_code_units=sum(map(len, patterns)),
        semantic_CT_codes=[
            list(
                p
                if engine
                in {"CT", "CT-Naive", "CT-Hash", "CT-Pairwise", "CT-Compact", "CT-ID"}
                else pd(p)
            )
            for p in patterns
        ],
    )


def mine(samples, engine, budget, work, *, lengths=None, timeout=1800, cap=2000000):
    """Reuse top-B columns across budgets, retaining at most B per mined length.

    Global frequency: stop when the next length cannot beat the current top B.
    Unweighted support is prefix-antimonotone, so maximum support cannot
    grow at the next length.
    This is exact for this study's univariate strict contiguous occurrences.
    """
    _java_ready()
    if engine != "CT":
        raise ValueError("Unknown engine")
    decay = 0.0
    requested = list(range(2, budget + 2)) if lengths is None else sorted(set(lengths))
    if not requested or min(requested) < 2 or max(requested) > 63:
        raise ValueError("This adapter uses lengths 2..63; use ctminer.native for longer patterns")
    start = perf_counter()
    records = []
    kept = []
    scores = {}
    values = np.empty((len(samples), 0))
    stop_reason = "all_requested_lengths"
    for length in requested:
        if length > max(map(len, samples)):
            stop_reason = "sequence_length_bound"
            break
        rows, stat = collection_batch(
            engine,
            samples,
            [length],
            1,
            decay,
            "drop_windows",
            timeout,
            Path(work) / f"length_{length}",
            CLASSES,
            _run,
            selection=dict(
                budget=budget,
                budget_mode="global",
                minsup=0.0,
                max_observed_patterns=cap,
            ),
        )
        selected = stat["corpus_selection"]
        patterns = [tuple(p) for p in selected["patterns"]]
        stage_scores = dict(zip(patterns, selected["scores"]))
        stage = np.asarray(
            [[float(r[p][0]) if p in r else 0.0 for p in patterns] for r in rows],
            dtype=np.float64,
        ).reshape(len(samples), len(patterns))
        previous = len(kept)
        all_patterns = kept + patterns
        scores.update(stage_scores)
        indices = sorted(
            range(len(all_patterns)),
            key=lambda i: (
                -scores[all_patterns[i]],
                len(all_patterns[i]),
                all_patterns[i],
            ),
        )[:budget]
        # Allocate only selected columns, not a corpus-wide candidate matrix.
        output = np.empty((len(samples), len(indices)))
        for j, i in enumerate(indices):
            output[:, j] = values[:, i] if i < previous else stage[:, i - previous]
        kept = [all_patterns[i] for i in indices]
        values = output
        scores = {p: scores[p] for p in kept}
        maximum = max(stage_scores.values(), default=0.0)
        cutoff = scores[kept[-1]] if len(kept) == budget else None
        records.append(
            dict(
                length=length,
                maximum_support=maximum,
                cutoff=cutoff,
                elapsed_seconds=perf_counter() - start,
                stats=stat,
            )
        )
        # This also certifies all later lengths in an explicit ascending grid.
        if not patterns or (cutoff is not None and maximum < cutoff * (1 - 1e-12)):
            stop_reason = "support_upper_bound"
            break
    return (
        kept,
        values,
        dict(
            engine=engine,
            decay=decay,
            support_definition="sum of per-sequence end-weighted occurrences"
            if decay
            else "sum of per-sequence occurrence counts",
            scores=[scores[p] for p in kept],
            steps=records,
            stop_reason=stop_reason,
            requested_budget=budget,
            seconds=perf_counter() - start,
            length_contract="global top-k; shorter-length tie break; bound B+1"
            if lengths is None
            else "fixed allowed lengths",
            scope="rebuild existing Java miner per length/collection; reuse selected columns across feature budgets",
        ),
    )

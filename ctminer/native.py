"""Per-series threshold mining and collection feature extraction."""

from pathlib import Path
import json
import math
import re
import struct
import tempfile
from time import perf_counter

import numpy as np

from ctminer.io import read_exact, unpack
from ctminer.engines import CLASSES, _java_ready, _run

ENGINES = {"CT", "CT-ID"}
VERSION = "native_objectives_v1"


def mine(
    samples,
    engine,
    budget,
    work,
    *,
    lengths=None,
    minsup=2.0,
    max_feature_cells=100_000_000,
    timeout=7200.0,
    cap=2_000_000,
    heap="16g",
    threads=1,
):
    _java_ready()
    unbounded_lengths = lengths is None
    lengths = (
        list(range(2, max(2, max(map(len, samples))) + 1))
        if lengths is None and len(samples)
        else sorted(set(map(int, lengths or [])))
    )
    if (
        engine not in ENGINES
        or not len(samples)
        or not lengths
        or min(lengths) < 2
        or max(lengths) > 1_000_000
    ):
        raise ValueError(
            "Native mining requires a known engine, samples, and lengths 2..1000000"
        )
    if (
        not all(math.isfinite(v) for v in (minsup, timeout))
        or minsup <= 0
        or timeout <= 0
    ):
        raise ValueError("Invalid native thresholds")
    if (
        not 0 <= budget <= 2**31 - 1
        or not 1 <= cap <= 2**31 - 1
        or min(max_feature_cells, threads) < 1
    ):
        raise ValueError("Invalid native budget/cap/resource setting")
    if not re.fullmatch(r"[1-9][0-9]*[kKmMgG]", heap):
        raise ValueError("Invalid Java heap")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    with tempfile.TemporaryDirectory(prefix="native-", dir=work) as tmp:
        inp, out = Path(tmp) / "input.bin", Path(tmp) / "output.bin"
        with inp.open("wb") as stream:
            stream.write(struct.pack(">4i", 0x434E4931, 1, len(samples), len(lengths)))
            stream.write(struct.pack(f">{len(lengths)}i", *lengths))
            stream.write(
                struct.pack(
                    ">2i4d2q",
                    budget,
                    cap,
                    minsup,
                    0.0,  # Reserved fields in the original binary protocol.
                    0.0,
                    0.0,
                    1,
                    max_feature_cells,
                )
            )
            for sample in samples:
                x = np.asarray(
                    [float(format(float(v), ".9g")) for v in sample], dtype=">f8"
                )
                if not len(x) or not np.isfinite(x).all():
                    raise ValueError("Invalid native input")
                stream.write(struct.pack(">i", len(x)))
                stream.write(x.tobytes())
        wall = _run(
            [
                "java",
                f"-Xmx{heap}",
                f"-XX:ActiveProcessorCount={threads}",
                "-cp",
                str(CLASSES),
                "NativePatternCollection",
                engine,
                str(inp),
                str(out),
                str(work),
            ],
            timeout,
            work / "java.log",
        )
        with out.open("rb") as stream:
            magic, version, ns, dim = unpack(stream, ">4i")
            if (
                (magic, version, ns) != (0x434E4F31, 1, len(samples))
                or not 0 <= dim <= cap
                or dim * ns > max_feature_cells
            ):
                raise ValueError("Invalid native output header")
            mining_s, feature_s, candidates, rules, eligible, visited = unpack(
                stream, ">2d2q2i"
            )
            if (
                not np.isfinite([mining_s, feature_s]).all()
                or min(mining_s, feature_s, candidates, rules, eligible, visited) < 0
                or eligible < dim
            ):
                raise ValueError("Invalid native diagnostics")
            patterns = []
            details = []
            matrix = np.empty((ns, dim), dtype=np.float64)
            for j in range(dim):
                (length,) = unpack(stream, ">i")
                if length not in lengths:
                    raise ValueError("Native output outside requested lengths")
                code = unpack(stream, f">{length}i")
                support, qualifying = unpack(stream, ">di")
                if (
                    not math.isfinite(support)
                    or support <= 0
                    or not 1 <= qualifying <= ns
                ):
                    raise ValueError("Invalid native support metadata")
                patterns.append(code)
                details.append(
                    dict(
                        support=support,
                        qualifying_series=qualifying,
                    )
                )
                matrix[:, j] = np.frombuffer(read_exact(stream, ns * 8), dtype=">f8")
            if (
                stream.read(1)
                or len(set(patterns)) != dim
                or not np.isfinite(matrix).all()
                or np.any(matrix < 0)
            ):
                raise ValueError("Invalid native feature matrix")
    is_ct = engine in {"CT", "CT-ID"}
    ct_meta = json.loads((work / "ct_native_stats.json").read_text()) if is_ct else {}
    if is_ct and ct_meta.get("version") != "ct_native_strict_id_v2":
        raise ValueError("Missing corrected strict CT diagnostics; rebuild Java")
    stats = dict(
        engine=engine,
        framework=VERSION,
        implementation_language="Java",
        implementation="CT suffix trie",
        seconds=mining_s + feature_s,
        native_mining_seconds=mining_s,
        feature_transform_selection_seconds=feature_s,
        invocation_seconds=perf_counter() - start,
        jvm_wall_seconds_including_warmups=wall,
        selected_details=details,
        scores=[d["support"] for d in details],
        requested_budget=budget,
        feature_mode="threshold" if budget == 0 else "topk",
        eligible_union_patterns=eligible,
        candidates=candidates,
        strong_rule_occurrences=rules,
        requested_lengths=lengths,
        output_length_policy="all lengths through input length"
        if unbounded_lengths
        else "explicit allowed lengths",
        visited_maximum=visited,
        minsup=minsup,
        max_feature_cells=max_feature_cells,
        mining_support_scope="per-series absolute support; union of native outputs",
        feature_value="occurrence count",
        feature_transform="reuse exact strict trie counts in every series, including below-threshold occurrences",
        selection="native eligible union; total support/length/code",
        pattern_objective="frequent CT on strictly distinct-valued windows",
        stop_early=budget > 0,
        topk_feedback_into_mining=budget > 0,
        warmups=0,
        tie_policy="strict distinct-valued windows",
        steps=ct_meta.get("steps", []),
        stop_reason=ct_meta.get("stop_reason", "native_threshold_mining_complete"),
        scope="native predicates, explicit output length policy; no labels in mining; not a claim of exact paper table reproduction",
    )
    if is_ct:
        stats.update(
            ct_execution_version=ct_meta["version"],
            implementation="CT corpus-wide strict trie counts; exact per-series eligibility and safe top-K cutoff",
            eligible_patterns_visited=eligible,
            eligible_union_complete=ct_meta["eligible_union_complete"],
            eligible_union_patterns=eligible
            if ct_meta["eligible_union_complete"]
            else None,
            materialized_patterns=ct_meta["materialized_patterns"],
            materialized_code_units=ct_meta["materialized_code_units"],
            prefix_ids=ct_meta["prefix_ids"],
            timing_partition="native mining includes index/count/eligibility/selection/final code materialization; transform is sparse count replay and retained-column allocation; invocation also includes transport/output",
        )
    (work / "native.json").write_text(
        json.dumps(stats, indent=2, allow_nan=False) + "\n"
    )
    return patterns, matrix, stats

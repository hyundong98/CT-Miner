"""CT feature construction for the public experiment runners."""

from pathlib import Path
import json
import re
import statistics
import struct
import tempfile
from time import perf_counter

import numpy as np

from ctminer.io import unpack, read_exact
from ctminer.engines import CLASSES, _java_ready, _run

from ctminer.native import mine as native_mine

ENGINES = LONG_CT_ENGINES = {"CT-ID", "CT-Hash"}
MAX_CT_LENGTH = 1000000
POLICIES = {
    "CT-ID": "same CH trie; level-synchronous radix ranks and direct addressing; no ancestor table or post-build ID hashing",
    "CT-Hash": "independent-window PD encoding and hash aggregation",
}


def mine(samples, engine, budget, work, *, lengths=None, timeout=1800,
         cap=2000000, heap="2g", threads=1, warmups=0, repeats=1,
         native_protocol=True, native_minsup=2.0, max_feature_cells=100000000,
         all_patterns=False):
    if engine not in {"CT", "CT-ID"}:
        raise ValueError("The clustering adapter accepts CT/CT-ID; use mine_hash for CT/CT-Hash comparisons")
    if not native_protocol or warmups or repeats != 1:
        raise ValueError("Use the native CT protocol with one fresh JVM per feature construction")
    return native_mine(samples, engine, 0 if all_patterns else budget, work,
                       lengths=lengths, minsup=native_minsup,
                       max_feature_cells=max_feature_cells, timeout=timeout,
                       cap=cap, heap=heap, threads=threads)


def read_output(path, samples, *, max_length=63, max_stages=62):
    with Path(path).open("rb") as stream:
        magic, version, actual, dimensions = unpack(stream, ">4i")
        if (magic, version, actual) != (0x43414F31, 2, samples) or dimensions < 0:
            raise ValueError("Invalid Java archive output header")
        input_seconds, warmups, repeats = unpack(stream, ">d2i")
        if warmups < 0 or repeats < 1:
            raise ValueError("Invalid Java repetition metadata")
        runs = []
        for _ in range(repeats):
            seconds, gc_ms = unpack(stream, ">dq")
            (size,) = unpack(stream, ">H")
            # Java writeUTF: these stop reason constants are ASCII.
            stop = read_exact(stream, size).decode("ascii")
            (count,) = unpack(stream, ">i")
            if (
                count < 0
                or count > max_stages
                or not np.isfinite(seconds)
                or seconds < 0
                or gc_ms < 0
            ):
                raise ValueError("Invalid Java stage metadata")
            stages = []
            for _ in range(count):
                length, observed, eligible, nodes, candidates = unpack(stream, ">3i2q")
                if not 2 <= length <= max_length:
                    raise ValueError("Invalid Java stage length")
                values = unpack(stream, ">9d")
                if (
                    any(v < 0 for v in (observed, eligible, nodes, candidates))
                    or eligible > observed
                ):
                    raise ValueError("Invalid Java pattern counts")
                if any(
                    not np.isfinite(v) or v < 0 for i, v in enumerate(values) if i != 1
                ):
                    raise ValueError("Invalid Java timings/support")
                keys = (
                    "maximum_unfiltered_support",
                    "cutoff",
                    "index_seconds",
                    "count_seconds",
                    "author_setup_parse_seconds",
                    "aggregation_spool_seconds",
                    "selection_seconds",
                    "replay_seconds",
                    "seconds",
                )
                stage = dict(zip(keys, values))
                if np.isnan(stage["cutoff"]):
                    stage["cutoff"] = None
                stage.update(
                    length=length,
                    observed_patterns=observed,
                    eligible_patterns=eligible,
                    nodes=nodes,
                    candidates=candidates,
                )
                stages.append(stage)
            runs.append(
                dict(
                    seconds=seconds,
                    gc_seconds=gc_ms / 1000,
                    stop_reason=stop,
                    steps=stages,
                )
            )
        patterns, details = [], []
        matrix = np.empty((samples, dimensions), dtype=np.float64)
        for j in range(dimensions):
            (length,) = unpack(stream, ">i")
            if not 2 <= length <= max_length:
                raise ValueError("Invalid Java pattern length")
            patterns.append(unpack(stream, f">{length}i"))
            support, ngap, gap, gap2 = unpack(stream, ">dq2d")
            (stable_series,) = unpack(stream, ">i")
            if not 0 <= stable_series <= samples:
                raise ValueError("Invalid stable-series coverage")
            if not np.isfinite([support, gap, gap2]).all() or support <= 0 or ngap < 0:
                raise ValueError("Invalid Java pattern metadata")
            cv = (
                np.sqrt(max(0.0, gap2 / ngap - (gap / ngap) ** 2)) / (gap / ngap)
                if ngap and gap > 0
                else None
            )
            details.append(
                dict(
                    support=support,
                    stable_series=stable_series,
                    gap_count=ngap,
                    gap_sum=gap,
                    gap_square_sum=gap2,
                    gap_cv=None if cv is None else float(cv),
                )
            )
            matrix[:, j] = np.frombuffer(read_exact(stream, samples * 8), dtype=">f8")
        if (
            stream.read(1)
            or len(set(patterns)) != len(patterns)
            or not np.isfinite(matrix).all()
            or np.any(matrix < 0)
        ):
            raise ValueError("Invalid Java feature matrix or trailing output")
    return (
        patterns,
        matrix,
        dict(
            java_input_seconds=input_seconds,
            warmups=warmups,
            repeats=runs,
            selected_details=details,
            scores=[d["support"] for d in details],
            seconds=statistics.median(r["seconds"] for r in runs),
            steps=runs[0]["steps"],
            stop_reason=runs[0]["stop_reason"],
        ),
    )


def _hash_mine(
    samples,
    engine,
    budget,
    work,
    *,
    lengths,
    timeout=1800,
    cap=2000000,
    heap="2g",
    threads=1,
    warmups=0,
    repeats=1,
    all_patterns=False,
    stop_early=True,
    allow_long_ct=False,
):
    _java_ready()
    lengths = sorted(set(map(int, lengths)))
    if allow_long_ct and engine not in LONG_CT_ENGINES:
        raise ValueError("Long-length transport is restricted to CT-ID and CT-Hash")
    limit = MAX_CT_LENGTH if allow_long_ct else 63
    if (
        engine not in ENGINES
        or not len(samples)
        or not lengths
        or not 2 <= min(lengths) <= max(lengths) <= limit
    ):
        raise ValueError(
            f"Nonempty samples, known Java engine and lengths 2..{limit} required"
        )
    if not 1 <= budget <= 2147483647 or not 1 <= cap <= 2147483647:
        raise ValueError("Invalid selection budget/cap")
    if (
        not re.fullmatch(r"[1-9][0-9]*[kKmMgG]", heap)
        or threads < 1
        or warmups < 0
        or repeats < 1
    ):
        raise ValueError("Invalid JVM heap/threads/repeats")
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("Invalid timeout")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    invocation = perf_counter()
    with tempfile.TemporaryDirectory(prefix="java-archive-", dir=work) as tmp:
        inp, out = Path(tmp) / "input.bin", Path(tmp) / "output.bin"
        tick = perf_counter()
        with inp.open("wb") as stream:
            stream.write(
                struct.pack(
                    ">4i",
                    0x43414A31,
                    3 if allow_long_ct else 2,
                    len(samples),
                    len(lengths),
                )
            )
            stream.write(struct.pack(f">{len(lengths)}i", *lengths))
            stream.write(struct.pack(">2idq", budget, cap, 0.5, 1))
            # Reserved policy fields retain the archive wire format.
            stream.write(struct.pack(">2iq", 0, 0, 2))
            for sample in samples:
                values = np.asarray(
                    [float(format(float(v), ".9g")) for v in sample], dtype=">f8"
                )
                if not len(values) or not np.isfinite(values).all():
                    raise ValueError("Invalid sample")
                stream.write(struct.pack(">i", len(values)))
                stream.write(values.tobytes())
        input_seconds = perf_counter() - tick
        command = [
            "java",
            f"-Xmx{heap}",
            f"-XX:ActiveProcessorCount={threads}",
            "-cp",
            str(CLASSES),
            "ArchiveJavaCollection",
            engine,
            str(CLASSES),
            str(inp),
            str(out),
            str(warmups),
            str(repeats),
            str(bool(all_patterns)).lower(),
            str(bool(stop_early)).lower(),
        ]
        wall = _run(command, timeout, work / "java.log")
        tick = perf_counter()
        patterns, matrix, stats = read_output(
            out, len(samples), max_length=max(lengths), max_stages=len(lengths)
        )
        parse_seconds = perf_counter() - tick
        if engine == "CT-ID":
            with (work / "java.log").open() as log:
                id_runs = [
                    json.loads(line[len("CT_ID_STATS ") :])
                    for line in log
                    if line.startswith("CT_ID_STATS ")
                ]
            if len(id_runs) != warmups + repeats or any(
                r["materialized_patterns"] != len(patterns) for r in id_runs
            ):
                raise ValueError("Missing/inconsistent CT-ID restoration diagnostics")
            stats["canonical_id_runs"] = id_runs[warmups:]
    stats.update(
        engine=engine,
        implementation_language="Java",
        implementation=POLICIES[engine],
        framework="ArchiveJavaCollection_v2",
        transport="binary_archive_v3_ct_input"
        if allow_long_ct
        else "binary_archive_v2",
        requested_lengths=lengths,
        allow_long_ct=allow_long_ct,
        requested_budget=budget,
        pattern_cap=cap,
        java_heap=heap,
        java_active_processors=threads,
        python_input_seconds=input_seconds,
        python_output_parse_seconds=parse_seconds,
        jvm_wall_seconds_including_warmups=wall,
        invocation_seconds=perf_counter() - invocation,
        repeat_output_parity=True if warmups + repeats > 1 else None,
        repeat_output_comparisons=warmups + repeats - 1,
        all_patterns=all_patterns,
        stop_early=stop_early,
        timing_scope="median measured Java dictionary+feature construction; JVM startup/input/output and Python transport reported separately",
        selection="total support descending, length ascending, lexicographic code",
        cutoff_metric="support",
        stopping_policy=(
            "disabled; visit every requested length within the input boundary"
            if not stop_early else "unfiltered support upper bound"
        ),
        state_policy=POLICIES[engine],
        memory_policy="common JVM heap; indexed miners retain per-sample state; CT-Hash stores no suffix index",
        candidate_counter_scope="CT-ID counts level records; CT-Hash has no index nodes",
        scope="exact top-k within requested lengths; strict distinct-valued windows; one dictionary per invocation, caller controls reuse",
    )
    return patterns, matrix, stats


def mine_hash(samples, engine, budget, work, **options):
    """Run the CT/CT-Hash comparison with the original integer-trie backend."""
    if engine not in {"CT", "CT-Hash"}:
        raise ValueError("Expected CT or CT-Hash")
    options.setdefault("allow_long_ct", True)
    options.setdefault("stop_early", False)
    # This is a Java protocol identifier, not the method name in saved results.
    backend = "CT-ID" if engine == "CT" else "CT-Hash"
    patterns, values, stats = _hash_mine(samples, backend, budget, work, **options)
    stats = {
        key: value.replace("CT-ID", "CT") if isinstance(value, str) else value
        for key, value in stats.items()
    }
    stats["engine"] = engine
    return patterns, values, stats

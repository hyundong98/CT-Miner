"""Versioned binary collection transport; no mining runs on import."""

from pathlib import Path
import struct
import tempfile
from time import perf_counter

import numpy as np


def read_exact(stream, size):
    value = stream.read(size)
    if len(value) != size:
        raise ValueError("Truncated Java collection output")
    return value


def unpack(stream, fmt):
    return struct.unpack(fmt, read_exact(stream, struct.calcsize(fmt)))


def read_output(path, samples, bins):
    maps, stats, patterns = [], [], []
    selection = None
    with Path(path).open("rb") as stream:
        magic, version, actual_samples, actual_bins = unpack(stream, ">4i")
        if (
            magic != 0x43544F32
            or version not in (1, 2)
            or (actual_samples, actual_bins) != (samples, bins)
        ):
            raise ValueError("Java collection header mismatch")
        while True:
            (tag,) = unpack(stream, ">B")
            if tag == 1:
                if version == 2 and selection is not None:
                    raise ValueError("Dictionary changed after corpus selection")
                i, n = unpack(stream, ">2i")
                if i != len(patterns) or n < 2:
                    raise ValueError("Invalid pattern definition")
                patterns.append(unpack(stream, f">{n}i"))
            elif tag == 4:
                if version != 2 or selection is not None or maps or bins != 1:
                    raise ValueError("Unexpected selection metadata")
                (selected,) = unpack(stream, ">i")
                if selected != len(patterns):
                    raise ValueError("Selected dictionary size mismatch")
                scores = unpack(stream, f">{selected}d")
                (nlengths,) = unpack(stream, ">i")
                if nlengths < 1 or not all(np.isfinite(v) and v >= 0 for v in scores):
                    raise ValueError("Invalid selection scores or lengths")
                by_length = {}
                for _ in range(nlengths):
                    length, observed, eligible, mass = unpack(stream, ">3id")
                    if (
                        length < 2
                        or str(length) in by_length
                        or not 0 <= eligible <= observed
                        or not np.isfinite(mass)
                        or mass < 0
                    ):
                        raise ValueError("Invalid corpus length metadata")
                    by_length[str(length)] = dict(
                        observed_patterns=observed,
                        eligible_after_selection_filter=eligible,
                        support_mass=mass,
                    )
                (
                    observed,
                    eligible,
                    source_records,
                    aggregation_s,
                    selection_s,
                    spool_bytes,
                ) = unpack(stream, ">2iq2dq")
                if (
                    not 0 <= selected <= eligible <= observed
                    or source_records < observed
                    or spool_bytes < 0
                    or not all(
                        np.isfinite(v) and v >= 0 for v in (aggregation_s, selection_s)
                    )
                    or sum(v["observed_patterns"] for v in by_length.values())
                    != observed
                    or sum(
                        v["eligible_after_selection_filter"] for v in by_length.values()
                    )
                    != eligible
                ):
                    raise ValueError("Invalid corpus selection metadata")
                selection = dict(
                    patterns=list(patterns),
                    scores=list(scores),
                    by_length=by_length,
                    observed_patterns=observed,
                    eligible_patterns=eligible,
                    candidate_sample_records=source_records,
                    aggregation_spool_seconds=aggregation_s,
                    selection_seconds=selection_s,
                    spool_bytes=spool_bytes,
                    backend="java_exact_corpus_frequency",
                    candidate_discovery="full_enumeration",
                    detailed_diagnostics_scope="selected_dictionary_only",
                )
            elif tag == 2:
                if version == 2 and selection is None:
                    raise ValueError("Missing corpus selection before sample rows")
                i, n = unpack(stream, ">2i")
                if i != len(maps) or n < 0:
                    raise ValueError("Invalid sample record")
                row = {}
                for _ in range(n):
                    (ident,) = unpack(stream, ">i")
                    if not 0 <= ident < len(patterns) or patterns[ident] in row:
                        raise ValueError("Invalid/duplicate feature ID")
                    value = np.array(unpack(stream, f">{bins}d"))
                    if not np.isfinite(value).all() or (value < 0).any():
                        raise ValueError("Invalid support")
                    row[patterns[ident]] = value
                index, count, setup, parse, nodes, candidates = unpack(stream, ">4d2i")
                if not all(
                    np.isfinite(v) and v >= 0 for v in (index, count, setup, parse)
                ):
                    raise ValueError("Invalid computation time")
                maps.append(row)
                stats.append(
                    dict(
                        index_seconds=index,
                        count_seconds=count,
                        class_setup_seconds=setup,
                        author_output_parse_seconds=parse,
                        nodes=nodes,
                        candidates=candidates,
                    )
                )
            elif tag == 3:
                read_s, write_s, internal_s, gc_ms = unpack(stream, ">3dq")
                if (
                    not all(
                        np.isfinite(v) and v >= 0 for v in (read_s, write_s, internal_s)
                    )
                    or gc_ms < 0
                ):
                    raise ValueError("Invalid collection timing metadata")
                if (
                    len(maps) != samples
                    or (version == 2 and selection is None)
                    or stream.read(1)
                ):
                    raise ValueError("Incomplete collection or trailing bytes")
                break
            else:
                raise ValueError("Unknown collection record")
    totals = {
        k: sum(r[k] for r in stats)
        for k in (
            "index_seconds",
            "count_seconds",
            "class_setup_seconds",
            "author_output_parse_seconds",
        )
    }
    for k in ("nodes", "candidates"):
        totals[k] = sum(r[k] for r in stats) if all(r[k] >= 0 for r in stats) else None
    return maps, {
        **totals,
        **({"corpus_selection": selection} if selection is not None else {}),
        "java_input_seconds": read_s,
        "java_output_seconds": write_s,
        "java_internal_seconds": internal_s,
        "gc_seconds": gc_ms / 1000,
        "output_patterns": sum(map(len, maps)),
        "unique_output_patterns": len(patterns),
    }


def collection_batch(
    engine,
    samples,
    lengths,
    bins,
    decay,
    ties,
    timeout,
    work,
    classes,
    run,
    wanted=None,
    lookup="direct",
    selection=None,
    benchmark_copies=False,
):
    if engine != "CT":
        raise ValueError("Only CT is included in this release")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    lengths = sorted(set(map(int, lengths)))
    if not samples or not lengths or lengths[0] < 2 or bins < 1:
        raise ValueError("Nonempty samples/lengths and positive bins required")
    if lookup not in {"direct", "enumerate"} or ties not in {"native", "drop_windows"}:
        raise ValueError("Invalid lookup or tie policy")
    if not np.isfinite(decay) or not 0 <= decay <= 100:
        raise ValueError("Invalid decay")
    if not isinstance(benchmark_copies, bool) or (benchmark_copies and engine != "CT"):
        raise ValueError("Copy reconstruction is a CT-only boolean benchmark option")
    mode = None
    groups = None
    if selection is not None:
        if engine != "CT" or bins != 1 or wanted is not None:
            raise ValueError(
                "Java corpus selection requires CT scalar discovery"
            )
        mode = {"global": 0, "per_length": 1, "all_observed": 2, "grouped": 3}[
            selection["budget_mode"]
        ]
        budget, cap, minsup = (
            selection["budget"],
            selection["max_observed_patterns"],
            selection["minsup"],
        )
        if (
            not (1 <= budget <= 2147483647 and 1 <= cap <= 2147483647)
            or not np.isfinite(minsup)
            or minsup < 0
        ):
            raise ValueError("Invalid Java corpus selection budget/cap/minsup")
        if mode == 3:
            groups = selection["groups"]
            covered = []
            for group in groups:
                if not group["lengths"] or not 1 <= group["budget"] <= 2147483647:
                    raise ValueError("Invalid grouped budget/lengths")
                covered.extend(group["lengths"])
            if not groups or sorted(covered) != lengths:
                raise ValueError(
                    "Selection groups must partition the requested lengths exactly"
                )
            if sum(group["budget"] for group in groups) > budget:
                raise ValueError("Group budgets exceed the total selection budget")
    with tempfile.TemporaryDirectory(prefix="collection-", dir=work) as tmp:
        inp, out = Path(tmp) / "input.bin", Path(tmp) / "output.bin"
        tick = perf_counter()
        with inp.open("wb") as stream:
            stream.write(
                struct.pack(
                    ">4i",
                    0x43544932,
                    3 if groups is not None else (2 if selection is not None else 1),
                    len(samples),
                    len(lengths),
                )
            )
            stream.write(struct.pack(f">{len(lengths)}i", *lengths))
            stream.write(
                struct.pack(
                    ">id??i",
                    bins,
                    decay,
                    ties == "drop_windows",
                    lookup == "direct",
                    -1 if wanted is None else len(wanted),
                )
            )
            if wanted is not None:
                for p in sorted(wanted):
                    stream.write(struct.pack(f">{len(p) + 1}i", len(p), *p))
            if selection is not None:
                stream.write(struct.pack(">2idi", budget, mode, minsup, cap))
            if groups is not None:
                stream.write(struct.pack(">i", len(groups)))
                for group in groups:
                    stream.write(
                        struct.pack(">2i", group["budget"], len(group["lengths"]))
                    )
                    stream.write(
                        struct.pack(f">{len(group['lengths'])}i", *group["lengths"])
                    )
            for x in samples:
                # Match legacy %.9g text transport numerically as doubles. Nine
                # digits preserve the shared float32 ordering, ties and values
                # seen by OP subtraction/comparison code; this avoids silently
                # changing author behavior while removing per-sample files.
                values = np.asarray(
                    [float(format(float(v), ".9g")) for v in x], dtype=">f8"
                )
                stream.write(struct.pack(">i", len(values)))
                stream.write(values.tobytes())
        input_s = perf_counter() - tick
        command = (
            ["CTTopKCollection", str(inp), str(out)]
            if selection is not None
            else ["JavaCollection", engine, str(classes), str(inp), str(out)]
        )
        copy_options = ["-Dctpm.benchmarkCopies=true"] if benchmark_copies else []
        wall = run(
            ["java", "-Xmx2g", *copy_options, "-cp", str(classes), *command],
            timeout,
            work / "collection.log",
        )
        tick = perf_counter()
        maps, stats = read_output(out, len(samples), bins)
        parse_s = perf_counter() - tick
        if selection is not None and "corpus_selection" not in stats:
            raise ValueError("Java did not return a selected corpus dictionary")
        if selection is not None:
            stats["corpus_selection"]["engine"] = engine
            stats["corpus_selection"]["groups"] = groups
    return maps, {
        **stats,
        "backend": "java_Cole_Hariharan_trie"
        if engine == "CT"
        else f"author_java_{engine}_Miner",
        "transport": (
            "binary_grouped_input_v3_selected_output_v2"
            if groups is not None
            else "binary_selected_collection_v2"
            if selection is not None
            else "binary_collection_v1"
        ),
        "class_policy": "reuse_CT_class"
        if engine == "CT"
        else "isolated_author_class_per_sample",
        "dictionary_lookup": lookup
        if engine == "CT" and wanted is not None
        else "enumerate",
        "python_input_seconds": input_s,
        "python_output_parse_seconds": parse_s,
        "java_batch_wall_seconds": wall,
        "jobs": len(samples),
        "benchmark_reconstructed_copies": benchmark_copies,
        "mining_seconds": stats["index_seconds"] + stats["count_seconds"],
    }

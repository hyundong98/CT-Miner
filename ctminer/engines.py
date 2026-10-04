"""Java execution and the Python CT reference adapter."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from ctminer.paths import java_sources
import re
import signal
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ROOT / "build/java"


class UnsupportedSetting(ValueError):
    pass


class EmptyFeatureSet(ValueError):
    """Selection completed, but no eligible feature exists."""

    pass


def _load_trie():
    # Use a private module name to avoid collisions with other CT implementations.
    name = "ct_miner_qsuftrie"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "ctminer/qsuftrie.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def _pd(x, inf):
    stack, codes = [], []
    for i, value in enumerate(x):
        while stack and x[stack[-1]] > value:
            stack.pop()
        codes.append(i - stack[-1] if stack else inf)
        stack.append(i)
    return codes


def _distinct_limits(x):
    # Longest distinct-valued prefix for each suffix, in O(n).
    next_seen, end = {}, len(x)
    result = np.empty(len(x), dtype=int)
    for i in range(len(x) - 1, -1, -1):
        end = min(end, next_seen.get(float(x[i]), len(x)))
        result[i] = end - i
        next_seen[float(x[i])] = i
    return result


def ct_counts(x, lengths, bins=1, decay=0.0, ties="native", wanted=None):
    mod = _load_trie()

    class IterativeTrie(mod.CHSuffixTree):
        # Same suffix construction, oracle, and links. Only postprocessing
        # traversal becomes iterative: no recursion-limit or full leaf validation.
        def preprocess_leaf_counts(self):
            self.ordered = []
            stack = [self.root]
            while stack:
                node = stack.pop()
                self.ordered.append(node)
                stack.extend(node.children.values())
            for node in reversed(self.ordered):
                node.leaf_count = (
                    1
                    if node.is_leaf
                    else sum(c.leaf_count for c in node.children.values())
                )

    start = perf_counter()
    tree = IterativeTrie(mod.PDOracle(_pd(x, mod.INF) + [-1]), len(x) + 1).build()
    indexed = perf_counter()
    # DFS intervals give occurrence starts without copying a list per node.
    need_positions = bins > 1 or ties == "drop_windows"
    intervals, starts = {}, []
    if need_positions:
        stack = [(tree.root, False)]
        while stack:
            node, done = stack.pop()
            if done:
                intervals[id(node)][1] = len(starts)
            else:
                intervals[id(node)] = [len(starts), 0]
                stack.append((node, True))
                if node.is_leaf:
                    starts.append(node.string_index)
                else:
                    stack.extend((child, False) for child in node.children.values())
        starts = np.asarray(starts, dtype=int)
    weights = {}
    if decay and not need_positions:
        for node in reversed(tree.ordered):
            weights[id(node)] = (
                np.exp(-decay * (len(x) - node.string_index - 1) / len(x))
                if node.is_leaf and node.string_index < len(x)
                else (
                    0.0
                    if node.is_leaf
                    else sum(weights[id(c)] for c in node.children.values())
                )
            )
    limits = _distinct_limits(x) if ties == "drop_windows" else None
    result = {}
    for node in tree.ordered:
        if node.parent is None:
            continue
        a, b = node.parent.length + 1, min(node.length, len(x) - node.index)
        for m in lengths:
            if not a <= m <= b:
                continue
            pattern = tuple(tree.path_label_truncate(node, m, True))
            if wanted is not None and pattern not in wanted:
                continue
            if need_positions:
                left, right = intervals[id(node)]
                pos = starts[left:right]
                pos = pos[pos <= len(x) - m]
                if limits is not None:
                    pos = pos[limits[pos] >= m]
                bins_at = np.minimum(bins - 1, (pos * bins // max(1, len(x) - m + 1)))
                mass = (
                    np.exp(-decay * (len(x) - pos - m) / len(x))
                    if decay
                    else np.ones(len(pos))
                )
                value = np.bincount(bins_at, weights=mass, minlength=bins).astype(float)
            else:
                value = np.array(
                    [
                        weights[id(node)] * np.exp(decay * (m - 1) / len(x))
                        if decay
                        else node.leaves
                    ],
                    dtype=float,
                )
            if value.sum() > 0:
                result[pattern] = value
    return result, {
        "index_seconds": indexed - start,
        "count_seconds": perf_counter() - indexed,
        "nodes": len(tree.ordered),
        "backend": "Cole_Hariharan_trie_iterative_postprocess",
    }


def _run(cmd, timeout, log_path):
    start = perf_counter()
    # Keep the descriptor open for diagnostics: a removed/renamed log pathname
    # must not hide the subprocess exit code or its original stderr.
    with Path(log_path).open("w+b") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True)
        try:
            returncode = proc.wait(timeout=timeout)
        except BaseException as exc:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            if isinstance(exc, subprocess.TimeoutExpired):
                raise TimeoutError(
                    f"{Path(cmd[0]).name} exceeded {timeout} seconds; {log_path}"
                ) from exc
            raise
        if returncode:
            try:
                log.seek(0, os.SEEK_END)
                log.seek(max(0, log.tell() - 8192))
                tail = log.read().decode("utf-8", errors="replace")
            except OSError as exc:
                tail = f"Unable to read subprocess log: {exc}"
            raise RuntimeError(
                f"{Path(cmd[0]).name} failed ({returncode}); {log_path}\n{tail}"
            )
    return perf_counter() - start


def _parse(path):
    result, metadata = {}, {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("M "):
            _, seconds, candidates = line.split()
            metadata = {"mining_seconds": float(seconds), "candidates": int(candidates)}
        elif line.startswith("P "):
            _, support, pattern = line.split(maxsplit=2)
            p = tuple(int(x) for x in re.findall(r"-?\d+", pattern))
            v = float(support)
            if not p or not np.isfinite(v) or v < 0 or p in result:
                raise ValueError(f"Invalid or duplicate author output: {path}")
            result[p] = v
    if not metadata:
        raise RuntimeError(f"Incomplete author adapter output: {path}")
    return result, metadata


def _java_ready():
    from scripts.build import verify_build

    verify_build()


def _parse_ct(path, bins):
    result, metadata = {}, None
    with Path(path).open() as stream:
        for raw in stream:
            fields = raw.rstrip("\n").split("\t")
            if fields[0] == "C" and len(fields) == 4 and metadata is None:
                metadata = {
                    "index_seconds": float(fields[1]),
                    "count_seconds": float(fields[2]),
                    "nodes": int(fields[3]),
                }
                if any(not np.isfinite(v) or v < 0 for v in metadata.values()):
                    raise ValueError(f"Invalid CT metadata: {path}")
            elif fields[0] == "V" and len(fields) == 3:
                p = tuple(int(v) for v in re.findall(r"-?\d+", fields[1]))
                values = np.asarray([float(v) for v in fields[2].split(",")])
                if (
                    not p
                    or p in result
                    or values.shape != (bins,)
                    or not np.all(np.isfinite(values))
                    or np.any(values < 0)
                ):
                    raise ValueError(f"Invalid or duplicate CT output: {path}")
                result[p] = values
            else:
                raise ValueError(f"Unexpected CT output: {path}")
    if metadata is None:
        raise RuntimeError(f"Incomplete CT adapter output: {path}")
    return result, metadata


def ct_java_batch(
    samples,
    lengths,
    bins,
    decay,
    ties,
    timeout,
    work,
    wanted=None,
    transport="legacy",
    lookup="direct",
    selection=None,
    benchmark_copies=False,
):
    """One JVM per collection; legacy isolates classes, collection reuses CT."""
    _java_ready()
    if transport == "collection":
        from ctminer.io import collection_batch

        return collection_batch(
            "CT",
            samples,
            lengths,
            bins,
            decay,
            ties,
            timeout,
            work,
            CLASSES,
            _run,
            wanted,
            lookup,
            selection,
            benchmark_copies,
        )
    if benchmark_copies:
        raise UnsupportedSetting(
            "Copy reconstruction benchmark requires java_transport=collection"
        )
    if selection is not None:
        raise UnsupportedSetting("Java corpus top-k requires java_transport=collection")
    if transport != "legacy" or lookup not in {"direct", "enumerate"}:
        raise ValueError("Invalid Java transport or CT lookup mode")
    lengths = sorted(set(map(int, lengths)))
    if (
        not lengths
        or lengths[0] < 2
        or bins < 1
        or ties not in {"native", "drop_windows"}
    ):
        raise ValueError("Invalid CT lengths, bins or tie policy")
    if not np.isfinite(decay) or not 0 <= decay <= 100:
        raise ValueError("CT decay must be in [0,100]")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    maps, timings = [], []
    with tempfile.TemporaryDirectory(prefix="ct-java-", dir=work) as tmp:
        tmp = Path(tmp)
        jobs = []
        wanted_path = "-"
        if wanted is not None:
            wanted_path = str(tmp / "wanted.csv")
            with Path(wanted_path).open("w") as stream:
                for pattern in sorted(wanted):
                    stream.write(",".join(map(str, pattern)) + "\n")
        for i, x in enumerate(samples):
            inp = tmp / f"{i}.txt"
            out = tmp / f"{i}.out"
            np.savetxt(inp, x, fmt="%.9g")
            jobs.append(
                "\t".join(
                    [
                        "CTMiner",
                        str(out),
                        str(inp),
                        ",".join(map(str, lengths)),
                        str(bins),
                        str(decay),
                        ties,
                        wanted_path,
                        str(lookup == "direct").lower(),
                    ]
                )
            )
        jobfile = tmp / "jobs.tsv"
        jobfile.write_text("\n".join(jobs) + "\n")
        wall = _run(
            [
                "java",
                "-Xmx2g",
                "-cp",
                str(CLASSES),
                "ReproBatch",
                str(CLASSES),
                str(jobfile),
            ],
            timeout,
            work / "ct-java.log",
        )
        for i in range(len(samples)):
            row, stat = _parse_ct(tmp / f"{i}.out", bins)
            maps.append(row)
            timings.append(stat)
    totals = {
        key: sum(t[key] for t in timings)
        for key in ("index_seconds", "count_seconds", "nodes")
    }
    return maps, {
        **totals,
        "backend": "java_Cole_Hariharan_trie",
        "transport": "legacy_text",
        "dictionary_lookup": lookup if wanted is not None else "enumerate",
        "java_batch_wall_seconds": wall,
        "mining_seconds": totals["index_seconds"] + totals["count_seconds"],
        "jobs": len(samples),
    }


def engine_provenance():
    path = ROOT / "ctminer/qsuftrie.py"
    manifest = CLASSES / "manifest.json"
    return {
        "trie_path": str(path),
        "trie_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "ct_java_source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in java_sources()
        },
        "java_manifest": json.loads(manifest.read_text())
        if manifest.is_file()
        else None,
    }

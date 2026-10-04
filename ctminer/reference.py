#!/usr/bin/env python3
"""Sparse mining adapter over the actual Cartesian suffix trie.

No dense Catalan feature space and no DP/Numba engine. The unchanged qsuftrie
module is imported directly. Pattern labels are only expanded when requested.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parent
from ctminer.qsuftrie import CHSuffixTree, PDOracle, INF


def pd_encoding(values, sentinel=False):
    stack = []
    result = []
    for i, v in enumerate(values):
        while stack and stack[-1][0] > v:
            stack.pop()
        result.append(i - stack[-1][1] if stack else (INF if sentinel else 0))
        stack.append((v, i))
    if sentinel:
        result.append(-1)
    return result


def mine(values, minsup, maxlen=0, dump=False, lo=2, topk=0):
    start = time.perf_counter()
    n = len(values)
    maxlen = min(maxlen or n, n)
    # Original implementation has recursive leaf preprocessing; accommodate
    # monotone inputs without changing its algorithm. Resource limits are external.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 4 * n + 100))
    tree = CHSuffixTree(PDOracle(pd_encoding(values, True)), n + 1)
    tree.build()
    built = time.perf_counter()
    nodes = tree.get_nodes()
    intervals = []
    frequent = 0
    max_found = 0
    for node in nodes:
        if node.parent is None or node.leaves < minsup:
            continue
        a = max(lo, node.parent.length + 1)
        b = min(maxlen, node.length, n - node.index)
        if a <= b:
            intervals.append((a, b, node))
            frequent += b - a + 1
            max_found = max(max_found, b)
    patterns = []
    if topk:
        if lo != maxlen:
            raise ValueError("topk queries require a single fixed length")
        intervals.sort(key=lambda it: (-it[2].leaves, it[2].index))
        intervals = intervals[:topk]
        frequent = len(intervals)
    if dump:
        for a, b, node in intervals:
            for m in range(a, b + 1):
                patterns.append(
                    (tuple(tree.path_label_truncate(node, m, True)), node.leaves)
                )
    end = time.perf_counter()
    result = {
        "algorithm": "CT-trie",
        "semantics": "ct_stable_left",
        "n": n,
        "frequent_count": frequent,
        "candidates": None,
        "nodes": len(nodes),
        "max_frequent_length": max_found,
        "index_seconds": built - start,
        "query_seconds": end - built,
        "mining_seconds": end - start,
        "output_contract": "explicit_patterns" if dump else "compact_pattern_intervals",
        "trie_sha256": hashlib.sha256((ROOT / "qsuftrie.py").read_bytes()).hexdigest(),
    }
    return result, patterns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("minsup", type=int)
    ap.add_argument("maxlen", type=int, nargs="?", default=0)
    ap.add_argument("dump", type=int, nargs="?", default=0)
    ap.add_argument("--lo", type=int, default=2)
    ap.add_argument("--topk", type=int, default=0)
    args = ap.parse_args()
    if args.minsup < 1 or args.maxlen < 0 or args.lo < 2 or args.topk < 0:
        ap.error("invalid parameter")
    vals = [
        struct.unpack("f", struct.pack("f", float(x)))[0]
        for x in args.input.read_text().split()
    ]
    if any(not __import__("math").isfinite(x) for x in vals):
        raise ValueError("nonfinite input")
    result, patterns = mine(
        vals, args.minsup, args.maxlen, bool(args.dump), args.lo, args.topk
    )
    print(json.dumps(result))
    for p, count in patterns:
        print("P", count, *p)


if __name__ == "__main__":
    main()

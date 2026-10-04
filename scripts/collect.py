"""Summarize a completed native, hash, or memory experiment directory."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    root = args.results
    if (root / "memory_run.json").is_file():
        from experiments.memory import collect

        collect(root)
        return
    config = json.loads((root / "config.json").read_text())
    if "source" in config and "l_values" in config:
        from experiments.hash_comparison import summarize
    else:
        from ctminer.study import summarize
    summarize(root)


if __name__ == "__main__":
    main()

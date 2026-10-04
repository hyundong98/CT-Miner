#!/usr/bin/env python3
"""Build, prepare data, and run the CT-Miner experiments."""

import argparse
import importlib
import os
from pathlib import Path
import sys

COMMANDS = {
    "build": ("scripts.build", "Compile the Java implementations"),
    "prepare": ("scripts.prepare", "Prepare the six UCR datasets"),
    "clustering": ("experiments.clustering", "Top-K and threshold clustering"),
    "input-scaling": ("experiments.clustering", "Clustering on growing input prefixes"),
    "hash": ("experiments.hash_comparison", "CT versus CT-Hash"),
    "memory": ("experiments.memory", "Independent sampled feature-RSS measurements"),
    "validate": ("scripts.smoke", "Check mining, features, and clustering"),
    "collect": ("scripts.collect", "Collect completed clustering runs"),
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, epilog="Use COMMAND --help for its options."
    )
    parser.add_argument(
        "command",
        choices=COMMANDS,
        help="; ".join(f"{k}: {v[1]}" for k, v in COMMANDS.items()),
    )
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        return
    args = parser.parse_args(sys.argv[1:2])
    rest = sys.argv[2:]
    if args.command in {"clustering", "input-scaling"}:
        if "--only" in rest or any(x.startswith("--only=") for x in rest):
            parser.error(
                "Select clustering or input-scaling; --only is reserved for these commands"
            )
        rest += ["--only", *(["1", "2"] if args.command == "clustering" else ["3"])]
    # Relative data and output paths are always relative to this repository.
    os.chdir(Path(__file__).resolve().parent)
    sys.argv = [f"run.py {args.command}", *rest]
    importlib.import_module(COMMANDS[args.command][0]).main()


if __name__ == "__main__":
    main()

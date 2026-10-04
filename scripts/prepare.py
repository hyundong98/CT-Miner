"""Prepare the six UCR datasets used in the experiments."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ctminer.paths import ROOT
from ctminer.worker import prepare

DATASETS = ["Car", "Beef", "ElectricDevices", "MoteStrain", "PigCVP", "Wafer"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--output", type=Path, default=Path("data/prepared"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--archive-variants", choices=["compatible", "original"], default="compatible"
    )
    args = parser.parse_args()
    args.catalogue = "ucr"
    prepare(args)
    manifest = json.loads((args.output / "manifest.json").read_text())
    failed = [
        name
        for name in args.datasets
        if manifest["datasets"].get(name, {}).get("status") != "ready"
    ]
    if failed:
        raise SystemExit(
            f"Datasets not ready: {', '.join(failed)}. See the data manifest."
        )
    expected = json.loads((ROOT / "experiments/datasets.json").read_text())
    for name in args.datasets:
        entry = manifest["datasets"][name]
        with np.load(args.output / entry["npz"], allow_pickle=False) as data:
            x = np.ascontiguousarray(data["X"], dtype="<f8")
            labels = data["y"].astype(str).tolist()
        actual = dict(
            samples=len(labels),
            length=x.shape[1],
            classes=len(set(labels)),
            values_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
            labels_sha256=hashlib.sha256(
                json.dumps(labels, separators=(",", ":")).encode()
            ).hexdigest(),
        )
        if actual != expected[name]:
            raise SystemExit(
                f"{name}: data differs from the recorded experiment input. Check the archive variant and aeon version."
            )
    print("All requested datasets match the recorded input fingerprints.")


if __name__ == "__main__":
    main()

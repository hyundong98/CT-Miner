"""Repository paths and source fingerprints."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ROOT / "build/java"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def java_sources():
    paths = list((ROOT / "ctminer/java").glob("*.java"))
    paths += list((ROOT / "baseline/common").glob("*.java"))
    paths += list((ROOT / "baseline/ct_hash").glob("*.java"))
    return sorted(paths)


def code_hashes():
    paths = [ROOT / "run.py"]
    for folder in ("ctminer", "experiments", "scripts"):
        paths.extend((ROOT / folder).glob("*.py"))
    paths.extend(java_sources())
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}

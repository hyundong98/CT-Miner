"""Compile CT-Miner, CT-Hash, and shared Java utilities."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from ctminer.paths import ROOT, CLASSES, java_sources, sha256

JAVA_VERSION = "25.0.2"


def verify_java():
    if os.environ.get("JAVA_HOME"):
        java_bin = str(Path(os.environ["JAVA_HOME"]) / "bin")
        os.environ["PATH"] = java_bin + os.pathsep + os.environ.get("PATH", "")
    versions = {}
    for tool in ("java", "javac"):
        if not shutil.which(tool):
            raise RuntimeError(
                f"{tool} is missing. Activate the environment from environment.yml"
            )
        output = subprocess.check_output(
            [tool, "-version"], text=True, stderr=subprocess.STDOUT
        ).strip()
        if not re.search(r"\b" + re.escape(JAVA_VERSION) + r"(?=[\s\"+\-]|$)", output):
            raise RuntimeError(
                f"Expected {tool} {JAVA_VERSION}, found {output.splitlines()[0]}. "
                "Activate the environment from environment.yml."
            )
        versions[tool + "_version"] = output
    return versions


def inputs():
    paths = java_sources() + [Path(__file__), ROOT / "ctminer/qsuftrie.py"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def verify_build():
    path = CLASSES / "manifest.json"
    if not path.is_file():
        raise RuntimeError("Java classes are missing. Run: python run.py build")
    info = json.loads(path.read_text())
    if info.get("status") != "built" or info.get("source_sha256") != inputs():
        raise RuntimeError("Java build is stale. Run: python run.py build")
    if info.get("required_java_version") != JAVA_VERSION:
        raise RuntimeError(f"Rebuild with OpenJDK {JAVA_VERSION}: python run.py build")
    for name, expected in info["class_sha256"].items():
        if not (CLASSES / name).is_file() or sha256(CLASSES / name) != expected:
            raise RuntimeError(
                f"Java class missing or changed: {name}; run python run.py build"
            )
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify an existing build")
    args = parser.parse_args()
    versions = verify_java()
    if args.check:
        info = verify_build()
        print(f"Build verified: {len(info['class_sha256'])} classes")
        return
    sources = java_sources()
    if len({p.name for p in sources}) != len(sources):
        raise ValueError("Duplicate Java source filenames")
    CLASSES.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="java-", dir=CLASSES.parent) as tmp:
        build = Path(tmp)
        for path in sources:
            shutil.copyfile(path, build / path.name)
        command = ["javac", "-encoding", "UTF-8", "-d", str(build)] + [
            str(build / p.name) for p in sources
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        info = dict(
            status="built",
            **versions,
            required_java_version=JAVA_VERSION,
            source_sha256=inputs(),
            ct_python_reference_sha256=sha256(ROOT / "ctminer/qsuftrie.py"),
            command=["javac", "-encoding", "UTF-8", "-d", "build/java"]
            + ["build/java/" + p.name for p in sources],
            class_sha256={p.name: sha256(p) for p in sorted(build.glob("*.class"))},
        )
        # Replace only the generated build after javac succeeds.
        if CLASSES.exists():
            shutil.rmtree(CLASSES)
        CLASSES.mkdir()
        for path in build.glob("*.class"):
            shutil.copyfile(path, CLASSES / path.name)
        (CLASSES / "manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Built {len(info['class_sha256'])} classes with {versions['javac_version']}")


if __name__ == "__main__":
    main()

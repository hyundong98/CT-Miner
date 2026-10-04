"""Reuse completed jobs only when inputs, code, and settings match."""

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from ctminer.study import read, write, digest, fingerprint

SPEC_FIELDS = (
    "dataset",
    "engine",
    "budget",
    "study",
    "length_percent",
    "repeat",
    "feature_mode",
    "minsup",
    "minconf",
    "maxsta",
)
# These describe a sweep, not the computation of one job. Individual budget,
# seed list, pattern lengths, input hash, packages/build and limits must match.
SWEEP_FIELDS = {
    "datasets",
    "methods",
    "study",
    "repeats",
    "length_percents",
    "input_manifest_sha256",
    "code",
}


def key(job):
    return tuple(job.get(name) for name in SPEC_FIELDS)


def compatible_code(saved, current):
    if saved != current:
        raise ValueError("Source code changed; use a new output directory")
    return {}


def validate_saved_job(job, result, expected):
    unsigned = {
        name: value
        for name, value in job.items()
        if name not in {"fingerprint", "execution"}
    }
    if (
        job.get("fingerprint") != expected
        or fingerprint(unsigned) != expected
        or result.get("job_fingerprint") != expected
    ):
        raise ValueError("Source job/result fingerprint mismatch")
    if result.get("status") not in {"complete", "timeout"}:
        raise ValueError("Only complete/timeout jobs can be reused")
    resources = result.get("resources", {})
    if not resources or "returncode" not in resources:
        raise ValueError("Source result is not finalized by its supervisor")
    if result["status"] == "complete" and (
        resources["returncode"] != 0 or not result.get("rows")
    ):
        raise ValueError("Incomplete successful source result")


def normalized_build(manifest):
    """Ignore build-directory relocation only, preserving compiler/options/code."""
    if manifest.get("status") != "built" or not manifest.get("class_sha256"):
        raise ValueError("Cannot compare incomplete Java build manifests")
    command = manifest.get("command")
    if (
        not isinstance(command, list)
        or len(command) < 6
        or not all(isinstance(part, str) for part in command)
        or command[:4] != ["javac", "-encoding", "UTF-8", "-d"]
    ):
        raise ValueError("Unrecognized Java build command; cannot normalize its paths")
    folder = Path(command[4])
    sources = [Path(part) for part in command[5:]]
    # Current builds record repository-relative paths; older manifests record
    # absolute temporary paths. Compare their lexical structure without making
    # the result depend on the current checkout or on files that no longer exist.
    if ".." in folder.parts or any(
        p.is_absolute() != folder.is_absolute()
        or ".." in p.parts
        or p.parent != folder
        or p.suffix != ".java"
        for p in sources
    ):
        raise ValueError(
            "Java source paths are not all inside the recorded build directory"
        )
    normalized = dict(manifest)
    normalized["command"] = (
        command[:4] + ["<BUILD_DIR>"] + ["<BUILD_DIR>/" + p.name for p in sources]
    )
    return normalized


def compatible_build(old, new, source_input, evidence=None):
    """Authenticate the old manifest, then compare content rather than paths.

    Legacy jobs saved only a file hash. Locate that exact manifest beneath an
    ancestor of their original input path, or use an authenticated audit snapshot.
    Never waive missing evidence or changes to classes, sources, compiler/options.
    """
    before, after = old["build_manifest_sha256"], new["build_manifest_sha256"]
    if before == after:
        return None
    current_path = Path(__file__).resolve().parents[1] / "build/java/manifest.json"
    current_bytes = current_path.read_bytes()
    if hashlib.sha256(current_bytes).hexdigest() != after:
        raise ValueError("Current Java build manifest changed after preflight")
    old_bytes, old_path = None, None
    if evidence and evidence.get("source_manifest_text"):
        candidate = evidence["source_manifest_text"].encode("utf-8")
        if hashlib.sha256(candidate).hexdigest() == before:
            old_bytes, old_path = (
                candidate,
                evidence.get("source_manifest_path", "saved snapshot"),
            )
    if old_bytes is None:
        for parent in Path(source_input).parents:
            candidate_path = parent / "build/java/manifest.json"
            if candidate_path.is_file():
                candidate = candidate_path.read_bytes()
                if hashlib.sha256(candidate).hexdigest() == before:
                    old_bytes, old_path = candidate, str(candidate_path)
                    break
    if old_bytes is None:
        raise ValueError(
            "Original Java build manifest matching the stored hash was not found. "
            "Keep the original project's build/java/manifest.json; "
            "do not overwrite hashes to force reuse."
        )
    a, b = (
        normalized_build(json.loads(old_bytes)),
        normalized_build(json.loads(current_bytes)),
    )
    different = [name for name in a.keys() | b.keys() if a.get(name) != b.get(name)]
    if different:
        raise ValueError(
            f"Java build content differs beyond directory paths: {sorted(different)}"
        )
    return dict(
        before=before,
        after=after,
        policy="Identical build content; only build paths/JSON serialization may differ",
        source_manifest_path=old_path,
        current_manifest_path=str(current_path),
        source_manifest_text=old_bytes.decode("utf-8"),
        current_manifest_text=current_bytes.decode("utf-8"),
    )


def compatible_job(source, target, build_evidence=None):
    if key(source) != key(target) or source["input_sha256"] != target["input_sha256"]:
        raise ValueError("Source job identity or prepared data differs")
    old, new = source["config"], target["config"]
    differences = [
        name
        for name in old.keys() | new.keys()
        if name not in SWEEP_FIELDS | {"build_manifest_sha256"}
        and old.get(name) != new.get(name)
    ]
    if differences:
        raise ValueError(
            f"Source scientific/execution settings differ: {sorted(differences)}"
        )
    changes = compatible_code(old["code"], new["code"])
    build = compatible_build(old, new, source["input"], build_evidence)
    if build:
        changes["build_manifest_sha256"] = build
    return changes


def existing_reuse(out, target):
    path = out / "reused_from.json"
    if not path.exists():
        return None
    audit = read(path)
    if audit["target_fingerprint"] != target["fingerprint"]:
        raise ValueError(
            f"Reused result was imported for different target settings: {out}"
        )
    job, result = read(out / "job.json"), read(out / "result.json")
    validate_saved_job(job, result, audit["source_fingerprint"])
    compatible_job(
        job, target, audit.get("source_changes", {}).get("build_manifest_sha256")
    )
    if digest(out / "result.json") != audit["result_sha256"]:
        raise ValueError(f"Reused result changed after import: {out}")
    return audit


class ReuseCatalog:
    def __init__(self, roots, target_root, cfg):
        self.roots = sorted({p.resolve() for p in roots})
        self.target_root, self.cfg = target_root, cfg
        self.stack, self.candidates = ExitStack(), {}

    def __enter__(self):
        import fcntl

        try:
            for root in self.roots:
                if (
                    root == self.target_root
                    or root.is_relative_to(self.target_root)
                    or self.target_root.is_relative_to(root)
                ):
                    raise ValueError(
                        "Reuse source and destination must be separate, non-nested result directories"
                    )
                if not (root / "job_index.json").is_file():
                    raise ValueError(
                        f"Reuse source does not contain job_index.json: {root}"
                    )
                lock = self.stack.enter_context((root / "run.lock").open("a"))
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError(
                        f"Reuse source is still running: {root}; finish or stop its supervisor first"
                    )
                for entry in read(root / "job_index.json"):
                    if (
                        entry["dataset"] not in self.cfg["datasets"]
                        or entry["engine"] not in self.cfg["methods"]
                    ):
                        continue
                    out = (root / entry["path"]).resolve()
                    if not out.is_relative_to(root) or out == root:
                        raise ValueError(f"Invalid source job path: {out}")
                    path = out / "result.json"
                    result = read(path) if path.exists() else {}
                    if result.get("status") not in {"complete", "timeout"}:
                        continue
                    job_key = key(entry)
                    if job_key in self.candidates:
                        raise ValueError(
                            f"Multiple source results for {job_key}; specify one source per condition"
                        )
                    self.candidates[job_key] = (out, entry, result)
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def copy(self, target, destination):
        candidate = self.candidates.get(key(target))
        if candidate is None:
            return None
        source, entry, result = candidate
        job = read(source / "job.json")
        try:
            validate_saved_job(job, result, entry["fingerprint"])
            if key(job) != key(entry):
                raise ValueError("Source index/job identity mismatch")
            earlier = (
                read(source / "reused_from.json")
                if (source / "reused_from.json").exists()
                else {}
            )
            changes = compatible_job(
                job,
                target,
                earlier.get("source_changes", {}).get("build_manifest_sha256"),
            )
        except ValueError as exc:
            raise ValueError(f"Cannot reuse {source}: {exc}") from exc
        if destination.exists():
            raise ValueError(
                f"Partial destination already exists: {destination}; inspect it before importing"
            )
        result_hash = digest(source / "result.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".reuse-", dir=destination.parent))
        try:
            # Copy, rather than symlink: the combined report owns its evidence.
            # Failed older attempts are retained in the original source only.
            shutil.copytree(
                source,
                stage,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("attempts"),
            )
            if digest(stage / "result.json") != result_hash:
                raise ValueError(f"Result changed while copying: {source}")
            audit = dict(
                source_job_path=str(source),
                source_fingerprint=entry["fingerprint"],
                target_fingerprint=target["fingerprint"],
                result_sha256=result_hash,
                source_changes=changes,
                status=result["status"],
                source_execution=job.get("execution"),
                earlier_reuse=read(source / "reused_from.json")
                if (source / "reused_from.json").exists()
                else None,
                policy="No re-mining; original job/config/fingerprint/result/resources retained. Copy time is not experimental runtime.",
            )
            write(stage / "reused_from.json", audit)
            stage.rename(destination)
            return audit
        finally:
            if stage.exists():
                shutil.rmtree(stage)

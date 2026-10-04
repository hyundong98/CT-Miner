"""UCR data loading with a checksummed Zenodo fallback."""

from http.client import IncompleteRead, RemoteDisconnected
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np

NETWORK_ERRORS = (
    HTTPError,
    URLError,
    TimeoutError,
    ConnectionError,
    IncompleteRead,
    RemoteDisconnected,
    socket.timeout,
)


def download(url, target):
    """Bounded retries for transient errors; a 401 is not retried as transient."""
    for attempt in range(3):
        delay = 2 ** (attempt + 1)
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "CTPM-reproduction/2 (public TSML archive downloader)"
                },
            )
            with (
                urlopen(request, timeout=60) as response,
                Path(target).open("wb") as stream,
            ):
                shutil.copyfileobj(response, stream)
            return
        except NETWORK_ERRORS as exc:
            if isinstance(exc, HTTPError):
                if exc.code not in {408, 429, 500, 502, 503, 504}:
                    raise
                retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
                if retry_after.isdigit():
                    delay = min(60, max(delay, int(retry_after)))
            if attempt == 2:
                raise
            time.sleep(delay)


def checksum(path, algorithm):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def zenodo_load(name, root, compatible, audit):
    from aeon.datasets import load_from_ts_file
    from aeon.datasets.tsc_datasets import tsc_zenodo

    if name not in tsc_zenodo:
        raise ValueError(f"No official aeon Zenodo record mapping for {name}")
    record_id = int(tsc_zenodo[name])
    api = f"https://zenodo.org/api/records/{record_id}"
    folder = Path(root) / name
    folder.mkdir(parents=True, exist_ok=True)
    audit.update(record_id=record_id, metadata_url=api)
    # Cached metadata pins a successful recovery to the same remote file hashes.
    metadata_path = folder / "zenodo-record.json"
    if metadata_path.is_file():
        record = json.loads(metadata_path.read_text())
    else:
        with tempfile.TemporaryDirectory(prefix="metadata-", dir=folder) as tmp:
            target = Path(tmp) / "record.json"
            download(api, target)
            record = json.loads(target.read_text())
            if str(record.get("id")) != str(record_id) or not isinstance(
                record.get("files"), list
            ):
                raise ValueError(f"Invalid public Zenodo record: {api}")
            target.replace(metadata_path)
    if str(record.get("id")) != str(record_id):
        raise ValueError("Cached Zenodo record ID mismatch")
    files = {entry["key"]: entry for entry in record["files"]}

    def paired(stem):
        return all(f"{stem}_{split}.ts" in files for split in ("TRAIN", "TEST"))

    # Match aeon 1.3.0's order: discrete, then equal length, then no missing.
    stem = name
    if paired(stem + "_disc"):
        stem += "_disc"
    if compatible and paired(stem + "_eq"):
        stem += "_eq"
    if compatible and paired(stem + "_nmv"):
        stem += "_nmv"
    if not paired(stem):
        raise ValueError(f"No complete requested TRAIN/TEST pair for {stem} in {api}")
    audit.update(selected_variant=stem, source_files=[])
    loaded = []
    for split in ("TRAIN", "TEST"):
        filename = f"{stem}_{split}.ts"
        if Path(filename).name != filename:
            raise ValueError("Invalid archive filename")
        entry = files[filename]
        algorithm, expected = entry.get("checksum", "").partition(":")[::2]
        if algorithm not in {"md5", "sha256"} or not re.fullmatch(
            r"[0-9a-fA-F]+", expected
        ):
            raise ValueError(f"Missing/unsupported published checksum for {filename}")
        target = folder / filename
        url = f"https://zenodo.org/records/{record_id}/files/{quote(filename, safe='')}?download=1"
        audit["active_url"] = url
        if (
            not target.is_file()
            or checksum(target, algorithm).lower() != expected.lower()
        ):
            with tempfile.TemporaryDirectory(prefix="file-", dir=folder) as tmp:
                pending = Path(tmp) / filename
                download(url, pending)
                if checksum(pending, algorithm).lower() != expected.lower():
                    raise ValueError(f"Published checksum mismatch: {url}")
                pending.replace(target)
        audit["source_files"].append(
            dict(
                path=str(target.resolve()),
                url=url,
                published_checksum=entry["checksum"],
                sha256=checksum(target, "sha256"),
            )
        )
        data, labels, metadata = load_from_ts_file(str(target), return_meta_data=True)
        if not metadata.get("classlabel"):
            raise ValueError(f"Expected classification labels in {filename}")
        loaded.append((data, labels, metadata))
    train, test = loaded
    # Lists support both unequal originals and equal-length archive variants.
    data = list(train[0]) + list(test[0])
    labels = np.concatenate([train[1], test[1]])
    metadata = dict(train[2])
    metadata.update(
        n_cases=len(labels), source_split_metadata={"TRAIN": train[2], "TEST": test[2]}
    )
    audit.pop("active_url", None)
    return data, labels, metadata


def load(name, cache, recovery_root, compatible, audit):
    from aeon.datasets import load_classification

    audit.update(loader="aeon_with_public_zenodo_fallback_v1")
    try:
        result = load_classification(
            name,
            split=None,
            extract_path=str(Path(cache).resolve()),
            return_metadata=True,
            load_equal_length=compatible,
            load_no_missing=compatible,
        )
        audit["backend"] = "aeon"
        return result
    except NETWORK_ERRORS as exc:
        primary = exc
    except ValueError as exc:
        # aeon's own Zenodo failure wraps connection errors in this message.
        # Do not relabel arbitrary data/parse errors as network failures.
        if (
            "not available on extract path" not in str(exc)
            or "zenodo" not in str(exc).lower()
        ):
            raise
        primary = exc
    audit.update(
        backend="public_zenodo_fallback",
        primary_error=str(primary),
        primary_url=getattr(primary, "url", None),
    )
    try:
        return zenodo_load(name, recovery_root, compatible, audit)
    except Exception as exc:
        raise RuntimeError(
            f"aeon download failed: {primary}; official Zenodo recovery failed: {exc}"
        ) from exc

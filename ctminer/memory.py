"""Peak sampled worker-plus-descendant RSS during feature construction."""

import json
import math
import os
from pathlib import Path
from threading import Event, Thread

from ctminer.resources import TreeMeter

VERSION = "linux_sampled_feature_rss_v3"
SAMPLE_INTERVAL = 0.05


def sampled_peak_bytes(memory):
    if memory.get("version") != VERSION or memory.get("status") != "ok":
        return None
    peak = memory.get("sampled_peak_rss_bytes")
    samples = memory.get("samples", 0)
    if not isinstance(peak, int) or peak <= 0 or samples < 1:
        return None
    if memory.get("worker_rss_samples") != samples:
        return None
    if memory.get("expect_children") and memory.get("java_rss_samples", 0) < 1:
        return None
    return peak


class FeatureMemory:
    def __init__(self, output, *, interval=SAMPLE_INTERVAL, expect_children=True):
        if not math.isfinite(interval) or interval < 0.02:
            raise ValueError("RSS sampling interval must be finite and >= 0.02 seconds")
        self.output = Path(output)
        self.interval = interval
        self.expect_children = expect_children
        self.stop = Event()
        self.error = None

    def sample(self):
        try:
            self.meter.sample("feature_construction")
        except Exception as exc:
            self.error = str(exc)
            self.stop.set()

    def poll(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.meter = TreeMeter(os.getpid(), self.interval)
        # Boundary samples cover stages shorter than the polling interval.
        self.sample()
        self.thread = Thread(target=self.poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join()
        self.sample()
        result = self.meter.result()
        status = (
            "incomplete"
            if exc_type is not None
            else "sampling_error"
            if self.error is not None
            else "missing_worker_samples"
            if not result["samples"]
            or result["worker_rss_samples"] != result["samples"]
            else "missing_java_samples"
            if self.expect_children and not result["java_rss_samples"]
            else "ok"
        )
        result.update(
            version=VERSION,
            status=status,
            expect_children=self.expect_children,
            boundary="feature_complete",
            scope="maximum sampled simultaneous RSS sum of worker and descendants during feature construction; includes invocation and feature generation; excludes clustering and later output diagnostics",
        )
        if self.error is not None:
            result["reason"] = self.error
        self.output.parent.mkdir(parents=True, exist_ok=True)
        pending = self.output.with_suffix(".tmp")
        pending.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        pending.replace(self.output)

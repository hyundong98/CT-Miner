"""Linux CPU affinity and sampled process-tree resource usage."""

import os
from pathlib import Path
import signal
from time import monotonic


def cpu_slots(jobs, cores_per_job, affinity):
    if affinity == "none":
        return [None for _ in range(jobs)]
    if not hasattr(os, "sched_getaffinity") or not Path("/proc").is_dir():
        raise RuntimeError(
            "Physical-core allocation requires Linux; use --affinity none explicitly elsewhere"
        )
    groups = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = (
            int((base / "physical_package_id").read_text()),
            int((base / "core_id").read_text()),
        )
        groups.setdefault(key, []).append(cpu)
    if len(groups) < jobs * cores_per_job:
        raise ValueError(
            f"Need {jobs * cores_per_job} available physical cores, found {len(groups)}; reduce --jobs or --cores-per-job"
        )
    cores = sorted(groups.items())
    return [
        sorted(
            cpu
            for _, siblings in cores[i * cores_per_job : (i + 1) * cores_per_job]
            for cpu in siblings
        )
        for i in range(jobs)
    ]


def process_table():
    table = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            s = (path / "stat").read_text()
            fields = s[s.rfind(")") + 2 :].split()
            table[int(path.name)] = (
                int(fields[1]),
                int(fields[19]),
                int(fields[11]) + int(fields[12]),
                max(0, int(fields[21])),
                s[s.find("(") + 1 : s.rfind(")")],
            )
        except (OSError, ValueError, IndexError):
            continue
    return table


class TreeMeter:
    def __init__(self, pid, interval):
        if not Path("/proc").is_dir():
            raise RuntimeError("Per-job process-tree measurement requires Linux /proc")
        self.pid, self.interval = pid, interval
        self.seen, self.last_cpu, self.stages = {}, {}, {}
        self.peak = 0
        self.java_peak = 0
        self.java_samples = 0
        self.worker_samples = 0
        self.samples = 0
        self.started = monotonic()
        self.page = os.sysconf("SC_PAGE_SIZE")
        self.hz = os.sysconf("SC_CLK_TCK")

    def sample(self, phase):
        table = process_table()
        active = (
            {self.pid}
            if self.pid in table
            and (self.pid not in self.seen or table[self.pid][1] == self.seen[self.pid])
            else set()
        )
        active.update(
            pid
            for pid, start in self.seen.items()
            if pid in table and table[pid][1] == start
        )
        # Children may have their own process group/session (the Java adapter does).
        while True:
            children = {pid for pid, data in table.items() if data[0] in active}
            enlarged = active | children
            if enlarged == active:
                break
            active = enlarged
        rss = java_rss = 0
        for pid in active:
            _, start, ticks, pages, name = table[pid]
            self.seen[pid] = start
            self.last_cpu[(pid, start)] = ticks
            rss += pages * self.page
            if pid == self.pid and pages > 0:
                self.worker_samples += 1
            if name == "java":
                java_rss += pages * self.page
        now = monotonic() - self.started
        self.peak = max(self.peak, rss)
        self.java_peak = max(self.java_peak, java_rss)
        if java_rss > 0:
            self.java_samples += 1
        self.samples += 1
        stage = self.stages.setdefault(
            phase,
            dict(
                first_observed_s=now,
                last_observed_s=now,
                sampled_peak_rss_bytes=0,
                samples=0,
            ),
        )
        stage["last_observed_s"] = now
        stage["sampled_peak_rss_bytes"] = max(stage["sampled_peak_rss_bytes"], rss)
        stage["samples"] += 1

    def signal(self, sig=signal.SIGTERM):
        self.sample("termination")
        table = process_table()
        for pid, start in self.seen.items():
            if pid in table and table[pid][1] == start:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass

    def result(self):
        return dict(
            sampled_peak_rss_bytes=self.peak,
            sampled_java_peak_rss_bytes=self.java_peak if self.java_samples else None,
            java_rss_samples=self.java_samples,
            worker_rss_samples=self.worker_samples,
            sampled_cpu_seconds=sum(self.last_cpu.values()) / self.hz,
            sample_interval_seconds=self.interval,
            samples=self.samples,
            tracked_processes=len(self.last_cpu),
            phases=self.stages,
            memory_scope="maximum sampled simultaneous RSS sum of worker and descendants; shared pages can be double-counted; not JVM heap or PSS",
            cpu_scope="sampled per-process CPU totals; short-lived processes may be missed; prefer worker_cpu_seconds when present",
        )

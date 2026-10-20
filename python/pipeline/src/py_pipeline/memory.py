"""Process memory sampling for one pipeline run."""

__all__ = ["MemoryMonitor"]

import threading

import psutil

_GIB = 1024**3
_SAMPLE_INTERVAL_S = 0.05
_STAGES = frozenset(
    (
        "startup",
        "setup",
        "preprocess",
        "inference",
        "postprocess",
        "between",
        "output",
    )
)
_PAYLOAD_STAGES = ("preprocess", "inference", "postprocess")


def _working_set_gib(process: psutil.Process) -> float:
    return process.memory_info().rss / _GIB


class MemoryMonitor:
    """Track the process working set for each pipeline stage and the whole run.

    The monitor samples every 50 ms to limit interference with stage timings.
    """

    def __init__(self) -> None:
        self._process = psutil.Process()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._stage = "startup"
        self._stage_peak = {stage: 0.0 for stage in _STAGES}
        self._run_peak = 0.0

    def start(self) -> None:
        """Start sampling the process working set."""
        if self._thread is not None:
            return
        with self._lock:
            self._sample_locked()
        self._thread = threading.Thread(
            target=self._run,
            name="memory-monitor",
            daemon=True,
        )
        self._thread.start()

    def enter_stage(self, stage: str) -> None:
        """Record the current stage and start tracking the next one."""
        if stage not in _STAGES:
            raise ValueError(f"Unknown memory stage: {stage}")
        with self._lock:
            self._sample_locked()
            self._stage = stage

    def stop(self) -> None:
        """Stop sampling and take the final sample."""
        thread = self._thread
        if thread is None:
            self._stop.set()
            return
        self._stop.set()
        thread.join()
        with self._lock:
            self._sample_locked()
            self._thread = None
            error = self._error
        if error is not None:
            raise error

    @property
    def peak_ram_gb(self) -> float:
        """Return the highest working set seen during this run."""
        with self._lock:
            return self._run_peak

    def stage_ram_gb(self, stage: str) -> float:
        """Return the highest working set seen during one stage."""
        with self._lock:
            return self._stage_peak.get(stage, 0.0)

    def end_repeat(self) -> tuple[float, float, float]:
        """Return the payload-stage peaks of the repeat that just ended.

        The peaks are reset for the next repeat, and the monitor moves to the
        internal `between` stage until the next repeat starts.
        """
        with self._lock:
            self._sample_locked()
            peaks = (
                self._stage_peak["preprocess"],
                self._stage_peak["inference"],
                self._stage_peak["postprocess"],
            )
            for stage in _PAYLOAD_STAGES:
                self._stage_peak[stage] = 0.0
            self._stage = "between"
            return peaks

    def _run(self) -> None:
        try:
            while not self._stop.wait(_SAMPLE_INTERVAL_S):
                with self._lock:
                    self._sample_locked()
        except Exception as error:
            with self._lock:
                self._error = error
            self._stop.set()

    def _sample_locked(self) -> None:
        value = _working_set_gib(self._process)
        if value > self._run_peak:
            self._run_peak = value
        if value > self._stage_peak[self._stage]:
            self._stage_peak[self._stage] = value

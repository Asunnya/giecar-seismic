from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, TypedDict

import h5py
import numpy as np
import segyio

from giecar_seismic.application.butterworth_filter import apply_lowpass_filter
from giecar_seismic.infrastructure.segy.reader import SegyTraceReader

VALID_STRATEGIES = frozenset({"baseline", "naive", "streaming"})


class ChunkReader(Protocol):
    @property
    def trace_count(self) -> int: ...

    @property
    def sample_count(self) -> int: ...

    def read_chunk(self, start: int, stop: int) -> np.ndarray: ...


@dataclass(frozen=True)
class SegyMetadata:
    file_size_bytes: int
    trace_count: int
    sample_count: int
    sample_rate_ms: float
    dtype: str
    bytes_per_sample: int

    @property
    def nyquist_hz(self) -> float:
        return 500 / self.sample_rate_ms


@dataclass(frozen=True)
class BenchmarkResult:
    strategy: str
    trace_count: int
    sample_count: int
    input_size_mb: float
    chunk_size: int
    repetition: int
    peak_rss_mb: float
    elapsed_seconds: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class BenchmarkRecord(TypedDict):
    strategy: str
    trace_count: int
    peak_rss_mb: float
    additional_peak_mb: float
    elapsed_seconds: float


class AggregatedRecord(TypedDict):
    strategy: str
    trace_count: int
    median_peak_rss_mb: float
    median_additional_peak_mb: float
    median_elapsed_seconds: float


def validate_request(
    strategy: str,
    trace_count: int,
    chunk_size: int,
    *,
    file_trace_count: int,
) -> None:
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {sorted(VALID_STRATEGIES)}, got {strategy!r}"
        )
    if trace_count <= 0:
        raise ValueError("trace_count must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if trace_count > file_trace_count:
        raise ValueError(
            f"trace_count {trace_count} exceeds SEG-Y physical trace count "
            f"{file_trace_count}"
        )


def iter_trace_ranges(trace_count: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    if trace_count <= 0:
        raise ValueError("trace_count must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, trace_count, chunk_size):
        yield start, min(start + chunk_size, trace_count)


def peak_rss_to_mb(value: float, *, platform: str = sys.platform) -> float:
    """Convert ru_maxrss to MiB using the operating system's unit."""
    if platform.startswith("linux"):
        return value / 1024
    if platform == "darwin":
        return value / 1024**2
    raise ValueError(f"unsupported platform for ru_maxrss conversion: {platform}")


def _peak_working_set_bytes_windows() -> int:
    """Peak working set of this process via ``GetProcessMemoryInfo`` (psapi).

    The working set is Windows' resident-set counterpart, so
    ``PeakWorkingSetSize`` is the equivalent of ``ru_maxrss`` on POSIX.
    """
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    # ``WinDLL`` only exists in typeshed under sys.platform == "win32".
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    psapi = ctypes.WinDLL("psapi", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    ok = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    )
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]
    return int(counters.PeakWorkingSetSize)


def measure_peak_rss_mb(*, platform: str = sys.platform) -> float:
    """Peak resident memory of this process in MiB, per operating system.

    POSIX reports ``ru_maxrss`` (``resource`` is a POSIX-only stdlib module, so
    it is imported lazily: importing this module -- e.g. to unit-test the pure
    helpers -- must work on Windows too); Windows reports the peak working set.
    """
    if platform == "win32":
        return _peak_working_set_bytes_windows() / 1024**2
    try:
        import resource
    except ImportError as exc:
        raise ValueError(
            f"peak RSS measurement is not supported on platform {platform!r}"
        ) from exc
    return peak_rss_to_mb(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, platform=platform
    )


def read_segy_metadata(source_path: Path) -> SegyMetadata:
    if not source_path.is_file():
        raise FileNotFoundError(
            f"SEG-Y not found at {source_path}. Configure SOURCE_PATH explicitly."
        )
    with segyio.open(source_path, mode="r", ignore_geometry=True) as segy:
        dtype = np.dtype(segy.dtype)
        return SegyMetadata(
            file_size_bytes=source_path.stat().st_size,
            trace_count=segy.tracecount,
            sample_count=len(segy.samples),
            sample_rate_ms=float(segyio.tools.dt(segy)) / 1000,
            dtype=str(dtype),
            bytes_per_sample=dtype.itemsize,
        )


def _create_trace_dataset(
    output: h5py.File,
    *,
    trace_count: int,
    sample_count: int,
    chunk_size: int,
    output_dtype: np.dtype,
):
    return output.create_dataset(
        "traces",
        shape=(trace_count, sample_count),
        dtype=output_dtype,
        chunks=(min(chunk_size, trace_count), sample_count),
    )


def stream_filter_to_hdf5(
    reader: ChunkReader,
    output_path: Path,
    *,
    trace_count: int,
    chunk_size: int,
    cutoff_hz: float,
    order: int,
    sample_rate_ms: float,
    output_dtype: np.dtype,
) -> None:
    with h5py.File(output_path, "w") as output:
        traces = _create_trace_dataset(
            output,
            trace_count=trace_count,
            sample_count=reader.sample_count,
            chunk_size=chunk_size,
            output_dtype=output_dtype,
        )
        for start, stop in iter_trace_ranges(trace_count, chunk_size):
            chunk = reader.read_chunk(start, stop)
            filtered = apply_lowpass_filter(chunk, cutoff_hz, order, sample_rate_ms)
            traces[start:stop] = filtered
            output.flush()
            del chunk, filtered


def naive_filter_to_hdf5(
    reader: ChunkReader,
    output_path: Path,
    *,
    trace_count: int,
    chunk_size: int,
    cutoff_hz: float,
    order: int,
    sample_rate_ms: float,
    output_dtype: np.dtype,
) -> None:
    all_traces = reader.read_chunk(0, trace_count)
    all_filtered = apply_lowpass_filter(all_traces, cutoff_hz, order, sample_rate_ms)
    with h5py.File(output_path, "w") as output:
        traces = _create_trace_dataset(
            output,
            trace_count=trace_count,
            sample_count=reader.sample_count,
            chunk_size=chunk_size,
            output_dtype=output_dtype,
        )
        traces[:] = all_filtered
        output.flush()


def aggregate_medians(records: Sequence[BenchmarkRecord]) -> list[AggregatedRecord]:
    grouped: dict[tuple[str, int], list[BenchmarkRecord]] = {}
    for record in records:
        key = (record["strategy"], record["trace_count"])
        grouped.setdefault(key, []).append(record)

    aggregated: list[AggregatedRecord] = []
    for (strategy, trace_count), group in sorted(grouped.items()):
        aggregated.append(
            {
                "strategy": strategy,
                "trace_count": trace_count,
                "median_peak_rss_mb": statistics.median(
                    item["peak_rss_mb"] for item in group
                ),
                "median_additional_peak_mb": statistics.median(
                    item["additional_peak_mb"] for item in group
                ),
                "median_elapsed_seconds": statistics.median(
                    item["elapsed_seconds"] for item in group
                ),
            }
        )
    return aggregated


def run_benchmark(
    source_path: Path,
    output_path: Path | None,
    *,
    strategy: str,
    trace_count: int,
    chunk_size: int,
    cutoff_hz: float,
    order: int,
    repetition: int,
    max_input_mb: float,
) -> BenchmarkResult:
    metadata = read_segy_metadata(source_path)
    validate_request(
        strategy,
        trace_count,
        chunk_size,
        file_trace_count=metadata.trace_count,
    )
    if not 0 < cutoff_hz < metadata.nyquist_hz:
        raise ValueError(
            f"cutoff_hz must be below Nyquist ({metadata.nyquist_hz:g} Hz)"
        )
    input_size_mb = (
        trace_count * metadata.sample_count * metadata.bytes_per_sample / 1024**2
    )
    if input_size_mb > max_input_mb:
        raise ValueError(
            f"raw input estimate {input_size_mb:.1f} MB exceeds safety limit "
            f"{max_input_mb:.1f} MB"
        )
    if strategy != "baseline" and output_path is None:
        raise ValueError("output_path is required for naive and streaming strategies")

    started = time.perf_counter()
    reader = SegyTraceReader(source_path)
    try:
        if strategy == "naive":
            assert output_path is not None
            naive_filter_to_hdf5(
                reader,
                output_path,
                trace_count=trace_count,
                chunk_size=chunk_size,
                cutoff_hz=cutoff_hz,
                order=order,
                sample_rate_ms=metadata.sample_rate_ms,
                output_dtype=np.dtype(metadata.dtype),
            )
        elif strategy == "streaming":
            assert output_path is not None
            stream_filter_to_hdf5(
                reader,
                output_path,
                trace_count=trace_count,
                chunk_size=chunk_size,
                cutoff_hz=cutoff_hz,
                order=order,
                sample_rate_ms=metadata.sample_rate_ms,
                output_dtype=np.dtype(metadata.dtype),
            )
        # Baseline deliberately opens the same reader and materializes no amplitudes.
    finally:
        reader.close()
    elapsed_seconds = time.perf_counter() - started
    peak_rss_mb = measure_peak_rss_mb()
    return BenchmarkResult(
        strategy=strategy,
        trace_count=trace_count,
        sample_count=metadata.sample_count,
        input_size_mb=input_size_mb,
        chunk_size=chunk_size,
        repetition=repetition,
        peak_rss_mb=peak_rss_mb,
        elapsed_seconds=elapsed_seconds,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isolated peak-RSS benchmark")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--trace-count", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--cutoff", type=float, default=30.0)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--max-input-mb", type=float, default=256.0)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        result = run_benchmark(
            args.source,
            args.output,
            strategy=args.strategy,
            trace_count=args.trace_count,
            chunk_size=args.chunk_size,
            cutoff_hz=args.cutoff,
            order=args.order,
            repetition=args.repetition,
            max_input_mb=args.max_input_mb,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.error(str(exc))
    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

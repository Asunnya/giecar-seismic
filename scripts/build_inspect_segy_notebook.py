"""Generates notebooks/01_inspect_segy.ipynb from scratch (source of truth for review).

Run with: uv run python scripts/build_inspect_segy_notebook.py
Then execute with: uv run jupyter execute --inplace notebooks/01_inspect_segy.ipynb
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip() + "\n"))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip() + "\n"))


# ---------------------------------------------------------------------------
md(
    """
# SEG-Y Dataset Investigation

## 1. Objective

Characterize the provided SEG-Y dataset before implementing the production
streaming reader (`infrastructure/segy/`).

The investigation focuses on:

- physical file characteristics (size, trace count, sample geometry);
- sampling interval and Nyquist frequency;
- inline/crossline geometry and its irregularities;
- why `segyio`'s strict geometry inference fails on this survey;
- sequential trace access and bounded chunk reading;
- a preliminary, exploratory chunk-size benchmark.

**Scope boundary — read this before reusing any code from this notebook.**
Everything here is exploratory: it is allowed to load full header arrays
(inline/crossline for every trace, ~2 MiB for this file) to *investigate*
geometry, because that cost is proportional to header size, not to the
seismic *sample* volume. This is fine for a ~1 GiB file but does **not**
generalize — the same call scales as `O(n_traces)` in memory and is not
something the production reader should do for a 25 GB+ survey.

`inspect_segy()` below must **not** be lifted as-is into
`infrastructure/segy/reader.py`. The production reader needs to:

- extract only the metadata required to build `SeismicDataset` (trace
  headers, not full per-trace geometry arrays kept in memory at once);
- process trace *amplitudes* strictly in bounded chunks.

The dataset path here (`../data/survey.segy`) is also notebook-only. In the
application, the file path always comes from the user (`QFileDialog`) and
flows into the application layer — business/domain code never hardcodes a
path.
"""
)

# ---------------------------------------------------------------------------
md("## 2. Setup")

code(
    """
from pathlib import Path
from time import perf_counter

import numpy as np
import segyio
"""
)

code(
    """
DATASET_PATH = Path("../data/survey.segy")

assert DATASET_PATH.exists(), f"SEG-Y not found: {DATASET_PATH}"

file_size_gib = DATASET_PATH.stat().st_size / 1024**3

print(f"Path: {DATASET_PATH}")
print(f"Size: {file_size_gib:.3f} GiB")
"""
)

# ---------------------------------------------------------------------------
md(
    """
## 3. Helper functions

Defined once, up front, and reused by every section below — this notebook
computes each fact exactly once instead of re-deriving it per section.
"""
)

code(
    '''
def inspect_segy(dataset_path: Path) -> dict[str, object]:
    """Single pass over the SEG-Y file: every fact this notebook reports is
    computed here exactly once. Downstream cells only format/display pieces
    of this result — they do not recompute anything.

    NOTE: this function is exploratory. It keeps a full-length inline/crossline
    array in memory (O(n_traces), a couple MiB for this file) to investigate
    geometry. That is acceptable for interactive investigation but is exactly
    the pattern the production streaming reader must avoid for trace
    *amplitudes* (see the scope boundary note in section 1).
    """
    file_size_gib = dataset_path.stat().st_size / 1024**3

    with segyio.open(dataset_path, mode="r", ignore_geometry=True) as segy:
        trace_count = segy.tracecount
        samples_per_trace = len(segy.samples)

        sample_interval_us = float(segyio.tools.dt(segy))
        sample_interval_ms = sample_interval_us / 1000

        sampling_frequency_hz = 1000 / sample_interval_ms
        nyquist_frequency_hz = sampling_frequency_hz / 2

        first_trace = np.asarray(segy.trace[0]).copy()
        sample_dtype = first_trace.dtype
        bytes_per_sample = sample_dtype.itemsize

        inline_values = np.asarray(
            segy.attributes(segyio.TraceField.INLINE_3D)[:]
        )
        crossline_values = np.asarray(
            segy.attributes(segyio.TraceField.CROSSLINE_3D)[:]
        )

    unique_inlines = np.unique(inline_values)
    unique_crosslines = np.unique(crossline_values)

    geometry_pairs = np.column_stack((inline_values, crossline_values))
    unique_geometry_pairs, pair_counts = np.unique(
        geometry_pairs, axis=0, return_counts=True
    )

    duplicated_grid_positions = int(np.count_nonzero(pair_counts > 1))

    # Occupancy over the *bounding* Cartesian grid (min..max on each axis).
    # A position being "missing" here means the survey footprint is
    # irregular, NOT that a trace was lost/corrupted (see section 5).
    inline_min = int(unique_inlines.min())
    crossline_min = int(unique_crosslines.min())

    occupancy = np.zeros(
        (len(unique_inlines), len(unique_crosslines)), dtype=bool
    )
    occupancy[inline_values - inline_min, crossline_values - crossline_min] = True

    expected_grid_positions = occupancy.size
    observed_grid_positions = int(occupancy.sum())
    missing_grid_positions = expected_grid_positions - observed_grid_positions

    missing_indexes = np.argwhere(~occupancy)
    missing_pairs = [
        (int(i) + inline_min, int(j) + crossline_min)
        for i, j in missing_indexes
    ]

    inline_ids, traces_per_inline = np.unique(inline_values, return_counts=True)
    expected_traces_per_inline = len(unique_crosslines)
    incomplete_mask = traces_per_inline != expected_traces_per_inline
    incomplete_inlines = [
        (int(inline), int(count), int(expected_traces_per_inline - count))
        for inline, count in zip(
            inline_ids[incomplete_mask], traces_per_inline[incomplete_mask]
        )
    ]

    expected_order = np.lexsort((crossline_values, inline_values))
    is_inline_major = np.array_equal(np.arange(trace_count), expected_order)

    strict_geometry_supported = True
    strict_geometry_error = None
    try:
        with segyio.open(dataset_path, mode="r"):
            pass
    except (ValueError, RuntimeError) as exc:
        strict_geometry_supported = False
        strict_geometry_error = f"{type(exc).__name__}: {exc}"

    return {
        "file_size_gib": file_size_gib,
        "trace_count": trace_count,
        "samples_per_trace": samples_per_trace,
        "sample_dtype": str(sample_dtype),
        "bytes_per_sample": bytes_per_sample,
        "sample_interval_ms": sample_interval_ms,
        "sampling_frequency_hz": sampling_frequency_hz,
        "nyquist_frequency_hz": nyquist_frequency_hz,
        "inline_values": inline_values,
        "crossline_values": crossline_values,
        "n_inlines": len(unique_inlines),
        "inline_min": inline_min,
        "inline_max": int(unique_inlines.max()),
        "n_crosslines": len(unique_crosslines),
        "crossline_min": crossline_min,
        "crossline_max": int(unique_crosslines.max()),
        "expected_grid_positions": expected_grid_positions,
        "observed_grid_positions": observed_grid_positions,
        "missing_grid_positions": missing_grid_positions,
        "duplicated_grid_positions": duplicated_grid_positions,
        "missing_pairs": missing_pairs,
        "incomplete_inlines": incomplete_inlines,
        "is_inline_major": is_inline_major,
        "strict_geometry_supported": strict_geometry_supported,
        "strict_geometry_error": strict_geometry_error,
    }


def print_segy_summary(summary: dict[str, object]) -> None:
    ordering = (
        "inline-major, then crossline"
        if summary["is_inline_major"]
        else "not inline-major"
    )
    geometry_status = (
        "supported" if summary["strict_geometry_supported"] else "not supported"
    )

    print(
        f"""
SEG-Y Dataset Summary
---------------------
File size:              {summary["file_size_gib"]:.3f} GiB
Trace count:            {summary["trace_count"]}
Samples per trace:      {summary["samples_per_trace"]}
Sample dtype:           {summary["sample_dtype"]} ({summary["bytes_per_sample"]} bytes/sample)

Sample interval:        {summary["sample_interval_ms"]:.3f} ms
Sampling frequency:     {summary["sampling_frequency_hz"]:.3f} Hz
Nyquist frequency:      {summary["nyquist_frequency_hz"]:.3f} Hz

Unique inlines:         {summary["n_inlines"]}
Inline range:           {summary["inline_min"]} .. {summary["inline_max"]}

Unique crosslines:      {summary["n_crosslines"]}
Crossline range:        {summary["crossline_min"]} .. {summary["crossline_max"]}

Bounding grid size:              {summary["expected_grid_positions"]:>10}
Occupied Cartesian positions:    {summary["observed_grid_positions"]:>10}
Unoccupied Cartesian positions:  {summary["missing_grid_positions"]:>10}
Duplicated (inline, crossline):  {summary["duplicated_grid_positions"]:>10}

Trace ordering:         {ordering}
Strict geometry:        {geometry_status}
""".strip()
    )

    if summary["strict_geometry_error"]:
        print(f"\\nStrict geometry error:\\n{summary['strict_geometry_error']}")


def read_trace_chunk(
    segy: segyio.SegyFile,
    start: int,
    stop: int,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    """Reads traces [start, stop) into one array.

    `dtype` is intentionally a parameter, not a hardcoded assumption: this
    generic helper infers it from the file when the caller doesn't pin one.
    (The production reader may instead decide the internal domain dtype is
    always float32 and document that decision explicitly — but that is a
    deliberate choice made once, not something this helper should assume.)
    """
    if start < 0:
        raise ValueError("start must be non-negative")
    if stop <= start:
        raise ValueError("stop must be greater than start")
    if stop > segy.tracecount:
        raise ValueError("stop exceeds trace count")

    sample_count = len(segy.samples)
    resolved_dtype = dtype if dtype is not None else np.asarray(segy.trace[start]).dtype

    chunk = np.empty((stop - start, sample_count), dtype=resolved_dtype)
    for row_index, trace_index in enumerate(range(start, stop)):
        chunk[row_index] = segy.trace[trace_index]

    return chunk


def benchmark_chunk_reading(
    dataset_path: Path, chunk_size: int, trace_count: int
) -> dict[str, float | int]:
    processed_traces = 0
    started_at = perf_counter()

    with segyio.open(dataset_path, mode="r", ignore_geometry=True) as segy:
        stop_limit = min(trace_count, segy.tracecount)
        for start in range(0, stop_limit, chunk_size):
            stop = min(start + chunk_size, stop_limit)
            chunk = read_trace_chunk(segy, start, stop)
            processed_traces += len(chunk)

    elapsed_seconds = perf_counter() - started_at
    return {
        "chunk_size": chunk_size,
        "processed_traces": processed_traces,
        "elapsed_seconds": elapsed_seconds,
        "traces_per_second": processed_traces / elapsed_seconds,
    }
'''
)

# ---------------------------------------------------------------------------
md(
    """
## 4. Consolidated dataset summary

One call, one pass over the file's headers — every later section reads from
`dataset_summary` instead of reopening/rescanning the file.
"""
)

code(
    """
dataset_summary = inspect_segy(DATASET_PATH)
print_segy_summary(dataset_summary)
"""
)

# ---------------------------------------------------------------------------
md(
    """
## 5. Geometry investigation

`segyio`'s strict geometry inference (`segyio.open(path)` without
`ignore_geometry=True`) fails on this file:

```text
ValueError: Inlines inconsistent, expect all inlines to be unique
```

Separately, the header scan above shows the survey's inline/crossline
footprint is irregular: 26 positions in the `401 × 720` bounding grid have no
trace. **These are two independently-observed facts, not a proven causal
chain** — this notebook does not trace `segyio`'s internal inference code to
confirm that these specific 26 gaps are *the* reason for that specific
`ValueError`. The correct, defensible statement is:

> The survey contains an irregular Cartesian footprint, and `segyio`'s
> strict geometry inference fails. Processing therefore uses raw sequential
> trace access with `ignore_geometry=True`.

Also note the semantics: "26 missing grid positions" means 26 unoccupied
`(inline, crossline)` combinations in the bounding rectangle — it does
**not** mean 26 traces were lost or corrupted. An irregular survey footprint
(non-rectangular acquisition geometry) is a legitimate, common outcome in
real seismic surveys.
"""
)

code(
    """
print(f"Duplicated (inline, crossline) pairs: {dataset_summary['duplicated_grid_positions']}")
print(f"Unoccupied grid positions: {dataset_summary['missing_grid_positions']}\\n")

print("Unoccupied (inline, crossline) positions:")
for inline, crossline in dataset_summary["missing_pairs"]:
    print(f"  inline={inline}, crossline={crossline}")
"""
)

code(
    """
print(f"Expected traces per complete inline: {dataset_summary['n_crosslines']}\\n")

print(f"Incomplete inlines: {len(dataset_summary['incomplete_inlines'])}")
for inline, count, missing_count in dataset_summary["incomplete_inlines"]:
    print(f"  inline={inline}: {count} traces, {missing_count} missing")
"""
)

# ---------------------------------------------------------------------------
md(
    """
## 6. Streaming investigation

Chunk memory footprint uses the dtype and per-sample byte size *measured*
from the file (`dataset_summary["bytes_per_sample"]`), not an assumed
`float32` constant — the same file could in principle store a different
sample format.
"""
)

code(
    """
samples_per_trace = dataset_summary["samples_per_trace"]
bytes_per_sample = dataset_summary["bytes_per_sample"]

chunk_sizes = [64, 128, 256, 512, 1024]

for chunk_size in chunk_sizes:
    memory_mib = (chunk_size * samples_per_trace * bytes_per_sample) / 1024**2
    print(f"{chunk_size:4d} traces -> {memory_mib:.3f} MiB")
"""
)

code(
    """
CHUNK_SIZE = 256

with segyio.open(DATASET_PATH, mode="r", ignore_geometry=True) as segy:
    chunk = read_trace_chunk(segy, start=0, stop=CHUNK_SIZE)

print(f"Shape: {chunk.shape}")
print(f"dtype: {chunk.dtype}")
print(f"Memory: {chunk.nbytes / 1024**2:.3f} MiB")
"""
)

code(
    """
# Correctness check: chunked reads must be byte-identical to direct trace access.
with segyio.open(DATASET_PATH, mode="r", ignore_geometry=True) as segy:
    chunk = read_trace_chunk(segy, start=0, stop=3)
    direct_traces = [np.asarray(segy.trace[i]).copy() for i in range(3)]

for i, direct_trace in enumerate(direct_traces):
    print(f"trace {i} matches direct access:", np.array_equal(chunk[i], direct_trace))

print("trace 0 equals trace 1:", np.array_equal(chunk[0], chunk[1]))
"""
)

# ---------------------------------------------------------------------------
md(
    """
## 7. Preliminary benchmark

**This benchmark is exploratory only.** Elapsed time is not deterministic
and may be affected by the operating system's page cache (the same 16,384
traces are re-read for every chunk size in this run) and by system load.
The result is used to rule out obviously poor chunk sizes, not to prove an
optimal value — a conclusive benchmark needs to run the full
read → filter → HDF5-write pipeline, once that exists.
"""
)

code(
    """
TRACE_COUNT_TO_BENCHMARK = 16_384

benchmark_results = []
for chunk_size in chunk_sizes:
    result = benchmark_chunk_reading(DATASET_PATH, chunk_size, TRACE_COUNT_TO_BENCHMARK)
    benchmark_results.append(result)
    print(
        f"chunk={result['chunk_size']:4d} | "
        f"traces={result['processed_traces']:5d} | "
        f"time={result['elapsed_seconds']:.4f}s | "
        f"rate={result['traces_per_second']:.0f} traces/s"
    )
"""
)

# ---------------------------------------------------------------------------
md(
    """
## 8. Findings and architectural consequences

**Findings** (all reproduced by section 4's single `inspect_segy()` pass):
the file has 288,694 real traces, 850 samples/trace, `float32` samples, a
4 ms sample interval (`f_s = 250 Hz`, Nyquist = 125 Hz); 401 inlines and 720
crosslines with no duplicate `(inline, crossline)` pairs but 26 unoccupied
grid positions (an irregular footprint, not lost data); trace order is
inline-major; and chunked reads are byte-identical to direct trace access.

**Consequences for the production design:**

- `segyio.open(..., ignore_geometry=True)` — processing must not depend on a
  regular inline/crossline grid.
- The *actual* `trace_count` (not `n_inlines * n_crosslines`) is the source
  of truth for progress reporting — the two differ by design for irregular
  surveys.
- Streaming must address traces by physical sequential index, not by
  inline/crossline coordinates, to keep memory bounded by chunk size alone.
- `sampleRateMs` must be read from each imported SEG-Y and the Nyquist limit
  (for `create_filter_job`'s `cutoff_hz` validation) computed dynamically —
  never hardcoded to this survey's 125 Hz.
"""
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": ".venv",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.12.14",
    },
}

with open("notebooks/01_inspect_segy.ipynb", "w") as f:
    nbf.write(nb, f)

print("wrote notebooks/01_inspect_segy.ipynb with", len(cells), "cells")

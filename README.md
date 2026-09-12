# GIECAR Seismic

Python/PyQt5 desktop application for streaming Butterworth filtering of SEG-Y
surveys into HDF5, with SQLite job history and a seismic comparison viewer.

## Development

Requires Python 3.12+ and uv:

```bash
uv sync
uv run giecar-seismic
uv run pytest
uv run ruff check .
uv run mypy src/giecar_seismic
```

## Streaming and memory

The application has two bounded streaming paths:

- **Import:** SEG-Y trace headers are read in batches of 4096 to derive metadata.
  Each batch updates sets of distinct inline and crossline identifiers and is then
  discarded. Import uses `segyio.open(..., ignore_geometry=True)`, takes the
  physical trace count from `segy.tracecount`, and never reads trace amplitudes.
- **Processing:** SEG-Y amplitudes are read in configurable bounded chunks,
  filtered with Butterworth, and written incrementally to HDF5.

Import working memory is
`O(batch_size + unique_inlines + unique_crosslines)`: temporary header arrays are
bounded by the batch size, while the exact distinct counts require retaining only
the geometric identifiers seen. Filtering working memory is
`O(chunk_size * n_samples)`. Neither path materializes the complete seismic volume.

## Butterworth filters

The original challenge's **Low-pass remains the default**. Existing calls to
`service.create_filter_job(dataset_id, cutoff_hz, order)` and
`apply_lowpass_filter(chunk, cutoff_hz, order, sample_rate_ms)` remain valid.
Optional `filter_type` and `upper_cutoff_hz` arguments are keyword-only on the
service and the generic `apply_butterworth_filter` operation.

| Domain `FilterType` | Meaning | `cutoff_hz` | `upper_cutoff_hz` |
| --- | --- | --- | --- |
| `LOW_PASS` | High-cut: removes high frequencies | High cutoff | `None` |
| `HIGH_PASS` | Low-cut: removes low frequencies | Low cutoff | `None` |
| `BAND_PASS` | Keeps the intermediate band | Low cutoff | High cutoff |

The application validates `fs = 1000 / sample_rate_ms`, `Nyquist = fs / 2`,
integer order 2–8, and `0 < cutoff < Nyquist`. Band-pass additionally requires
`cutoff_hz < upper_cutoff_hz < Nyquist`; other types reject an upper cutoff.
The UI adjusts labels and limits, but the service remains the validation boundary.

`butterworth_sos` is the shared SciPy `butter(..., output="sos")` design used by
processing and theoretical response. `sosfiltfilt(..., axis=-1)` preserves the
existing zero-phase, independent-per-trace behavior. For Band-pass, the entered
order is the **prototype order** passed to SciPy: the resulting single-pass
transfer function has order `2 * order`. Forward/backward filtering squares the
single-pass magnitude response for every filter type.

Processing still reads bounded chunks, filters, writes HDF5 incrementally, reports
progress and observes cooperative cancellation at existing boundaries. Job
finalization, partial-output handling and geometry indexing retain their existing
behavior. Resume uses the durable chunk checkpoints described below; multiprocessing
is not implemented.

## Spectrum and viewer

The viewer uses the selected job's persisted configuration. Both Matplotlib and
PyQtGraph receive the same application-layer `TraceSpectrum` data and labeled
cutoff markers. The frequency axis is always linear, **0–Nyquist Hz**. For odd
sample counts, the last actual FFT bin lies below Nyquist; bins are not relabeled
or interpolated to invent a Nyquist sample.

- **Linear:** unnormalized `abs(rfft(trace))`, preserving the existing magnitude.
- **dB:** `20 * log10(magnitude / reference)`, with the **original selected
  spectrum's peak as the common reference for both curves**. Original peak is
  0 dB; attenuation of the filtered trace remains visible. An all-zero original
  uses reference 1. Computation is safe in log space with a **−120 dB floor**.
- **Show filter response:** off by default. An optional green curve on a separate
  right Y axis shows ideal zero-phase gain `abs(sosfreqz(SOS))**2`, using the same
  design and job parameters as processing. Linear gain is relative to unity;
  dB gain uses that same unity reference and −120 dB floor. This theoretical
  steady-state response does not model finite-trace edge transients.

Scale/response toggles transform cached magnitudes only. Renderer switches reuse
those display data. Neither operation rereads SEG-Y/HDF5, repeats the FFT, or
starts a worker. The renderers only draw scientific data supplied by the application.

### Detached Spectrum window

Select a trace, then click **Open Spectrum**. The modeless 1100×760 window
can remain open while selecting other traces. Repeated clicks bring the same
window forward; each viewer owns at most one. Closing it releases its renderer
and disconnects signals. Closing the viewer also closes its spectrum window;
the existing guard against closing during a section load still applies.

The existing application `TraceSpectrum` is the shared spectrum data object:

```text
application → TraceSpectrum → SeismicViewer → embedded spectrum
                                          → SpectrumWindow
```

Opening the window passes the exact current display object, including the
already computed response. It calls no service/repository, reads no SEG-Y/HDF5,
performs no Butterworth/FFT calculation and starts no worker. Scale conversions
remain in the application function `spectrum_for_display`; both presentations
receive the same result. Scientific processing is unchanged.

**State policy:** Linear/dB belongs to `SeismicViewer` and stays synchronized in
both directions, including while no trace is selected. Renderer choice and
Original/Filtered/Filter response visibility are local to the detached window;
initial renderer/response follow the embedded view. Hiding all curves produces
an explicit empty-state message. Closing/reopening resets these local controls.

Navigation clears the selected spectrum and disables Open Spectrum until another
trace is selected. The open window shows a selection prompt. The existing single
in-flight section load remains; request identities reject stale/duplicate results,
and clicks on the old section are ignored during loading.

A minimal `SpectrumRenderer` is reused by **both** embedded and detached views.
Matplotlib reuses its `Line2D` artists and provides a native navigation toolbar;
PyQtGraph reuses its items with `setData()` and provides native zoom/pan. Only
presentation is performed on the GUI thread. Response gain retains its separate
right axis; frequency remains linear.

Interactive smoke on a Linux graphical session, with temporary real SEG-Y/SQLite/
HDF5 data (12 traces × 1024 samples), exercised all three filters, both renderers,
resize, native zoom/pan, visibility, scale synchronization and live trace updates.
Observed click-to-first-completed-Qt-paint times: **PyQtGraph 45–52 ms** (three
openings), **Matplotlib 96 ms** (one opening). These are local observations, not a
pytest benchmark or a guarantee for other machines.

## Resuming a cancelled job (creativity track)

Select a **CANCELLED** job in the history and click **Resume**. Eligibility uses
its saved trace count and the presence of its output file, without opening HDF5
on the GUI thread. The button requires `processed_traces < dataset.n_traces`.
Run and Resume share the existing QThread lifecycle and are mutually exclusive;
Cancel remains cooperative. Progress starts from the saved value and is then
reconciled by the worker. The service can also finish a fully checkpointed
cancelled job even though the history button is intentionally disabled at 100%.

`resume_filter_job(job_id, progress_callback, cancel_token)` explicitly continues
the **same** job and output path. `run_filter_job()` remains initial execution.
The state machine is `CREATED -> RUNNING -> COMPLETED / FAILED / CANCELLED`,
plus the creativity track's single additional transition:
`CANCELLED --resume()--> RUNNING`. FAILED and COMPLETED cannot resume. There is
no automatic recovery/transition of a stranded RUNNING job after a process crash.
`created_at` and the original `started_at` are preserved; resume increments
`resume_count`, clears `finished_at`, and subsequent termination records a new end.

### Durable checkpoint and reconciliation

Each chunk follows this order:

`read -> Butterworth -> write_chunk -> HDF5 checkpoint/flush -> processed_traces
-> progress -> SQLite commit -> progress callback`.

`processed_traces` is an exact integer prefix length; percentage is derived as
`100 * processed_traces / dataset.n_traces`. It never locates a checkpoint.
The HDF5 `traces` shape is `(written_trace_count, n_samples)`, with attributes
`expected_trace_count`, `n_samples`, `written_trace_count`, and `complete`.
`checkpoint()` flushes amplitudes and the prefix metadata before SQLite advances.
This provides reopen visibility, not an atomic distributed transaction or a
power-loss guarantee beyond HDF5/filesystem behavior.

Resume opens the same HDF5 in **r+**, validates shape/counts/completeness, and reads
only metadata. It never copies, refilters or rewrites the existing prefix.
The SEG-Y is reopened and checked against the imported size/mtime fingerprint,
physical trace count, sample count and interval. Missing fingerprints are rejected;
size/mtime detects ordinary replacement but is not a cryptographic identity check.

- **HDF5 = SQLite:** continue at that trace index.
- **HDF5 > SQLite:** reconcile SQLite and progress to the durable HDF5 prefix.
- **SQLite > HDF5**, malformed/missing output, or changed source: reject explicitly
  before entering RUNNING. `CheckpointInconsistentError` identifies unsafe output
  checkpoints; source validation has separate clear errors. No automatic truncation.
- **Full prefix, complete=False:** finalize and complete without filtering.
- **Full prefix, complete=True:** reconcile the CANCELLED job to COMPLETED without
  writing/finalizing the file again. Incomplete data marked complete is rejected.

The loop starts at `written_trace_count`, retaining configurable chunk size,
bounded memory, execution registration and the existing point of no return.
A resumed execution can be cancelled again. Runtime processing/checkpoint failures
still produce FAILED; preflight rejection leaves CANCELLED unchanged.

### Decision record

**Context:** Seismic processing may take hours and datasets can exceed 25 GB.
Restarting from zero wastes CPU and I/O; incremental writes already provide a
natural checkpoint boundary.

**Decision:** Choose resume as the creativity track: continue the same CANCELLED
job from its last durably stored physical trace, with HDF5 as prefix authority.

**Alternatives:** Restarting is simpler but repeats expensive work. Percentage
progress loses exact trace boundaries through rounding and says nothing about
whether amplitudes were flushed. A distributed transaction or attempts/event
model would add unnecessary scope for this challenge.

**Trade-offs:** Flush each chunk, accept its I/O cost, and explicitly reconcile two
stores. Reject ambiguous/corrupt checkpoints instead of guessing or rolling back.

**Consequences:** Repeated cancellation/resume preserves work with bounded memory
and numerically identical output; schema changes and consistency tests are required.

### Local checkpoint cost measurement

A manual pipeline benchmark compared the preceding implementation with this one:
real SEG-Y -> Butterworth -> HDF5, plus SQLite writes; 4096 traces x 2048 samples,
256 traces/chunk (16 checkpoints). After one warm-up pair, five alternating runs
had medians **190.6 ms before** and **189.3 ms after** (-0.7%). The observed cost
was indistinguishable from local timing/page-cache variation, not evidence that
flush is free on larger datasets or slower disks. No performance threshold is
asserted by pytest.

## Development database schema change

Jobs additionally persist non-null integer `processed_traces` and `resume_count`,
both defaulting to zero. Jobs also persist `filter_type` as the enum **name** (`LOW_PASS`, `HIGH_PASS`,
`BAND_PASS`) and nullable `upper_cutoff_hz`. New/default jobs use `LOW_PASS` and
`NULL`. `create_schema()` only creates missing tables; it does **not** migrate
existing ones. No database is deleted or updated automatically.

**Before a manual smoke test, consciously update or recreate the development
database** at `~/.giecar-seismic/giecar.sqlite`. Close the app and back up any
history you need first. For a database matching the immediately preceding schema,
a reviewed manual update is:

```sql
ALTER TABLE jobs ADD COLUMN filter_type VARCHAR NOT NULL DEFAULT 'LOW_PASS';
ALTER TABLE jobs ADD COLUMN upper_cutoff_hz FLOAT;
ALTER TABLE jobs ADD COLUMN processed_traces INTEGER NOT NULL DEFAULT 0;
ALTER TABLE jobs ADD COLUMN resume_count INTEGER NOT NULL DEFAULT 0;
```

Apply this only if those columns are absent. Older schema differences require
separate review. Never backfill `processed_traces` from a percentage: eligible old
partial files must pass HDF5/source validation and reconcile their physical prefix.
Alternatively, deliberately move the old development database
aside and let startup create a fresh one; existing HDF5 outputs are not removed.

## Validation

Tests retain the original Low-pass suite and add frequency preservation/attenuation
for all three types, invalid parameters, three small real SEG-Y→HDF5 round trips,
persistence, common-reference dB/floor behavior, theoretical response, dynamic UI,
and equivalent renderer data. UI tests run headlessly via the shared Qt fixtures;
assertions inspect behavior and data, not pixels.

Resume tests additionally cover timestamp/state transitions, durable prefix
visibility, metadata rejection, exact read/filter/write boundaries, repeated
cancellation, concurrent execution rejection, both final-chunk crash windows,
HDF5/SQLite divergence, source replacement, and GUI worker routing. The real
irregular SEG-Y E2E closes/reopens SQLite and HDF5 between attempts and compares
the final amplitudes both to the scientific function and an uninterrupted pipeline.

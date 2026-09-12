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
behavior. No resume or multiprocessing is implemented.

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

## Development database schema change

Jobs now persist `filter_type` as the enum **name** (`LOW_PASS`, `HIGH_PASS`,
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
```

Apply this only if those columns are absent. Older schema differences require
separate review. Alternatively, deliberately move the old development database
aside and let startup create a fresh one; existing HDF5 outputs are not removed.

## Validation

Tests retain the original Low-pass suite and add frequency preservation/attenuation
for all three types, invalid parameters, three small real SEG-Y→HDF5 round trips,
persistence, common-reference dB/floor behavior, theoretical response, dynamic UI,
and equivalent renderer data. UI tests run headlessly via the shared Qt fixtures;
assertions inspect behavior and data, not pixels.

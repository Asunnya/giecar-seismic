# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This is the deliverable for a GIECAR/UFF technical assessment ("Bolsista de Pesquisa e
Desenvolvimento — Python/PyQt5"): a PyQt5 desktop app that applies a streaming Butterworth
low-pass filter to SEG-Y seismic surveys. Traces are read via `segyio`, filtered, and written
incrementally to HDF5/Zarr; job/dataset metadata is persisted to SQLite via SQLAlchemy.

**Before doing any architecture, implementation, review, debugging, or test work here, invoke
the `giecar-seismic-engineering` skill.** It contains the full domain model (`SeismicDataset`,
the `Job` state machine), the required business API contract
(`create_filter_job`/`run_filter_job`/`cancel_job`/`get_job_status`/`list_jobs`), the
non-negotiable constraints (bounded memory, cooperative cancellation, responsive UI,
business/UI separation), the PyQt5 threading checklist, testing strategy, and the code-review
protocol. This CLAUDE.md intentionally does not repeat that content — treat the skill as the
source of truth for domain/architecture decisions, and this file as the source of truth for
day-to-day commands and repo layout.

The original assignment PDF is `Prova_Tecnica_GIECAR_UFF.pdf`; a text copy also lives at
`~/.claude/skills/giecar-seismic-engineering/reference/challenge-spec.md`.

## Commands

Dependency management is via `uv`; a venv already exists at `.venv`.

```bash
uv sync                          # install/sync dependencies (incl. dev group)
uv run python -m giecar_seismic  # run the app (once an entry point exists)
uv run pytest                    # run the full test suite
uv run pytest src/tests/unit     # unit tests only
uv run pytest path/to/test_file.py::test_name   # run a single test
uv run pytest --cov              # with coverage (pytest-cov is installed)
uv run ruff check .              # lint
uv run ruff format .             # format
uv run mypy src                  # type-check
```

There is no test file yet under `src/tests/unit` or `src/tests/integration` — those
directories currently only exist as empty scaffolding.

## Architecture

Target layout (per the skill's reference architecture — not all of it exists yet):

```
src/giecar_seismic/
  domain/          entities, Job states, validation rules
  application/     use cases: job orchestration, filtering workflow, cancellation
  infrastructure/
    segy/          streaming SEG-Y reader (segyio)
    storage/       HDF5/Zarr trace writer
    database/      SQLAlchemy models/repositories for SeismicDataset + Job metadata
  ui/              PyQt5 windows/widgets, presenters, worker (QThread/QRunnable) glue
src/tests/
  unit/
  integration/
```

The business layer (`domain/` + `application/`) must be importable and testable with plain
`pytest` — no `QApplication` instantiated. Qt code in `ui/` talks to it only through
signals/callbacks, never the reverse.

The `domain/`, `application/`, `infrastructure/{segy,storage,database}/`, and `ui/` packages
above exist as empty scaffolding (each just an `__init__.py`) — no business logic, SEG-Y
reading, persistence, or Qt code has been implemented yet.

## Data files

- `volve_psdm_full_time.segy` — a large (~1GB) real SEG-Y file (Volve dataset) for local manual
  testing, and `Prova_Tecnica_GIECAR_UFF.pdf` — the assignment PDF. Both are gitignored; do not
  remove them from `.gitignore` or commit them.
- The assessment explicitly requires handling SEG-Y files far larger than this local sample
  (25GB+) without ever loading the full volume into memory — see the skill's "bounded memory"
  constraint before writing any read/write path.

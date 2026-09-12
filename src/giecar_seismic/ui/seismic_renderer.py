"""Renderer contract for the 2D seismic viewer, plus the display helpers
both renderers share so they can never disagree on *what* is drawn.

The viewer owns state (orientation, line, mode, gain, clip, colormap,
wiggle, selected trace, workers); a renderer owns only widgets and how
the already-loaded SeismicSection / TraceView / TraceSpectrum are drawn.
Renderers never read SEG-Y/HDF5, never query a repository and never run
off the GUI thread. Switching renderer therefore never touches data.
"""

from dataclasses import dataclass
from math import ceil

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QWidget

from giecar_seismic.application.seismic_viewer import (
    SeismicSection,
    TraceSpectrum,
    TraceView,
)
from giecar_seismic.domain.geometry import LineOrientation

DISPLAY_MODES = ["Original", "Filtered", "Difference", "Side-by-side"]
# Exposed to the user by their matplotlib names; the PyQtGraph renderer
# converts them through pyqtgraph.colormap.getFromMatplotlib, so the two
# libraries always offer the same list from this single definition.
COLORMAPS = ["seismic", "gray", "RdBu_r", "viridis"]
RENDERERS = ["Matplotlib", "PyQtGraph"]
# Wiggle draws at most this many traces per panel; beyond it, only every
# k-th trace is drawn (display-only decimation -- the section is untouched).
MAX_WIGGLE_TRACES = 200


@dataclass(frozen=True)
class DisplaySettings:
    mode: str
    wiggle: bool
    gain: float
    clip_percentile: float
    colormap: str


@dataclass(frozen=True)
class PanelRender:
    """What a renderer drew for one panel -- an inspectable, library-free
    record used by tests to check equivalence without rasterizing."""

    title: str
    shape: tuple[int, int]
    extent: tuple[float, float, float, float]  # x0, x1, t_max, t_min
    levels: tuple[float, float]
    coordinates: tuple[int, ...]
    wiggle: bool


class SeismicRenderer(QObject):
    """Minimal interface the viewer needs. Two concrete implementations:
    MatplotlibSeismicRenderer and PyQtGraphSeismicRenderer."""

    # Geometric x coordinate the user clicked on the section (crossline
    # number in inline view, inline number in crossline view). Resolving
    # it to a physical trace stays in the viewer/service, never here.
    coordinate_clicked = pyqtSignal(float)

    def section_widget(self) -> QWidget:
        raise NotImplementedError

    def analysis_widget(self) -> QWidget:
        """Trace overlay + amplitude spectrum, stacked."""
        raise NotImplementedError

    def show_section(
        self, section: SeismicSection | None, settings: DisplaySettings
    ) -> None:
        raise NotImplementedError

    def show_trace(
        self, view: TraceView | None, spectrum: TraceSpectrum | None
    ) -> None:
        raise NotImplementedError

    def dispose(self) -> None:
        """Release widgets and disconnect anything that could keep drawing."""
        raise NotImplementedError

    # Filled in by implementations after each show_section(); exposed for
    # tests and for the manual rendering-time measurement.
    last_panels: list[PanelRender]
    last_render_seconds: float


# --- shared display math ----------------------------------------------------


def amplitude_limit(
    section: SeismicSection, clip_percentile: float, gain: float
) -> float:
    """Symmetric colour/wiggle scale shared by every panel of a section.

    The limit is the `clip_percentile` of |amplitude| over the *present*
    samples of original and filtered together, divided by `gain`. One
    limit for original, filtered, difference and both halves of
    side-by-side keeps the comparison honest -- panels are never
    normalized independently.
    """
    values = np.concatenate(
        [
            section.original[section.present_mask].ravel(),
            section.filtered[section.present_mask].ravel(),
        ]
    )
    values = np.abs(values[np.isfinite(values)])
    if values.size == 0:
        return 1.0
    limit = float(np.percentile(values, clip_percentile))
    if limit <= 0:
        limit = float(values.max()) or 1.0
    return limit / max(gain, 1e-6)


def wiggle_stride(n_present_traces: int, max_traces: int = MAX_WIGGLE_TRACES) -> int:
    return max(1, ceil(n_present_traces / max_traces))


def panels_for_mode(section: SeismicSection, mode: str) -> list[tuple[str, np.ndarray]]:
    """(title, data) per panel for a display mode. `data` is a view of the
    section's own arrays -- never a copy -- so both renderers draw the
    very same memory."""
    line = f"{section.orientation.value} {section.line_number}"
    if mode == "Side-by-side":
        return [
            (f"Original -- {line}", section.original),
            (f"Filtered -- {line}", section.filtered),
        ]
    if mode == "Filtered":
        return [(f"Filtered -- {line}", section.filtered)]
    if mode == "Difference":
        return [(f"Filtered - Original -- {line}", section.difference)]
    return [(f"Original -- {line}", section.original)]


def section_extent(section: SeismicSection) -> tuple[float, float, float, float]:
    """(x0, x1, t_max, t_min): half a coordinate step of padding on each
    side so each trace is centred on its coordinate; time increases
    downwards, hence t_max first."""
    coords = section.coordinates
    step = coordinate_step(section)
    return (
        float(coords[0]) - step / 2,
        float(coords[-1]) + step / 2,
        float(section.time_ms[-1]),
        0.0,
    )


def coordinate_step(section: SeismicSection) -> float:
    coords = section.coordinates
    return 1.0 if len(coords) < 2 else float(coords[1] - coords[0])


def x_axis_label(section: SeismicSection) -> str:
    return "Crossline" if section.orientation is LineOrientation.INLINE else "Inline"


def wiggle_traces(
    section: SeismicSection, data: np.ndarray, limit: float
) -> list[tuple[float, np.ndarray]]:
    """(x coordinate, scaled trace) for the traces a wiggle panel draws:
    each scaled so an amplitude of `limit` spans half a coordinate step,
    clipped to that, decimated by wiggle_stride() beyond the budget."""
    present = np.flatnonzero(section.present_mask)
    stride = wiggle_stride(len(present))
    scale = 0.5 * coordinate_step(section) / limit
    return [
        (float(section.coordinates[p]), np.clip(data[p], -limit, limit) * scale)
        for p in present[::stride]
    ]


def trace_amplitude_scale(view: TraceView) -> float:
    """One symmetric amplitude scale for the original/filtered overlay."""
    return float(
        np.nanmax(np.abs(np.concatenate([view.original, view.filtered]))) or 1.0
    )

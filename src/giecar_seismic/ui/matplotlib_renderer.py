import time

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from giecar_seismic.application.seismic_viewer import (
    SeismicSection,
    TraceSpectrum,
    TraceView,
)
from giecar_seismic.ui.matplotlib_spectrum_renderer import MatplotlibSpectrumRenderer
from giecar_seismic.ui.seismic_renderer import (
    DisplaySettings,
    PanelRender,
    SeismicRenderer,
    amplitude_limit,
    panels_for_mode,
    section_extent,
    trace_amplitude_scale,
    wiggle_traces,
    x_axis_label,
)


class MatplotlibSeismicRenderer(SeismicRenderer):
    """Matplotlib (Qt5Agg) renderer -- the original viewer drawing code,
    unchanged in output. Figures/canvases are created once; only axes and
    artists are rebuilt per update, and a variable-density update that
    keeps the same panel layout reuses the existing images (set_data /
    set_clim / set_extent) instead of recreating them."""

    def __init__(self) -> None:
        super().__init__()
        self.last_panels: list[PanelRender] = []
        self.last_render_seconds = 0.0
        self.last_spectrum: TraceSpectrum | None = None

        self._section_panel = QWidget()
        layout = QVBoxLayout(self._section_panel)
        self._section_figure = Figure(figsize=(8, 6), tight_layout=True)
        self._section_canvas = FigureCanvasQTAgg(self._section_figure)
        self._toolbar = NavigationToolbar2QT(self._section_canvas, self._section_panel)
        self._click_cid = self._section_canvas.mpl_connect(
            "button_press_event", self._on_click
        )
        layout.addWidget(self._toolbar)
        layout.addWidget(self._section_canvas)

        self._analysis_panel = QWidget()
        analysis_layout = QVBoxLayout(self._analysis_panel)
        self._trace_figure = Figure(figsize=(4, 3), tight_layout=True)
        self._trace_canvas = FigureCanvasQTAgg(self._trace_figure)
        self._spectrum_renderer = MatplotlibSpectrumRenderer(toolbar=False)
        self._spectrum_figure = self._spectrum_renderer.figure
        self._spectrum_canvas = self._spectrum_renderer.canvas
        analysis_layout.addWidget(self._trace_canvas)
        analysis_layout.addWidget(self._spectrum_renderer.widget())

        self._images: list = []  # reusable imshow artists, one per panel
        self._images_key: tuple[int, str] | None = None  # (n_panels, cmap)

    # -- SeismicRenderer ----------------------------------------------------

    def section_widget(self) -> QWidget:
        return self._section_panel

    def analysis_widget(self) -> QWidget:
        return self._analysis_panel

    def show_section(
        self, section: SeismicSection | None, settings: DisplaySettings
    ) -> None:
        started = time.perf_counter()
        self.last_panels = []
        if section is None:
            self._section_figure.clear()
            self._images, self._images_key = [], None
            self._section_canvas.draw_idle()
            return

        limit = amplitude_limit(section, settings.clip_percentile, settings.gain)
        panels = panels_for_mode(section, settings.mode)
        extent = section_extent(section)
        coords = tuple(int(c) for c in section.coordinates)

        key = (len(panels), settings.colormap)
        if (
            settings.wiggle
            or self._images_key != key
            or len(self._images) != len(panels)
        ):
            self._section_figure.clear()
            self._images, self._images_key = [], None
            axes = np.atleast_1d(
                self._section_figure.subplots(1, len(panels), sharex=True, sharey=True)
            )
            for ax, (title, data) in zip(axes, panels):
                if settings.wiggle:
                    self._draw_wiggle(ax, section, data, limit, title)
                else:
                    self._images.append(
                        ax.imshow(
                            data.T,
                            aspect="auto",
                            cmap=settings.colormap,
                            vmin=-limit,
                            vmax=limit,
                            extent=extent,
                            interpolation="nearest",
                        )
                    )
                    self._label(ax, section, title)
            if not settings.wiggle:
                self._images_key = key
        else:
            # same layout: update the existing artists in place
            for ax, image, (title, data) in zip(
                self._section_figure.axes, self._images, panels
            ):
                image.set_data(data.T)
                image.set_clim(-limit, limit)
                image.set_extent(extent)
                self._label(ax, section, title)
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])

        self.last_panels = [
            PanelRender(
                title, data.shape, extent, (-limit, limit), coords, settings.wiggle
            )
            for title, data in panels
        ]
        self._section_canvas.draw_idle()
        self.last_render_seconds = time.perf_counter() - started

    def show_trace(
        self, view: TraceView | None, spectrum: TraceSpectrum | None
    ) -> None:
        self.last_spectrum = spectrum
        self._trace_figure.clear()
        if view is not None:
            ax = self._trace_figure.subplots()
            ax.plot(view.original, view.time_ms, label="original", linewidth=0.8)
            ax.plot(view.filtered, view.time_ms, label="filtered", linewidth=0.8)
            scale = trace_amplitude_scale(view)
            ax.set_xlim(-scale, scale)
            ax.set_ylim(float(view.time_ms[-1]), 0.0)
            ax.set_xlabel("Amplitude")
            ax.set_ylabel("Time (ms)")
            ax.set_title(f"Trace {view.geometry.trace_index}")
            ax.legend(loc="lower right", fontsize="small")
        self._spectrum_renderer.show_spectrum(spectrum)
        self._trace_canvas.draw_idle()

    def dispose(self) -> None:
        self._spectrum_renderer.dispose()
        self._section_canvas.mpl_disconnect(self._click_cid)
        for widget in (self._section_panel, self._analysis_panel):
            widget.setParent(None)  # type: ignore[call-overload]
            widget.deleteLater()

    # -- internals ---------------------------------------------------------------

    def _on_click(self, event: object) -> None:
        # ignore clicks while zoom/pan is active or outside the axes
        xdata = getattr(event, "xdata", None)
        if xdata is None or self._toolbar.mode:
            return
        self.coordinate_clicked.emit(float(xdata))

    @staticmethod
    def _draw_wiggle(
        ax: Axes, section: SeismicSection, data: np.ndarray, limit: float, title: str
    ) -> None:
        t = section.time_ms
        for x, trace in wiggle_traces(section, data, limit):
            ax.plot(x + trace, t, color="black", linewidth=0.5)
            ax.fill_betweenx(
                t, x, x + trace, where=trace > 0, color="black", linewidth=0
            )
        x0, x1, t_max, t_min = section_extent(section)
        ax.set_xlim(x0, x1)
        ax.set_ylim(t_max, t_min)
        MatplotlibSeismicRenderer._label(ax, section, title)

    @staticmethod
    def _label(ax: Axes, section: SeismicSection, title: str) -> None:
        ax.set_title(title)
        ax.set_xlabel(x_axis_label(section))
        ax.set_ylabel("Time (ms)")

    # test/debug access to the matplotlib objects
    @property
    def section_figure(self) -> Figure:
        return self._section_figure

    @property
    def trace_figure(self) -> Figure:
        return self._trace_figure

    @property
    def spectrum_figure(self) -> Figure:
        return self._spectrum_figure

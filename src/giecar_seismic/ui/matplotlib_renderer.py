import time

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from giecar_seismic.application.seismic_viewer import SeismicSection
from giecar_seismic.ui.seismic_renderer import (
    DisplaySettings,
    PanelRender,
    SeismicRenderer,
    amplitude_limit,
    panels_for_mode,
    section_extent,
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

        self._images: list = []  # reusable imshow artists, one per panel
        self._images_key: tuple[int, str] | None = None  # (n_panels, cmap)
        self._region_artists: list[Artist] = []
        self.last_region: tuple[float, float] | None = None
        self.last_region_start: float | None = None

    # -- SeismicRenderer ----------------------------------------------------

    def section_widget(self) -> QWidget:
        return self._section_panel

    def show_section(
        self, section: SeismicSection | None, settings: DisplaySettings
    ) -> None:
        started = time.perf_counter()
        self.last_panels = []
        self._clear_region_artists()
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

    def dispose(self) -> None:
        self._section_canvas.mpl_disconnect(self._click_cid)
        self._section_panel.setParent(None)  # type: ignore[call-overload]
        self._section_panel.deleteLater()

    # -- internals ---------------------------------------------------------------

    def _on_click(self, event: object) -> None:
        # ignore clicks while zoom/pan is active or outside the axes
        xdata = getattr(event, "xdata", None)
        if xdata is None or self._toolbar.mode:
            return
        self.coordinate_clicked.emit(float(xdata))

    def click_at_coordinate(self, coordinate: float) -> None:
        """Programmatic equivalent of a click at x=coordinate (tests)."""
        self.coordinate_clicked.emit(float(coordinate))

    # -- region highlight ----------------------------------------------------------

    def show_region_start(self, coordinate: float) -> None:
        self._clear_region_artists()
        for ax in self._section_figure.axes:
            line = ax.axvline(coordinate, color="#00a000", linestyle="--", linewidth=1)
            line.set_gid("region-start")
            self._region_artists.append(line)
        self.last_region_start = float(coordinate)
        self._section_canvas.draw_idle()

    def show_region(self, lower: float, upper: float) -> None:
        self._clear_region_artists()
        lower, upper = min(lower, upper), max(lower, upper)
        for ax in self._section_figure.axes:
            span = ax.axvspan(lower, upper, color="#00a000", alpha=0.12, linewidth=0)
            span.set_gid("region-span")
            self._region_artists.append(span)
            for bound in (lower, upper):
                line = ax.axvline(bound, color="#00a000", linestyle="--", linewidth=1)
                line.set_gid("region-bound")
                self._region_artists.append(line)
        self.last_region = (float(lower), float(upper))
        self._section_canvas.draw_idle()

    def clear_region(self) -> None:
        self._clear_region_artists()
        self._section_canvas.draw_idle()

    def _clear_region_artists(self) -> None:
        for artist in self._region_artists:
            if artist.axes is not None:  # figure.clear() may have removed it
                artist.remove()
        self._region_artists.clear()
        self.last_region = None
        self.last_region_start = None

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

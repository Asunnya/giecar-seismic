import time

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QRectF, Qt
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


class PyQtGraphSeismicRenderer(SeismicRenderer):
    """PyQtGraph renderer: ImageItem per panel inside PlotItems of one
    GraphicsLayoutWidget (variable density), PlotCurveItems for wiggle,
    a LinearRegionItem plus two InfiniteLines per panel for the region.

    Same data, same clipping, same modes as the matplotlib renderer --
    all of that comes from seismic_renderer's shared helpers. Known
    presentation differences: matplotlib leaves NaN gaps blank while here
    a gap is drawn at zero amplitude (colormap midpoint) in a
    display-only copy of the section panel; zoom/pan are PyQtGraph's
    native mouse interactions (no toolbar); wiggle positive lobes are
    filled via FillBetweenItem.
    """

    def __init__(self) -> None:
        super().__init__()
        self.last_panels: list[PanelRender] = []
        self.last_render_seconds = 0.0
        self.last_region: tuple[float, float] | None = None
        self.last_region_start: float | None = None

        self._section_panel = QWidget()
        layout = QVBoxLayout(self._section_panel)
        self._layout_widget = pg.GraphicsLayoutWidget()
        self._layout_widget.setBackground("w")  # match matplotlib; pg defaults to black
        layout.addWidget(self._layout_widget)
        self._plots: list[pg.PlotItem] = []
        self._images: list[pg.ImageItem] = []
        self._region_items: list[tuple[pg.PlotItem, pg.GraphicsObject]] = []
        self._layout_widget.scene().sigMouseClicked.connect(self._on_scene_clicked)

    # -- SeismicRenderer ----------------------------------------------------

    def section_widget(self) -> QWidget:
        return self._section_panel

    def show_section(
        self, section: SeismicSection | None, settings: DisplaySettings
    ) -> None:
        started = time.perf_counter()
        self.last_panels = []
        self._layout_widget.clear()  # drops every item, region included
        self._region_items.clear()
        self.last_region = None
        self.last_region_start = None
        self._plots, self._images = [], []
        if section is None:
            return

        limit = amplitude_limit(section, settings.clip_percentile, settings.gain)
        panels = panels_for_mode(section, settings.mode)
        x0, x1, t_max, t_min = section_extent(section)
        coords = tuple(int(c) for c in section.coordinates)
        lut = pg.colormap.getFromMatplotlib(settings.colormap).getLookupTable(nPts=256)

        first: pg.PlotItem | None = None
        for column, (title, data) in enumerate(panels):
            plot = self._layout_widget.addPlot(row=0, col=column, title=title)
            _darken_axes(plot)
            plot.setLabel("bottom", x_axis_label(section))
            plot.setLabel("left", "Time (ms)")
            plot.invertY(True)  # time increases downwards
            if settings.wiggle:
                self._draw_wiggle(plot, section, data, limit)
            else:
                image = pg.ImageItem()
                # display-only copy of one line: NaN gaps -> 0 (midpoint);
                # the section's arrays themselves are never modified.
                display = np.nan_to_num(data, nan=0.0)
                image.setImage(display, levels=(-limit, limit), autoLevels=False)
                image.setLookupTable(lut)
                image.setRect(QRectF(x0, t_min, x1 - x0, t_max - t_min))
                plot.addItem(image)
                self._images.append(image)
            plot.setXRange(x0, x1, padding=0)
            plot.setYRange(t_min, t_max, padding=0)
            if first is None:
                first = plot
            else:  # side-by-side: linked time and space axes
                plot.setXLink(first)
                plot.setYLink(first)
            self._plots.append(plot)

        self.last_panels = [
            PanelRender(
                title,
                data.shape,
                (x0, x1, t_max, t_min),
                (-limit, limit),
                coords,
                settings.wiggle,
            )
            for title, data in panels
        ]
        self.last_render_seconds = time.perf_counter() - started

    def dispose(self) -> None:
        self._layout_widget.scene().sigMouseClicked.disconnect(self._on_scene_clicked)
        self._layout_widget.clear()
        self._section_panel.setParent(None)  # type: ignore[call-overload]
        self._section_panel.deleteLater()

    # -- internals ---------------------------------------------------------------

    def _on_scene_clicked(self, event: object) -> None:
        scene_pos = getattr(event, "scenePos", lambda: None)()
        if scene_pos is None:
            return
        for plot in self._plots:
            view_box = plot.getViewBox()
            if view_box.sceneBoundingRect().contains(scene_pos):
                self.coordinate_clicked.emit(
                    float(view_box.mapSceneToView(scene_pos).x())
                )
                return

    def click_at_coordinate(self, coordinate: float) -> None:
        """Programmatic equivalent of a click at x=coordinate on the first
        panel (used by tests; goes through the same signal)."""
        if self._plots:
            self.coordinate_clicked.emit(float(coordinate))

    # -- region highlight ----------------------------------------------------------

    _REGION_PEN = ("#00a000", Qt.PenStyle.DashLine)

    def show_region_start(self, coordinate: float) -> None:
        self._clear_region_items()
        pen = pg.mkPen(self._REGION_PEN[0], style=self._REGION_PEN[1], width=1)
        for plot in self._plots:
            line = pg.InfiniteLine(pos=float(coordinate), angle=90, pen=pen)
            line.role = "region-start"
            plot.addItem(line)
            self._region_items.append((plot, line))
        self.last_region_start = float(coordinate)

    def show_region(self, lower: float, upper: float) -> None:
        self._clear_region_items()
        lower, upper = min(lower, upper), max(lower, upper)
        pen = pg.mkPen(self._REGION_PEN[0], style=self._REGION_PEN[1], width=1)
        for plot in self._plots:
            span = pg.LinearRegionItem(
                values=(lower, upper),
                movable=False,
                brush=pg.mkBrush(0, 160, 0, 30),
                pen=pg.mkPen(None),
            )
            span.setZValue(10)
            plot.addItem(span)
            self._region_items.append((plot, span))
            for bound in (lower, upper):
                line = pg.InfiniteLine(pos=bound, angle=90, pen=pen)
                line.role = "region-bound"
                plot.addItem(line)
                self._region_items.append((plot, line))
        self.last_region = (float(lower), float(upper))

    def clear_region(self) -> None:
        self._clear_region_items()

    def _clear_region_items(self) -> None:
        for plot, item in self._region_items:
            plot.removeItem(item)
        self._region_items.clear()
        self.last_region = None
        self.last_region_start = None

    @staticmethod
    def _draw_wiggle(
        plot: pg.PlotItem, section: SeismicSection, data: np.ndarray, limit: float
    ) -> None:
        t = section.time_ms
        pen = pg.mkPen("k", width=0.5)
        for x, trace in wiggle_traces(section, data, limit):
            curve = pg.PlotCurveItem(x + trace, t, pen=pen)
            baseline = pg.PlotCurveItem(np.full_like(t, x), t, pen=None)
            positive = pg.PlotCurveItem(x + np.clip(trace, 0, None), t, pen=None)
            plot.addItem(curve)
            plot.addItem(pg.FillBetweenItem(positive, baseline, brush=pg.mkBrush("k")))

    # test/debug access
    @property
    def plots(self) -> list[pg.PlotItem]:
        return self._plots

    @property
    def images(self) -> list[pg.ImageItem]:
        return self._images


def _darken_axes(plot: pg.PlotItem) -> None:
    """PyQtGraph's default axis/tick/title colours are chosen for a black
    background; on white, use black like matplotlib."""
    for side in ("left", "bottom"):
        axis = plot.getAxis(side)
        axis.setPen(pg.mkPen("k"))
        axis.setTextPen(pg.mkPen("k"))
    if plot.titleLabel is not None:
        plot.titleLabel.setText(plot.titleLabel.text, color="k")

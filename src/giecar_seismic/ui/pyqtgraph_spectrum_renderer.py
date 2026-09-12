import pyqtgraph as pg
from PyQt5.QtWidgets import QVBoxLayout

from giecar_seismic.application.seismic_viewer import TraceSpectrum
from giecar_seismic.ui.spectrum_renderer import SpectrumRenderer, SpectrumVisibility


class PyQtGraphSpectrumRenderer(SpectrumRenderer):
    """Reusable PlotDataItems with native zoom/pan and an independent gain axis."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self._widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget(background="w")
        layout.addWidget(self.plot)
        self.plot.setLabel("bottom", "Frequency (Hz)")
        self.plot.setLabel("left", "Magnitude")
        self.plot.setTitle("Amplitude spectrum", color="k")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.addLegend(labelTextColor="k", brush=(255, 255, 255, 220))
        self.original_curve = self.plot.plot(pen=pg.mkPen("#1f77b4"), name="original")
        self.filtered_curve = self.plot.plot(pen=pg.mkPen("#ff7f0e"), name="filtered")
        self.cutoff_lines = [
            pg.InfiniteLine(
                angle=90,
                pen=pg.mkPen("r", style=pg.QtCore.Qt.PenStyle.DashLine),
                label="cutoff",
                labelOpts={"position": position, "color": "#a00000"},
            )
            for position in (0.15, 0.35)
        ]
        for line in self.cutoff_lines:
            self.plot.addItem(line)
            line.hide()
        plot_item = self.plot.getPlotItem()
        for side in ("left", "bottom"):
            plot_item.getAxis(side).setPen(pg.mkPen("k"))
            plot_item.getAxis(side).setTextPen(pg.mkPen("k"))
        self.response_view = pg.ViewBox()
        plot_item.scene().addItem(self.response_view)
        plot_item.getAxis("right").linkToView(self.response_view)
        self.response_view.setXLink(plot_item.vb)
        self.response_curve = pg.PlotCurveItem(
            pen=pg.mkPen("green", style=pg.QtCore.Qt.PenStyle.DotLine)
        )
        self.response_view.addItem(self.response_curve)
        plot_item.vb.sigResized.connect(self._sync_response_geometry)
        self._sync_response_geometry()
        self.response_view.hide()
        self.response_curve.hide()
        self._response_in_legend = False

    def show_spectrum(
        self,
        spectrum: TraceSpectrum | None,
        visibility: SpectrumVisibility | None = None,
    ) -> None:
        settings = self._record(spectrum, visibility)
        for line in self.cutoff_lines:
            line.hide()
        if spectrum is None:
            self.original_curve.setData([], [])
            self.filtered_curve.setData([], [])
        else:
            self.original_curve.setData(spectrum.frequencies_hz, spectrum.original)
            self.filtered_curve.setData(spectrum.frequencies_hz, spectrum.filtered)
            self.original_curve.setVisible(settings.original)
            self.filtered_curve.setVisible(settings.filtered)
            for line, marker in zip(self.cutoff_lines, spectrum.cutoff_markers):
                # InfLineLabel ignores text changes while hidden.
                line.show()
                line.setPos(marker.frequency_hz)
                line.label.setFormat(f"{marker.label} {marker.frequency_hz} Hz")
                line.setVisible(not settings.empty)
            self.plot.setLabel("left", spectrum.magnitude_label)
            self.plot.setXRange(0, spectrum.nyquist_hz, padding=0)
            self.plot.enableAutoRange(axis=pg.ViewBox.YAxis)
        item = self.plot.getPlotItem()
        show_response = (
            spectrum is not None
            and settings.response
            and spectrum.filter_response is not None
        )
        self.response_curve.setVisible(show_response)
        self.response_view.setVisible(show_response)
        item.showAxis("right", show_response)
        if show_response:
            assert spectrum is not None
            self.response_curve.setData(
                spectrum.frequencies_hz, spectrum.filter_response
            )
            item.setLabel("right", spectrum.response_label, color="green")
            item.getAxis("right").setPen(pg.mkPen("green"))
            item.getAxis("right").setTextPen(pg.mkPen("green"))
            self.response_view.enableAutoRange(axis=pg.ViewBox.YAxis)
            self._sync_response_geometry()
        else:
            self.response_curve.setData([], [])
        if show_response and not self._response_in_legend:
            item.legend.addItem(self.response_curve, "Filter response (zero-phase)")
        elif not show_response and self._response_in_legend:
            item.legend.removeItem("Filter response (zero-phase)")
        self._response_in_legend = show_response

    def _sync_response_geometry(self) -> None:
        view = self.plot.getViewBox()
        self.response_view.setGeometry(view.sceneBoundingRect())
        self.response_view.linkedViewChanged(view, self.response_view.XAxis)

    def dispose(self) -> None:
        if self.disposed:
            return
        self.plot.getViewBox().sigResized.disconnect(self._sync_response_geometry)
        self.response_view.setXLink(None)
        self.plot.getPlotItem().scene().removeItem(self.response_view)
        self.response_view.deleteLater()
        super().dispose()

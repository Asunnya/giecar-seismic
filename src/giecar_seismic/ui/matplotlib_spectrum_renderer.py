from matplotlib.axes import Axes
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PyQt5.QtWidgets import QVBoxLayout

from giecar_seismic.application.seismic_viewer import TraceSpectrum
from giecar_seismic.ui.spectrum_renderer import SpectrumRenderer, SpectrumVisibility


class _SpectrumToolbar(NavigationToolbar2QT):
    # Native navigation only; exporting/editing figures is outside this view's scope.
    toolitems = [  # noqa: RUF012 -- Matplotlib requires this shared class-level toolbar definition
        item
        for item in NavigationToolbar2QT.toolitems
        if item[0] not in {"Save", "Subplots", "Customize"}
    ]


class MatplotlibSpectrumRenderer(SpectrumRenderer):
    """Persistent spectrum artists; the large view adds native zoom/pan."""

    def __init__(self, *, toolbar: bool = True) -> None:
        super().__init__()
        layout = QVBoxLayout(self._widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(8, 5), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = _SpectrumToolbar(self.canvas, self._widget) if toolbar else None
        if self.toolbar is not None:
            layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.axes = self.figure.subplots()
        self.axes.set_xlabel("Frequency (Hz)")
        self.axes.set_ylabel("Magnitude")
        self.axes.set_title("Amplitude spectrum")
        self.axes.grid(alpha=0.2)
        (self.original_line,) = self.axes.plot([], [], label="original", linewidth=0.8)
        (self.filtered_line,) = self.axes.plot([], [], label="filtered", linewidth=0.8)
        self.cutoff_lines: list[Line2D] = []
        self.response_axes: Axes | None = None
        self.response_line: Line2D | None = None

    def show_spectrum(
        self,
        spectrum: TraceSpectrum | None,
        visibility: SpectrumVisibility | None = None,
    ) -> None:
        settings = self._record(spectrum, visibility)
        if spectrum is None:
            self.original_line.set_data([], [])
            self.filtered_line.set_data([], [])
            for line in self.cutoff_lines:
                line.remove()
            self.cutoff_lines.clear()
            self._hide_response()
            legend = self.axes.get_legend()
            if legend is not None:
                legend.remove()
            self.canvas.draw_idle()
            return
        self.original_line.set_data(spectrum.frequencies_hz, spectrum.original)
        self.filtered_line.set_data(spectrum.frequencies_hz, spectrum.filtered)
        self.original_line.set_visible(settings.original)
        self.filtered_line.set_visible(settings.filtered)
        markers = spectrum.cutoff_markers
        while len(self.cutoff_lines) > len(markers):
            self.cutoff_lines.pop().remove()
        while len(self.cutoff_lines) < len(markers):
            self.cutoff_lines.append(self.axes.axvline(0, color="red", linestyle="--"))
        for line, marker in zip(self.cutoff_lines, markers, strict=True):
            line.set_xdata([marker.frequency_hz, marker.frequency_hz])
            line.set_label(f"{marker.label} {marker.frequency_hz} Hz")
            line.set_visible(not settings.empty)
        if settings.response and spectrum.filter_response is not None:
            if self.response_axes is None:
                self.response_axes = self.axes.twinx()
                (self.response_line,) = self.response_axes.plot(
                    [],
                    [],
                    color="green",
                    linestyle=":",
                    label="Filter response (zero-phase)",
                )
                self.response_axes.tick_params(axis="y", colors="green")
            assert self.response_line is not None
            self.response_line.set_data(
                spectrum.frequencies_hz, spectrum.filter_response
            )
            self.response_axes.set_ylabel(spectrum.response_label, color="green")
            self.response_axes.relim()
            self.response_axes.autoscale(enable=True, axis="y")
            self.response_axes.legend(loc="lower left", fontsize="small")
        else:
            self._hide_response()
        self.axes.set_xlim(0, spectrum.nyquist_hz)
        self.axes.set_ylabel(spectrum.magnitude_label)
        self.axes.relim(visible_only=True)
        self.axes.autoscale(enable=True, axis="y")
        lines = [line for line in self.axes.lines if line.get_visible()]
        legend = self.axes.get_legend()
        if legend is not None:
            legend.remove()
        if lines:
            self.axes.legend(handles=lines, loc="upper right", fontsize="small")
        # Old zoom history describes a different trace/scale after an update.
        if self.toolbar is not None:
            self.toolbar.update()
        self.canvas.draw_idle()

    def _hide_response(self) -> None:
        if self.response_axes is not None:
            self.response_axes.remove()
            self.response_axes = None
            self.response_line = None

    def dispose(self) -> None:
        if self.disposed:
            return
        if self.toolbar is not None:
            for cid in (
                self.toolbar._id_press,
                self.toolbar._id_release,
                self.toolbar._id_drag,
            ):
                self.canvas.mpl_disconnect(cid)
        super().dispose()

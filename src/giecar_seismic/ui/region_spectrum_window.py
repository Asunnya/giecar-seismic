"""Modeless, larger view of the viewer's CURRENT regional spectrum: one
aggregate original curve vs one aggregate filtered curve (mean of
per-trace amplitude spectra over the selected region).

Presentation only. The window owns scale, curve visibility and renderer;
it never acquires data or computes science: it receives the viewer's raw
(linear) RegionalSpectrum and derives every display variant with
regional_spectrum_for_display() (no FFT). When the viewer's region or
section changes, the owner empties the window instead of leaving curves
from another line on screen.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from giecar_seismic.application.seismic_viewer import (
    RegionalSpectrum,
    ResolvedRegion,
    SpectrumScale,
    regional_spectrum_for_display,
)
from giecar_seismic.domain.geometry import LineOrientation
from giecar_seismic.ui.filter_labels import FILTER_LABELS
from giecar_seismic.ui.matplotlib_spectrum_renderer import MatplotlibSpectrumRenderer
from giecar_seismic.ui.pyqtgraph_spectrum_renderer import PyQtGraphSpectrumRenderer
from giecar_seismic.ui.seismic_renderer import RENDERERS
from giecar_seismic.ui.spectrum_renderer import SpectrumRenderer, SpectrumVisibility
from giecar_seismic.ui.window_flags import make_resizable_dialog


def make_spectrum_renderer(name: str) -> SpectrumRenderer:
    if name == "Matplotlib":
        return MatplotlibSpectrumRenderer()
    return PyQtGraphSpectrumRenderer()


def axis_abbreviation(orientation: LineOrientation) -> str:
    """The axis a region spans: crosslines on an inline, inlines on a crossline."""
    return "XL" if orientation is LineOrientation.INLINE else "IL"


def region_summary(region: ResolvedRegion) -> str:
    """'Inline 10020 / Region: XL 2000–2050 / 48 traces present / 3 missing
    positions' -- the same wording in the viewer panel and this window."""
    line = f"{region.orientation.value.capitalize()} {region.line_number}"
    bounds = f"{region.region.lower_coordinate}–{region.region.upper_coordinate}"
    present = "trace" if region.n_present == 1 else "traces"
    missing = "position" if region.n_missing == 1 else "positions"
    text = (
        f"{line}\n"
        f"Region: {axis_abbreviation(region.orientation)} {bounds}\n"
        f"{region.n_present} {present} present\n"
        f"{region.n_missing} missing {missing}"
    )
    if region.is_single_trace:
        text += f"\nSingle trace (physical index {region.trace_indices[0]})"
    return text


class RegionSpectrumWindow(QDialog):
    def __init__(
        self,
        regional: RegionalSpectrum | None,
        parent: QWidget | None = None,
        *,
        renderer_name: str = "PyQtGraph",
        scale: SpectrumScale = SpectrumScale.LINEAR,
    ) -> None:
        super().__init__(parent)
        make_resizable_dialog(self)
        self.setWindowTitle("Regional spectrum")
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(1100, 760)
        self._disposed = False
        self.regional: RegionalSpectrum | None = None  # raw, linear
        self.display: RegionalSpectrum | None = None  # scale applied

        root = QVBoxLayout(self)
        self._method_label = QLabel(self)
        self._method_label.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(self._method_label)
        self._identity_label = QLabel(self)
        self._identity_label.setWordWrap(True)
        self._identity_label.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(self._identity_label)
        bar = QHBoxLayout()
        self._scale_combo = QComboBox(self)
        self._scale_combo.addItems([s.value for s in SpectrumScale])
        self._scale_combo.setCurrentText(scale.value)
        self._renderer_combo = QComboBox(self)
        self._renderer_combo.addItems(RENDERERS)
        self._renderer_combo.setCurrentText(renderer_name)
        self._original_checkbox = QCheckBox("Aggregate original", self)
        self._filtered_checkbox = QCheckBox("Aggregate filtered", self)
        self._response_checkbox = QCheckBox("Show filter response", self)
        self._original_checkbox.setChecked(True)
        self._filtered_checkbox.setChecked(True)
        for label, widget in (
            ("Scale:", self._scale_combo),
            ("Renderer:", self._renderer_combo),
            ("", self._original_checkbox),
            ("", self._filtered_checkbox),
            ("", self._response_checkbox),
        ):
            if label:
                bar.addWidget(QLabel(label, self))
            bar.addWidget(widget)
        bar.addStretch(1)
        root.addLayout(bar)
        self._status_label = QLabel(self)
        root.addWidget(self._status_label)
        self._plot_layout = QVBoxLayout()
        root.addLayout(self._plot_layout, 1)
        self._metadata_label = QLabel(self)
        self._metadata_label.setWordWrap(True)
        root.addWidget(self._metadata_label)
        self._reference_label = QLabel(
            "Per-trace |FFT| magnitudes are averaged in linear scale; dB is applied "
            "to the regional mean (reference: aggregate original peak, floor −120 dB). "
            "Filter response: one zero-phase gain from the job's design, right axis.",
            self,
        )
        self._reference_label.setWordWrap(True)
        root.addWidget(self._reference_label)

        self.renderer = make_spectrum_renderer(renderer_name)
        self._plot_layout.addWidget(self.renderer.widget())
        self._scale_combo.currentTextChanged.connect(self._redraw)
        self._renderer_combo.currentTextChanged.connect(self._change_renderer)
        for checkbox in (
            self._original_checkbox,
            self._filtered_checkbox,
            self._response_checkbox,
        ):
            checkbox.toggled.connect(self._redraw)
        self.set_regional(regional)

    def set_regional(self, regional: RegionalSpectrum | None) -> None:
        """Called only by the owner; None empties the window (stale guard)."""
        self.regional = regional
        if regional is None:
            self._method_label.setText("")
            self._identity_label.setText("")
            self._metadata_label.setText("")
        else:
            d, j = regional.dataset, regional.job
            region = regional.region
            self.setWindowTitle(
                f"Regional spectrum — {region.orientation.value} {region.line_number}, "
                f"{axis_abbreviation(region.orientation)} "
                f"{region.region.lower_coordinate}–{region.region.upper_coordinate}"
            )
            self._method_label.setText(
                "Single-trace amplitude spectrum (original vs filtered)"
                if region.is_single_trace
                else f"{RegionalSpectrum.METHOD} — {regional.n_present} traces "
                "(original vs filtered, one aggregate curve each)"
            )
            self._identity_label.setText(region_summary(region))
            cutoffs = "  |  ".join(
                f"{m.label}: {m.frequency_hz:g} Hz"
                for m in regional.spectrum.cutoff_markers
            )
            self._metadata_label.setText(
                f"Dataset: {d.name}  |  Filter type: {FILTER_LABELS[j.filter_type]}\n"
                f"{cutoffs}  |  Order: {j.order}  |  Sample rate: {d.sample_rate_ms} ms"
                f"  |  Nyquist: {regional.spectrum.nyquist_hz:g} Hz"
            )
        self._response_checkbox.setEnabled(
            regional is not None and regional.spectrum.filter_response is not None
        )
        self._redraw()

    def set_scale(self, scale: SpectrumScale) -> None:
        self._scale_combo.setCurrentText(scale.value)

    def _redraw(self, *_args: object) -> None:
        visibility = SpectrumVisibility(
            self._original_checkbox.isChecked(),
            self._filtered_checkbox.isChecked(),
            self._response_checkbox.isChecked() and self._response_checkbox.isEnabled(),
        )
        if self.regional is None:
            self.display = None
            self._status_label.setText(
                "No regional spectrum — select a region on the section."
            )
        else:
            self.display = regional_spectrum_for_display(
                self.regional,
                SpectrumScale(self._scale_combo.currentText()),
                show_filter_response=visibility.response,
            )
            self._status_label.setText(
                "No curves selected. Enable Original, Filtered or Filter response."
                if visibility.empty
                else ""
            )
        self.renderer.show_waveform(
            self.regional.waveform if self.regional is not None else None
        )
        self.renderer.show_spectrum(
            self.display.spectrum if self.display is not None else None, visibility
        )

    def _change_renderer(self, name: str) -> None:
        self._plot_layout.removeWidget(self.renderer.widget())
        self.renderer.dispose()
        self.renderer = make_spectrum_renderer(name)
        self._plot_layout.addWidget(self.renderer.widget())
        self._redraw()

    def done(self, result: int) -> None:
        if not self._disposed:
            self._disposed = True
            self.renderer.dispose()
            self.regional = None
            self.display = None
        super().done(result)

    def reject(self) -> None:
        self.done(QDialog.Rejected)

    def closeEvent(self, event: QCloseEvent | None) -> None:
        self.reject()
        if event is not None:
            event.accept()

"""Modeless consumer of the viewer's exact display spectrum; no acquisition.

Scale is requested from the owner and echoed back with fresh display data.
Curve visibility and renderer selection belong only to this inspection window.
"""

from PyQt5.QtCore import Qt, pyqtSignal
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
    SpectrumScale,
    TraceSpectrum,
    TraceView,
)
from giecar_seismic.ui.filter_labels import FILTER_LABELS
from giecar_seismic.ui.matplotlib_spectrum_renderer import MatplotlibSpectrumRenderer
from giecar_seismic.ui.pyqtgraph_spectrum_renderer import PyQtGraphSpectrumRenderer
from giecar_seismic.ui.seismic_renderer import RENDERERS
from giecar_seismic.ui.spectrum_renderer import SpectrumRenderer, SpectrumVisibility


def make_spectrum_renderer(name: str) -> SpectrumRenderer:
    if name == "Matplotlib":
        return MatplotlibSpectrumRenderer()
    return PyQtGraphSpectrumRenderer()


class SpectrumWindow(QDialog):
    scale_requested = pyqtSignal(str)

    def __init__(
        self,
        view: TraceView,
        spectrum: TraceSpectrum,
        parent: QWidget | None = None,
        *,
        renderer_name: str = "PyQtGraph",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Spectrum")
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(1100, 760)
        self._disposed = False
        self._view: TraceView | None = None
        self.spectrum: TraceSpectrum | None = None
        root = QVBoxLayout(self)
        self._identity_label = QLabel(self)
        self._identity_label.setWordWrap(True)
        self._identity_label.setTextFormat(Qt.TextFormat.PlainText)
        root.addWidget(self._identity_label)
        bar = QHBoxLayout()
        self._scale_combo = QComboBox(self)
        self._scale_combo.addItems([scale.value for scale in SpectrumScale])
        self._scale_combo.setCurrentText(spectrum.scale.value)
        self._scale_combo.currentTextChanged.connect(self.scale_requested.emit)
        self._renderer_combo = QComboBox(self)
        self._renderer_combo.addItems(RENDERERS)
        self._renderer_combo.setCurrentText(renderer_name)
        self._original_checkbox = QCheckBox("Original", self)
        self._filtered_checkbox = QCheckBox("Filtered", self)
        self._response_checkbox = QCheckBox("Show filter response", self)
        self._original_checkbox.setChecked(True)
        self._filtered_checkbox.setChecked(True)
        self._response_checkbox.setChecked(spectrum.show_filter_response)
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
            "dB: original and filtered share the original spectrum peak reference; "
            "floor −120 dB (all-zero original: reference 1). "
            "Filter response: zero-phase gain on the right axis, relative to unity.",
            self,
        )
        self._reference_label.setWordWrap(True)
        root.addWidget(self._reference_label)
        self.renderer = make_spectrum_renderer(renderer_name)
        self._plot_layout.addWidget(self.renderer.widget())
        self._renderer_combo.currentTextChanged.connect(self._change_renderer)
        for checkbox in (
            self._original_checkbox,
            self._filtered_checkbox,
            self._response_checkbox,
        ):
            checkbox.toggled.connect(self._redraw)
        self.set_spectrum(view, spectrum)

    def set_spectrum(
        self, view: TraceView | None, spectrum: TraceSpectrum | None
    ) -> None:
        """Called only by the owner: retain the same object, never request data."""
        self._view, self.spectrum = view, spectrum
        if view is None or spectrum is None:
            self._identity_label.setText("")
            self._metadata_label.setText("")
        else:
            self.set_scale(spectrum.scale)
            g, d, j = view.geometry, view.dataset, view.job
            self.setWindowTitle(f"Spectrum — Job {j.id}, trace {g.trace_index}")
            self._identity_label.setText(
                f"Job {j.id}  |  Dataset: {d.name}\n"
                f"Inline {g.inline}  |  Crossline {g.crossline}  |  Physical trace index {g.trace_index}"
            )
            cutoffs = "  |  ".join(
                f"{m.label}: {m.frequency_hz:g} Hz" for m in spectrum.cutoff_markers
            )
            self._metadata_label.setText(
                f"Filter type: {FILTER_LABELS[j.filter_type]}\n"
                f"{cutoffs}  |  Order: {j.order}  |  Sample rate: {d.sample_rate_ms} ms"
                f"  |  Nyquist: {spectrum.nyquist_hz:g} Hz"
            )
        self._response_checkbox.setEnabled(
            spectrum is not None and spectrum.filter_response is not None
        )
        self._redraw()

    def set_scale(self, scale: SpectrumScale) -> None:
        """Reflect the owner's scale even when navigation cleared the spectrum."""
        self._scale_combo.blockSignals(True)
        self._scale_combo.setCurrentText(scale.value)
        self._scale_combo.blockSignals(False)

    def _redraw(self, *_args: object) -> None:
        visibility = SpectrumVisibility(
            self._original_checkbox.isChecked(),
            self._filtered_checkbox.isChecked(),
            self._response_checkbox.isChecked() and self._response_checkbox.isEnabled(),
        )
        self._status_label.setText(
            "Select a seismic trace to inspect its spectrum."
            if self.spectrum is None
            else "No curves selected. Enable Original, Filtered or Filter response."
            if visibility.empty
            else ""
        )
        self.renderer.show_spectrum(self.spectrum, visibility)

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
            self._view = None
            self.spectrum = None
        super().done(result)

    def reject(self) -> None:
        self.done(QDialog.Rejected)

    def closeEvent(self, event: QCloseEvent | None) -> None:
        self.reject()
        if event is not None:
            event.accept()

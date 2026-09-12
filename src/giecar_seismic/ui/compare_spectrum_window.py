"""Modeless multi-trace spectral QC: ONE aggregate original curve vs ONE
aggregate filtered curve (mean of per-trace amplitude spectra) for the
viewer's Ctrl+click comparison set.

Distinct from SpectrumWindow (single active trace): no TraceView is faked
here. The window owns only presentation state -- scale, curve visibility,
renderer -- and never acquires data or computes science: it receives an
AggregateSpectrum from the viewer and derives every display variant with
aggregate_for_display() (no FFT). The raw aggregate is cached until the
owner replaces it; a selection or section change empties the window
instead of leaving stale curves on screen.
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
    AggregateSpectrum,
    SpectrumScale,
    aggregate_for_display,
)
from giecar_seismic.domain.geometry import LineOrientation
from giecar_seismic.ui.filter_labels import FILTER_LABELS
from giecar_seismic.ui.seismic_renderer import RENDERERS
from giecar_seismic.ui.spectrum_renderer import SpectrumVisibility
from giecar_seismic.ui.spectrum_window import make_spectrum_renderer

# Beyond this many coordinates the identity line shows an ellipsis.
MAX_LISTED_COORDINATES = 8


def coordinates_summary(aggregate: AggregateSpectrum) -> str:
    axis = (
        "crosslines" if aggregate.orientation is LineOrientation.INLINE else "inlines"
    )
    listed = ", ".join(str(c) for c in aggregate.coordinates[:MAX_LISTED_COORDINATES])
    if aggregate.n_traces > MAX_LISTED_COORDINATES:
        listed += ", …"
    return f"{axis} {listed}"


class CompareSpectrumWindow(QDialog):
    def __init__(
        self,
        aggregate: AggregateSpectrum,
        parent: QWidget | None = None,
        *,
        renderer_name: str = "PyQtGraph",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Compare spectra")
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(1100, 760)
        self._disposed = False
        self.aggregate: AggregateSpectrum | None = None  # raw, linear
        self.display: AggregateSpectrum | None = None  # scale applied

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
        self._scale_combo.addItems([scale.value for scale in SpectrumScale])
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
            "to the aggregate (reference: aggregate original peak, floor −120 dB). "
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
        self.set_aggregate(aggregate)

    def set_aggregate(self, aggregate: AggregateSpectrum | None) -> None:
        """Called only by the owner; None empties the window (stale guard)."""
        self.aggregate = aggregate
        if aggregate is None:
            self._method_label.setText("")
            self._identity_label.setText("")
            self._metadata_label.setText("")
        else:
            d, j = aggregate.dataset, aggregate.job
            n = aggregate.n_traces
            line = f"{aggregate.orientation.value.capitalize()} {aggregate.line_number}"
            self.setWindowTitle(f"Compare spectra — {n} traces, {line}")
            self._method_label.setText(
                f"{AggregateSpectrum.METHOD} — {n} traces "
                f"(original vs filtered, one aggregate curve each)"
            )
            self._identity_label.setText(
                f"{line}  |  {coordinates_summary(aggregate)}\n"
                f"Physical trace indices: "
                + ", ".join(
                    str(i) for i in aggregate.trace_indices[:MAX_LISTED_COORDINATES]
                )
                + (", …" if n > MAX_LISTED_COORDINATES else "")
            )
            cutoffs = "  |  ".join(
                f"{m.label}: {m.frequency_hz:g} Hz"
                for m in aggregate.spectrum.cutoff_markers
            )
            self._metadata_label.setText(
                f"Dataset: {d.name}  |  Filter type: {FILTER_LABELS[j.filter_type]}\n"
                f"{cutoffs}  |  Order: {j.order}  |  Sample rate: {d.sample_rate_ms} ms"
                f"  |  Nyquist: {aggregate.spectrum.nyquist_hz:g} Hz"
            )
        self._response_checkbox.setEnabled(
            aggregate is not None and aggregate.spectrum.filter_response is not None
        )
        self._redraw()

    def _redraw(self, *_args: object) -> None:
        visibility = SpectrumVisibility(
            self._original_checkbox.isChecked(),
            self._filtered_checkbox.isChecked(),
            self._response_checkbox.isChecked() and self._response_checkbox.isEnabled(),
        )
        if self.aggregate is None:
            self.display = None
            self._status_label.setText(
                "Selection changed — press Compare Spectra in the viewer again."
            )
        else:
            self.display = aggregate_for_display(
                self.aggregate,
                SpectrumScale(self._scale_combo.currentText()),
                show_filter_response=visibility.response,
            )
            self._status_label.setText(
                "No curves selected. Enable Original, Filtered or Filter response."
                if visibility.empty
                else ""
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
            self.aggregate = None
            self.display = None
        super().done(result)

    def reject(self) -> None:
        self.done(QDialog.Rejected)

    def closeEvent(self, event: QCloseEvent | None) -> None:
        self.reject()
        if event is not None:
            event.accept()

"""Spectrum presentation shared by the viewer panel and the detached
window: the amplitude spectrum plus, for a single-trace region, the
trace's waveform (amplitude x time) stacked above it.

No service, worker or scientific calculations: arrays/markers arrive ready
in the application's TraceSpectrum / TraceWaveform value objects.
"""

from dataclasses import dataclass

from PyQt5.QtWidgets import QVBoxLayout, QWidget

from giecar_seismic.application.seismic_viewer import TraceSpectrum, TraceWaveform


@dataclass(frozen=True)
class SpectrumVisibility:
    original: bool = True
    filtered: bool = True
    response: bool = False

    @property
    def empty(self) -> bool:
        return not (self.original or self.filtered or self.response)


class SpectrumRenderer:
    def __init__(self) -> None:
        self._widget = QWidget()
        self._layout = QVBoxLayout(self._widget)
        self._layout.setContentsMargins(0, 0, 0, 0)
        # Waveform panel above the spectrum; hidden unless a single trace.
        self._waveform_widget = QWidget(self._widget)
        self._waveform_widget.hide()
        self._layout.addWidget(self._waveform_widget, 1)
        self.last_spectrum: TraceSpectrum | None = None
        self.last_waveform: TraceWaveform | None = None
        self.visibility = SpectrumVisibility()
        self.disposed = False

    def widget(self) -> QWidget:
        return self._widget

    def waveform_widget(self) -> QWidget:
        return self._waveform_widget

    def show_waveform(self, waveform: TraceWaveform | None) -> None:
        """Draw one trace's original/filtered amplitude against time (time
        downwards), or hide the panel when there is no single trace."""
        raise NotImplementedError

    def _record_waveform(self, waveform: TraceWaveform | None) -> None:
        self.last_waveform = waveform
        self._waveform_widget.setVisible(waveform is not None)

    def show_spectrum(
        self,
        spectrum: TraceSpectrum | None,
        visibility: SpectrumVisibility | None = None,
    ) -> None:
        raise NotImplementedError

    def _record(
        self, spectrum: TraceSpectrum | None, visibility: SpectrumVisibility | None
    ) -> SpectrumVisibility:
        self.last_spectrum = spectrum
        self.visibility = visibility or SpectrumVisibility(
            response=spectrum.show_filter_response if spectrum is not None else False
        )
        return self.visibility

    def dispose(self) -> None:
        if self.disposed:
            return
        self.disposed = True
        self.last_spectrum = None
        self.last_waveform = None
        self._widget.hide()
        self._widget.setParent(None)  # type: ignore[call-overload]
        self._widget.deleteLater()

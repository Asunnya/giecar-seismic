"""Spectrum-only presentation shared by embedded and detached views.

No service, worker or scientific calculations: arrays/markers arrive ready
in the existing application TraceSpectrum (the project's SpectrumData).
"""

from dataclasses import dataclass

from PyQt5.QtWidgets import QWidget

from giecar_seismic.application.seismic_viewer import TraceSpectrum


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
        self.last_spectrum: TraceSpectrum | None = None
        self.visibility = SpectrumVisibility()
        self.disposed = False

    def widget(self) -> QWidget:
        return self._widget

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
        self._widget.hide()
        self._widget.setParent(None)  # type: ignore[call-overload]
        self._widget.deleteLater()

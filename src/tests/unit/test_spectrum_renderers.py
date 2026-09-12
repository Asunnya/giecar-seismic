import numpy as np
import pytest
from test_spectrum_science import make_view

from giecar_seismic.application.seismic_viewer import (
    SeismicViewerService,
    SpectrumScale,
    spectrum_for_display,
)
from giecar_seismic.domain.job import FilterType
from giecar_seismic.ui.matplotlib_spectrum_renderer import MatplotlibSpectrumRenderer
from giecar_seismic.ui.pyqtgraph_spectrum_renderer import PyQtGraphSpectrumRenderer
from giecar_seismic.ui.spectrum_renderer import SpectrumVisibility


@pytest.mark.parametrize(
    "kind,upper",
    [
        (FilterType.LOW_PASS, None),
        (FilterType.HIGH_PASS, None),
        (FilterType.BAND_PASS, 50),
    ],
)
@pytest.mark.parametrize("scale", list(SpectrumScale))
def test_shared_spectrum_renderers_data_markers_axes_and_visibility(
    qapp, kind, upper, scale
):
    raw = SeismicViewerService.spectrum(make_view(kind, 15, upper))
    data = spectrum_for_display(raw, scale)
    mpl, pg = MatplotlibSpectrumRenderer(), PyQtGraphSpectrumRenderer()
    try:
        for r in (mpl, pg):
            r.show_spectrum(data, SpectrumVisibility(response=True))
            assert r.last_spectrum is data
        np.testing.assert_array_equal(mpl.original_line.get_ydata(), data.original)
        np.testing.assert_array_equal(pg.original_curve.getData()[1], data.original)
        np.testing.assert_array_equal(mpl.filtered_line.get_ydata(), data.filtered)
        np.testing.assert_array_equal(pg.filtered_curve.getData()[1], data.filtered)
        np.testing.assert_array_equal(
            mpl.response_line.get_ydata(), data.filter_response
        )
        np.testing.assert_array_equal(
            pg.response_curve.getData()[1], data.filter_response
        )
        expected_label = (
            "Magnitude" if scale is SpectrumScale.LINEAR else "Magnitude (dB)"
        )
        assert mpl.axes.get_ylabel() == expected_label
        assert pg.plot.getAxis("left").labelText == expected_label
        assert mpl.axes.get_xscale() == "linear"
        assert mpl.axes.get_xlim() == (0, 125)
        assert pg.plot.getViewBox().viewRange()[0] == [0, 125]
        assert len(mpl.cutoff_lines) == len(data.cutoff_markers)
        visible = [line for line in pg.cutoff_lines if line.isVisible()]
        assert [line.value() for line in visible] == [
            m.frequency_hz for m in data.cutoff_markers
        ]
        for line, marker in zip(mpl.cutoff_lines, data.cutoff_markers, strict=True):
            assert line.get_xdata()[0] == marker.frequency_hz
            assert marker.label in line.get_label()
        assert mpl.toolbar is not None  # built-in zoom/pan
        assert pg.plot.getViewBox().state["mouseEnabled"] == [True, True]
        for original, filtered, response in [
            (False, True, False),
            (True, False, True),
            (False, False, False),
        ]:
            for r in (mpl, pg):
                r.show_spectrum(data, SpectrumVisibility(original, filtered, response))
            assert mpl.original_line.get_visible() is original
            assert mpl.filtered_line.get_visible() is filtered
            assert pg.original_curve.isVisible() is original
            assert pg.filtered_curve.isVisible() is filtered
            assert pg.response_curve.isVisible() is response
            assert (mpl.response_axes is not None) is response
    finally:
        mpl.dispose()
        pg.dispose()


@pytest.mark.parametrize(
    "renderer_class", [MatplotlibSpectrumRenderer, PyQtGraphSpectrumRenderer]
)
def test_trace_updates_reuse_items_and_dispose_is_idempotent(qapp, renderer_class):
    renderer = renderer_class()
    first = spectrum_for_display(
        SeismicViewerService.spectrum(make_view()), SpectrumScale.DB
    )
    second = spectrum_for_display(
        SeismicViewerService.spectrum(make_view(FilterType.HIGH_PASS, n=2000)),
        SpectrumScale.DB,
    )
    renderer.show_spectrum(first)
    item = (
        renderer.original_line
        if isinstance(renderer, MatplotlibSpectrumRenderer)
        else renderer.original_curve
    )
    renderer.show_spectrum(second)
    assert renderer.last_spectrum is second
    if isinstance(renderer, MatplotlibSpectrumRenderer):
        assert renderer.original_line is item
        assert len(renderer.axes.lines) == 3
    else:
        assert renderer.original_curve is item
    renderer.show_spectrum(None)
    assert renderer.last_spectrum is None
    renderer.dispose()
    renderer.dispose()
    assert renderer.disposed

import numpy as np
import pytest

from giecar_seismic.application.seismic_viewer import (
    SpectrumScale,
    spectrum_for_display,
)
from giecar_seismic.domain.job import FilterType
from giecar_seismic.ui.matplotlib_spectrum_renderer import MatplotlibSpectrumRenderer
from giecar_seismic.ui.pyqtgraph_spectrum_renderer import PyQtGraphSpectrumRenderer
from giecar_seismic.ui.spectrum_renderer import SpectrumVisibility
from tests.unit.test_spectrum_science import make_regional


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
    raw = make_regional(kind, 15, upper).spectrum
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
def test_spectrum_updates_reuse_items_and_dispose_is_idempotent(qapp, renderer_class):
    renderer = renderer_class()
    first = spectrum_for_display(make_regional().spectrum, SpectrumScale.DB)
    second = spectrum_for_display(
        make_regional(FilterType.HIGH_PASS, n_traces=5).spectrum, SpectrumScale.DB
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


@pytest.mark.parametrize(
    "renderer_class", [MatplotlibSpectrumRenderer, PyQtGraphSpectrumRenderer]
)
def test_waveform_panel_shows_a_single_trace_and_hides_for_regions(
    qapp, renderer_class
):
    renderer = renderer_class()
    single = make_regional(n_traces=1)
    waveform = single.waveform
    assert waveform is not None
    try:
        assert renderer.last_waveform is None
        assert renderer.waveform_widget().isHidden()

        renderer.show_waveform(waveform)

        assert renderer.last_waveform is waveform
        assert not renderer.waveform_widget().isHidden()
        if isinstance(renderer, MatplotlibSpectrumRenderer):
            ax = renderer.waveform_axes
            original, filtered = ax.get_lines()[:2]
            np.testing.assert_array_equal(original.get_xdata(), waveform.original)
            np.testing.assert_array_equal(original.get_ydata(), waveform.time_ms)
            np.testing.assert_array_equal(filtered.get_xdata(), waveform.filtered)
            assert ax.get_ylim()[0] > ax.get_ylim()[1]  # time downwards
            assert ax.get_xlim() == (
                -waveform.amplitude_scale,
                waveform.amplitude_scale,
            )
            assert ax.get_xlabel() == "Amplitude" and ax.get_ylabel() == "Time (ms)"
        else:
            np.testing.assert_array_equal(
                renderer.waveform_original.getData()[0], waveform.original
            )
            np.testing.assert_array_equal(
                renderer.waveform_original.getData()[1], waveform.time_ms
            )
            np.testing.assert_array_equal(
                renderer.waveform_filtered.getData()[0], waveform.filtered
            )
            assert renderer.waveform_plot.getViewBox().yInverted()
            assert renderer.waveform_plot.getViewBox().viewRange()[0] == [
                -waveform.amplitude_scale,
                waveform.amplitude_scale,
            ]

        renderer.show_waveform(None)  # a region: no time-domain curve at all
        assert renderer.last_waveform is None
        assert renderer.waveform_widget().isHidden()
    finally:
        renderer.dispose()


def test_both_renderers_receive_the_same_waveform_arrays(qapp):
    waveform = make_regional(n_traces=1).waveform
    mpl, pg = MatplotlibSpectrumRenderer(), PyQtGraphSpectrumRenderer()
    try:
        for r in (mpl, pg):
            r.show_waveform(waveform)
        np.testing.assert_array_equal(
            mpl.waveform_axes.get_lines()[0].get_xdata(),
            pg.waveform_original.getData()[0],
        )
    finally:
        mpl.dispose()
        pg.dispose()

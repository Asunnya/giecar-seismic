"""Filter cutoff spinboxes must accept normal keyboard entry -- the real
user path (focus, select, type digits, commit) through Qt key events, not
setValue(). Nyquist for sample_rate_ms=4.0 is 125 Hz, so 120 / 124.9 are
valid and 125 is not; band-pass keeps 0 < low < high < Nyquist, but that
rule must gate Run/Preview and the service -- never block typing.
"""

import numpy as np
import pytest
from PyQt5.QtCore import QLocale, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QDoubleSpinBox

from giecar_seismic.application.filter_jobs import InvalidFilterParametersError
from giecar_seismic.domain.dataset import SeismicDataset
from giecar_seismic.domain.job import FilterType
from giecar_seismic.ui.main_window import MainWindow
from tests.unit.test_main_window import (
    N_SAMPLES,
    FakeTraceReader,
    FakeTraceWriter,
    _build_service,
    _dataset,
)

DECIMAL = QLocale().decimalPoint()  # "." or "," depending on the machine


def _type(spinbox: QDoubleSpinBox, text: str, *, commit=Qt.Key_Return) -> list[str]:
    """Focus, select the current text and type `text` key by key, the way a
    user replaces a value; returns the line edit's text after each key."""
    spinbox.setFocus()
    spinbox.selectAll()
    seen = []
    for char in text:
        QTest.keyClick(spinbox, DECIMAL if char == "." else char)
        seen.append(spinbox.lineEdit().text())
    if commit is not None:
        QTest.keyClick(spinbox, commit)
    return seen


@pytest.fixture
def window(qapp):
    dataset = _dataset()  # sample_rate_ms=4.0 -> Nyquist 125 Hz
    service = _build_service(
        dataset, FakeTraceReader(np.zeros((4, N_SAMPLES))), FakeTraceWriter()
    )
    window = MainWindow(service=service, open_viewer=lambda target, parent: parent)
    window.set_dataset(dataset)
    window.show()
    yield window
    window.close()


def _select(window, kind: FilterType) -> None:
    window._filter_type_combo.setCurrentIndex(window._filter_type_combo.findData(kind))


@pytest.mark.parametrize("kind", [FilterType.LOW_PASS, FilterType.HIGH_PASS])
@pytest.mark.parametrize(
    "text,expected",
    [
        ("1", 1.0),
        ("10", 10.0),
        ("99", 99.0),
        ("100", 100.0),
        ("120", 120.0),
        ("124", 124.0),
        ("124.9", 124.9),
    ],
)
def test_single_cutoff_accepts_typed_values_below_nyquist(window, kind, text, expected):
    _select(window, kind)

    _type(window._cutoff_spinbox, text)

    assert window._cutoff_spinbox.value() == pytest.approx(expected)
    assert window._run_button.isEnabled()
    assert window._preview_button.isEnabled()


def test_typing_nyquist_itself_is_never_a_valid_cutoff(window):
    _select(window, FilterType.LOW_PASS)

    _type(window._cutoff_spinbox, "125")

    assert window._cutoff_spinbox.value() < 125.0
    assert window._cutoff_spinbox.maximum() == pytest.approx(124.99)
    # and the business layer rejects 125 independently of the widget
    with pytest.raises(InvalidFilterParametersError):
        window._service.create_filter_job(1, 125.0, 4)


def test_two_decimals_are_typed_and_kept(window):
    _type(window._cutoff_spinbox, "33.25")
    assert window._cutoff_spinbox.value() == pytest.approx(33.25)
    assert window._cutoff_spinbox.decimals() == 2


def test_band_pass_high_cutoff_120_is_typed_after_the_default_low(window):
    _select(window, FilterType.BAND_PASS)  # defaults: low 30, high 40

    _type(window._upper_cutoff_spinbox, "120")

    assert window._upper_cutoff_spinbox.value() == pytest.approx(120.0)
    assert window._cutoff_spinbox.value() == pytest.approx(30.0)
    assert window._run_button.isEnabled()


def test_band_pass_three_digit_low_is_typed_in_the_natural_order(window):
    # low first, then high -- the user must not have to raise high first or
    # reach 110 with the arrow buttons.
    _select(window, FilterType.BAND_PASS)  # defaults: low 30, high 40

    seen = _type(window._cutoff_spinbox, "110")
    assert seen[-1].startswith("110")
    assert window._cutoff_spinbox.value() == pytest.approx(110.0)
    assert window._upper_cutoff_spinbox.value() == pytest.approx(40.0)  # untouched
    assert not window._run_button.isEnabled()  # low >= high: gated, not clamped
    assert not window._preview_button.isEnabled()
    assert "below" in window._band_hint.text().lower()

    _type(window._upper_cutoff_spinbox, "120")

    assert window._upper_cutoff_spinbox.value() == pytest.approx(120.0)
    assert window._cutoff_spinbox.value() == pytest.approx(110.0)
    assert window._run_button.isEnabled() and window._preview_button.isEnabled()
    job = window._service.create_filter_job(
        1, 110.0, 4, filter_type=FilterType.BAND_PASS, upper_cutoff_hz=120.0
    )
    assert (job.cutoff_hz, job.upper_cutoff_hz) == (110.0, 120.0)


def test_band_pass_equal_or_inverted_cutoffs_gate_run_without_touching_values(window):
    _select(window, FilterType.BAND_PASS)
    _type(window._upper_cutoff_spinbox, "50")
    _type(window._cutoff_spinbox, "50")

    assert (window._cutoff_spinbox.value(), window._upper_cutoff_spinbox.value()) == (
        50.0,
        50.0,
    )
    assert not window._run_button.isEnabled()
    with pytest.raises(InvalidFilterParametersError):
        window._service.create_filter_job(
            1, 50.0, 4, filter_type=FilterType.BAND_PASS, upper_cutoff_hz=50.0
        )

    _type(window._upper_cutoff_spinbox, "60")
    assert window._run_button.isEnabled()


def test_mouse_steps_and_typed_values_agree_on_validity(window):
    _select(window, FilterType.BAND_PASS)
    _type(window._cutoff_spinbox, "39.99")
    window._upper_cutoff_spinbox.stepDown()  # 40 -> 39
    assert not window._run_button.isEnabled()
    window._upper_cutoff_spinbox.stepUp()  # 40 again: 39.99 < 40
    assert window._run_button.isEnabled()


def test_switching_filter_type_keeps_consistent_ranges(window):
    _select(window, FilterType.BAND_PASS)
    _type(window._cutoff_spinbox, "110")
    _type(window._upper_cutoff_spinbox, "120")
    for kind in (FilterType.LOW_PASS, FilterType.HIGH_PASS, FilterType.BAND_PASS):
        _select(window, kind)
        for spinbox in (window._cutoff_spinbox, window._upper_cutoff_spinbox):
            assert spinbox.minimum() == pytest.approx(0.01)
            assert spinbox.maximum() == pytest.approx(124.99)
        assert window._cutoff_spinbox.value() == pytest.approx(110.0)
        assert window._run_button.isEnabled()


def test_lower_nyquist_dataset_clamps_typed_values_and_revalidates(window):
    _select(window, FilterType.BAND_PASS)
    _type(window._cutoff_spinbox, "110")
    _type(window._upper_cutoff_spinbox, "120")
    low_nyquist = SeismicDataset(
        id=1,
        name="low",
        source_path="/data/low.segy",
        n_inlines=1,
        n_crosslines=1,
        n_traces=4,
        n_samples=N_SAMPLES,
        sample_rate_ms=200.0,  # Nyquist 2.5 Hz
    )

    window.set_dataset(low_nyquist)

    for spinbox in (window._cutoff_spinbox, window._upper_cutoff_spinbox):
        assert spinbox.maximum() == pytest.approx(2.49)
        assert spinbox.value() == pytest.approx(2.49)
    assert not window._run_button.isEnabled()  # low == high after clamping
    _type(window._cutoff_spinbox, "1.5")
    assert window._run_button.isEnabled()


def test_focus_change_commits_a_typed_value_like_enter(window):
    _type(window._cutoff_spinbox, "120", commit=None)
    window._order_spinbox.setFocus()  # user moves on without pressing Enter
    assert window._cutoff_spinbox.value() == pytest.approx(120.0)
    assert window._run_button.isEnabled()

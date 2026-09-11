from giecar_seismic.domain.dataset import SeismicDataset


def test_nyquist_hz_derived_from_sample_rate_ms():
    dataset = SeismicDataset(
        id=1,
        name="survey",
        source_path="/data/survey.segy",
        n_inlines=401,
        n_crosslines=720,
        n_samples=850,
        sample_rate_ms=4.0,
    )

    assert dataset.nyquist_hz == 125.0

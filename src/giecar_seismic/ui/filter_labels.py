"""Presentation labels shared by configuration, history and viewer."""

from giecar_seismic.domain.job import FilterType, Job

FILTER_NAMES = {
    FilterType.LOW_PASS: "Low-pass",
    FilterType.HIGH_PASS: "High-pass",
    FilterType.BAND_PASS: "Band-pass",
}
FILTER_LABELS = {
    FilterType.LOW_PASS: "Low-pass (High-cut) — removes high frequencies",
    FilterType.HIGH_PASS: "High-pass (Low-cut) — removes low frequencies",
    FilterType.BAND_PASS: "Band-pass (Low + High cut) — keeps the intermediate band",
}


def cutoff_summary(job: Job) -> str:
    if job.filter_type is FilterType.BAND_PASS:
        assert job.upper_cutoff_hz is not None
        return f"{job.cutoff_hz:g}–{job.upper_cutoff_hz:g} Hz"
    return f"{job.cutoff_hz:g} Hz"

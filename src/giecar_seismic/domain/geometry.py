from dataclasses import dataclass
from enum import Enum


class LineOrientation(Enum):
    INLINE = "inline"
    CROSSLINE = "crossline"


@dataclass(frozen=True)
class TraceGeometry:
    """Where one physical trace sits in the survey's inline/crossline grid.

    `trace_index` is the physical (sequential) position in the SEG-Y --
    and, because the filter pipeline preserves order, also in the HDF5
    output. The grid can be irregular, so this mapping must be looked
    up, never computed from inline * n_crosslines + crossline.
    """

    trace_index: int
    inline: int
    crossline: int

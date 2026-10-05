from .baseline import DetectorOnly
from .bytetrack import ByteTrackSingle
from .kalman_tracker import KalmanTracker

TRACKERS = {
    "detector_only": DetectorOnly,
    "kalman": KalmanTracker,
    "bytetrack": ByteTrackSingle,
}


def build_tracker(kind, **params):
    """Create a fresh tracker, e.g. build_tracker("kalman", det_thresh=0.4)."""
    return TRACKERS[kind](**params)

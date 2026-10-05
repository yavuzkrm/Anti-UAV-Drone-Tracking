"""
Detector-only baseline: no tracking at all.

Each frame reports the highest-scoring detection above det_thresh, or nothing.
Any gain a tracker shows over this baseline comes from tracking, not detection.

On frames where the detector did not run (detections=None) it simply repeats the
last box it reported - the naive way to fill gaps without a motion model.
"""


class DetectorOnly:

    def __init__(self, det_thresh=0.5):
        self.det_thresh = det_thresh
        self.last_box = None
        self.status = "lost"

    def update(self, detections):
        if detections is None:
            return self.last_box

        dets = detections.above(self.det_thresh)
        if len(dets) == 0:
            self.status = "lost"
            self.last_box = None
        else:
            self.status = "detected"
            self.last_box = dets.xyxy[dets.conf.argmax()].astype(float)
        return self.last_box

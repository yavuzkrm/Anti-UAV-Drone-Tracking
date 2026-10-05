"""
ByteTrack (ultralytics implementation) reduced to a single target.

ByteTrack is a multi-object tracker: it keeps several tracks, each with its own
Kalman Filter (constant velocity, box = center + aspect ratio + height), and matches
them to detections with the Hungarian algorithm on IoU. Its key idea is BYTE:
    1st round - match tracks with high-score detections
    2nd round - match the tracks still unmatched with LOW-score detections
A drone that turns small or blurry often gets a low score; the second round lets
the track continue instead of being lost.

Anti-UAV has one drone per sequence, so one track is reported per frame: the
current target while it is still tracked, otherwise the highest-scoring track.

Out of the box ByteTrack reports only tracks matched in the current frame. With
coast > 0 the target's Kalman prediction is reported for up to `coast` frames after
it was lost, the same thing KalmanTracker does with max_missed. Lost time is counted
in frames the detector ran on.
"""

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace


class ByteTrackSingle:

    def __init__(self, track_high_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.6,
                 track_buffer=30, match_thresh=0.8, fuse_score=True, coast=0):
        """
        track_high_thresh - detections at or above this go to the 1st association round
        track_low_thresh  - detections between low and high go to the 2nd round
                            (set it equal to track_high_thresh to switch BYTE off)
        new_track_thresh  - minimum score to start a new track
        track_buffer      - frames a lost track is kept for re-matching
        match_thresh      - maximum matching cost (1 - IoU) accepted in the 1st round
        coast             - frames to keep reporting the lost target's predicted box
        """
        args = IterableSimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=fuse_score,
        )
        self.tracker = BYTETracker(args)
        self.tracker.reset_id()
        self.coast = coast
        self.target_id = None
        self.status = "lost"

    def update(self, detections):
        """
        Process one frame. Returns the drone box [x1, y1, x2, y2] or None.
        detections=None means the detector did not run on this frame.
        """
        if detections is None:
            return self._predict_only()

        tracks = self.tracker.update(detections)  # rows: x1, y1, x2, y2, id, score, cls, idx
        if len(tracks) == 0:
            return self._coast_target()

        same = tracks[tracks[:, 4] == self.target_id]
        if len(same) > 0:
            row = same[0]
        else:
            row = tracks[tracks[:, 5].argmax()]
            self.target_id = row[4]

        self.status = "detected"
        return row[:4].astype(float)

    def _predict_only(self):
        """
        Frame without detector output: move every track one frame forward with its
        Kalman Filter, without any matching. Calling update() with no detections instead
        would mark the tracks as lost and delete tracks that are not confirmed yet.
        """
        active = [t for t in self.tracker.tracked_stracks if t.is_activated]
        self.tracker.multi_predict(active + self.tracker.lost_stracks)

        for track in active:
            if track.track_id == self.target_id:
                self.status = "predicted"
                return track.xyxy.astype(float)
        return self._coast_target()

    def _coast_target(self):
        """Report the lost target's Kalman prediction, if it was lost recently enough."""
        for track in self.tracker.lost_stracks:
            frames_lost = self.tracker.frame_id - track.end_frame
            if track.track_id == self.target_id and frames_lost <= self.coast:
                self.status = "predicted"
                return track.xyxy.astype(float)
        self.status = "lost"
        return None

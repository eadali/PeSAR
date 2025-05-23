import supervision as sv


class ByteTrack:
    def __init__(self):
        self.tracker = sv.ByteTrack()
        self.smoother = sv.DetectionsSmoother()

    def __call__(self, detections):
        """Process detections using ByteTrack."""
        # supervision_detections = self._convert_pytorch_to_supervision(detections)
        tracked_detections = self.tracker.update_with_detections(detections)
        smoothed_detections = self.smoother.update_with_detections(tracked_detections)
        return smoothed_detections
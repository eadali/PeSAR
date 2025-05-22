import supervision as sv


class ByteTrack:
    def __init__(self):
        self.tracker = sv.ByteTrack()
        self.smoother = sv.DetectionsSmoother()

    # @staticmethod
    # def _convert_pytorch_to_supervision(detections):
    #     """Convert PyTorch detections to Supervision Detections."""
    #     boxes = detections['boxes'].cpu().numpy()
    #     scores = detections['scores'].cpu().numpy()
    #     labels = detections['labels'].cpu().numpy()
    #     return sv.Detections(xyxy=boxes, confidence=scores, class_id=labels)

    # @staticmethod
    # def _convert_supervision_to_pytorch(detections):
    #     """Convert Supervision Detections to PyTorch detections."""
    #     return {
    #         'boxes': torch.tensor(detections.xyxy, dtype=torch.float32),
    #         'labels': torch.tensor(detections.class_id, dtype=torch.int64),
    #         'scores': torch.tensor(detections.confidence, dtype=torch.float32),
    #         'ids': torch.tensor(detections.tracker_id, dtype=torch.float32)
    #     }

    def __call__(self, detections):
        """Process detections using ByteTrack."""
        # supervision_detections = self._convert_pytorch_to_supervision(detections)
        tracked_detections = self.tracker.update_with_detections(detections)
        smoothed_detections = self.smoother.update_with_detections(tracked_detections)
        return smoothed_detections


def build(args):
    """Factory function to create a ByteTrack instance."""
    return ByteTrack()

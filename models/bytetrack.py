import supervision as sv


class ByteTrack:
    def __init__(self):
        self.tracker = sv.ByteTrack()

    def __call__(self, detections):
        # Convert bounding boxes to numpy
        boxes = detections[0]['boxes'].cpu().numpy()
        scores = detections[0]['scores'].cpu().numpy()  # Convert scores to numpy
        labels = detections[0]['labels'].cpu().numpy()  # Convert labels to numpy

# Create a supervision.Detections object
        return self.tracker.update_with_detections(sv.Detections(
            # Bounding boxes in [x_min, y_min, x_max, y_max] format
            xyxy=boxes,
            confidence=scores,  # Confidence scores
            class_id=labels  # Class labels
        ))


def build(args):
    return ByteTrack()

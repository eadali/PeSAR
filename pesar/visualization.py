import supervision as sv
import numpy as np


def draw_estimations(frame, estimations, class_id_to_name=None):
    if class_id_to_name is None:
        class_id_to_name = {}
    # Extract bounding boxes, class labels, and tracker IDs from estimations
    bounding_boxes = estimations['boxes'].cpu().numpy()
    class_labels = estimations['labels'].cpu().numpy()
    tracker_ids = estimations['ids'].cpu().numpy()
    # Create a Detections object for easier handling
    detections = sv.Detections(
        xyxy=bounding_boxes, class_id=class_labels, tracker_id=tracker_ids)
    # Generate descriptive labels for each detection
    detection_labels = [_generate_label(class_id, tracker_id, class_id_to_name)
                        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)]
    # Annotate the frame with bounding boxes and labels
    frame = _annotate_frame(frame, detections, detection_labels)
    return frame


def _generate_label(class_id, tracker_id, class_id_to_name):
    """Generate a label for a detection based on class name and tracker ID."""
    class_name = class_id_to_name.get(class_id, 'Unknown')
    if np.isnan(tracker_id):
        return class_name
    return f"#{int(tracker_id)} {class_name}"


def _annotate_frame(frame, detections, labels):
    """Annotate the frame with bounding boxes and labels."""
    box_annotator = sv.BoxCornerAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_padding=1)
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(
        scene=frame, detections=detections, labels=labels)
    return frame

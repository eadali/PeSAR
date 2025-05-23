from .yolo import YOLO


def build_detector(config):
    """
    Build the detection model based on the provided configuration.
    """
    # Initialize the YOLO object detection model
    return YOLO(config.thresholds.confidence,
                config.thresholds.iou,
                config.slicing.overlap,
                config.categories,
                config.device)
    
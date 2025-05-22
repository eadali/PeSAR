from .yolo import YOLO


def build_detection(cfg, onnx_path):
    """
    Build the detection model based on the provided configuration.
    """
    # Initialize the YOLO object detection model
    detection = YOLO(onnx_path, cfg.confidence_threshold, cfg.iou_threshold)
    return detection
    
# Import the necessary modules
from .perception import Perception
from .detection import build_detection
from .tracking import build_tracking

def build_perception(cfg, onnx_path):
    """
    Build the perception perception based on the provided configuration.
    """

    # Build the detector and tracker
    detection = build_detection(cfg.detection, onnx_path)
    tracking = build_tracking(cfg.tracking)

    # Create a perception object with the detector and tracker
    return Perception(detection=detection, tracking=tracking)
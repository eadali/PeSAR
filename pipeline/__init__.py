from .pipeline import Pipeline
from .detectors import build_detector
from .trackers import build_tracker

def build_pipeline(config):
    """
    Build and return a pipeline based on the provided configuration.
    """
    # Build detector and tracker using the config
    detector = build_detector(config.detector)
    tracker = build_tracker(config.tracker)

    # Create and return a Pipeline object with detector and tracker
    return Pipeline(detector=detector, tracker=tracker)
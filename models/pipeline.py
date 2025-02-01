from .waldo30 import build as build_waldo30
from .bytetrack import build as build_bytetrack

class Pipeline:
    def __init__(self, detector, tracker=None):
        self.detector = detector
        self.tracker = tracker

    def __call__(self, frame):
        detections = self._apply_detector(frame)
        return self._apply_tracker(detections) if self.tracker else detections

    def _apply_detector(self, frame):
        return self.detector(frame)

    def _apply_tracker(self, detections):
        return self.tracker(detections)


def build(args):
    if args.detector == 'waldo30':
        detector = build_waldo30(args)
    else:
        raise ValueError(f'Invalid detector: {args.detector}')

    tracker = None
    if args.tracker == 'bytetrack':
        tracker = build_bytetrack(args)
    elif args.tracker and args.tracker != 'none':
        raise ValueError(f'Invalid tracker: {args.tracker}')

    return Pipeline(detector=detector, tracker=tracker)

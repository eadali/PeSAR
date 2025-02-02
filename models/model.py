from .waldo30 import build as build_waldo30
from .bytetrack import build as build_bytetrack
from .dummytrack import build as build_bypasstrack


class Model:
    def __init__(self, detector, tracker=None):
        self.detector = detector
        self.tracker = tracker

    def __call__(self, frame):
        detections = self._apply_detector(frame)
        detections = self._apply_tracker(detections)
        return detections

    def _apply_detector(self, frame):
        return self.detector(frame)

    def _apply_tracker(self, detections):
        return self.tracker(detections)

    def get_class_mapping(self):
        return self.detector.get_class_mapping()


def _build_detector(args):
    if args.detector == 'waldo30':
        return build_waldo30(args)
    raise ValueError(f'Invalid detector: {args.detector}')


def _build_tracker(args):
    if args.tracker == 'bytetrack':
        return build_bytetrack(args)
    if args.tracker is None:
        return build_bypasstrack(args)
    raise ValueError(f'Invalid tracker: {args.tracker}')


def build(args):
    detector = _build_detector(args)
    tracker = _build_tracker(args)
    return Model(detector=detector, tracker=tracker)

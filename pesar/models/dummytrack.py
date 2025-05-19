import torch
import numpy as np
from typing import Dict


class DummyTrack:
    """A placeholder tracker that assigns NaN values as tracker IDs."""
    def __call__(self, detections: Dict[str, torch.Tensor]):
        num_boxes = detections['boxes'].shape[0]
        detections['ids'] = torch.full((num_boxes,), np.nan, dtype=torch.float32)
        return detections


def build(args):
    """Factory function to create a DummyTrack instance."""
    return DummyTrack()

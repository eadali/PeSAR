from .bytetrack import ByteTrack

def build_tracker(cfg):
    """
    Build the tracking model based on the provided configuration.
    """
    # Initialize the tracker
    return ByteTrack()
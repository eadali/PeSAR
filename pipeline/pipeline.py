class Pipeline:
    def __init__(self, detector, tracker):
        """
        Initialize the Pipeline class with a detector and tracker.

        Args:
            detector (object): The object detection model.
            tracker (object): The object tracking model.
        """
        self.detector = detector
        self.tracker = tracker

    def load_state_dict(self, onnx_path):
        self.detector.load_state_dict(onnx_path)

    def __call__(self, frame):
        """
        Run the detection and tracking on the input image.

        Args:
            frame (np.ndarray): The input image to process.

        Returns:
            supervision.Detections: Detections object after tracking.
        """
        return self.tracker(self.detector(frame))
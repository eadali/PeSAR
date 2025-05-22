import cv2

class Perception:
    def __init__(self, detection, tracking):
        """
        Initialize the Perception class with a detector and tracker.

        Args:
            detector (object): The object detection model.
            tracker (object): The object tracking model.
        """
        self.detection = detection
        self.tracking = tracking
    
    def __call__(self, frame):
        """
        Run the detection and tracking on the input image.

        Args:
            image (np.ndarray): The input image to process.

        Returns:
            List[Dict]: List of dictionaries containing detection information such as class_id, class_name, confidence,
            box coordinates, and scale factor.
        """
        # resize image to 640x640
        frame = cv2.resize(frame, (416, 416))
        
        # Perform detection
        detections = self.detection(frame)
        
        # Perform tracking
        tracked_objects = self.tracking(detections)
        
        
        return tracked_objects
    
    
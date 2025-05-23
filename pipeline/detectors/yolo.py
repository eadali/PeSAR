import numpy as np
import supervision as sv
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

class YOLO:
    def __init__(self, confidence_threshold, iou_threshold, slicing_overlap, categories, device):
        """
        YOLO detector wrapper using SAHI for sliced prediction.

        Args:
            confidence_threshold (float): Minimum confidence for detections.
            iou_threshold (float): IoU threshold for NMS (not used directly here).
            slicing_overlap (float): Overlap ratio for slicing.
            categories (list): List of class names.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.model = None
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.slicing_overlap = slicing_overlap
        self.categories = categories
        self.category_mapping = {str(i): category for i, category in enumerate(categories)}
        self.device = device

    def load_onnx_model(self, path):
        """
        Loads the ONNX model using SAHI's AutoDetectionModel.
        """
        self.model = AutoDetectionModel.from_pretrained(
            model_type='yolov8onnx',
            model_path=path,
            confidence_threshold=self.confidence_threshold,
            category_mapping=self.category_mapping,
            device=self.device
        )

    def __call__(self, frame):
        """
        Runs sliced prediction on the input frame and returns a supervision.Detections object.
        """
        # Get input shape from ONNX model
        input_shape = self.model.model.get_inputs()[0].shape[2]
        result = get_sliced_prediction(
            frame,
            self.model,
            slice_height=input_shape,
            slice_width=input_shape,
            overlap_height_ratio=self.slicing_overlap,
            overlap_width_ratio=self.slicing_overlap,
            verbose=False,
        )
        boxes = []
        confidences = []
        class_ids = []
        for det in result.object_prediction_list:
            boxes.append(det.bbox.to_xyxy())
            confidences.append(det.score.value)
            class_ids.append(det.category.id)
        if boxes:
            boxes = np.array(boxes)
            confidences = np.array(confidences)
            class_ids = np.array(class_ids)
        else:
            boxes = np.zeros((0, 4))
            confidences = np.zeros((0,))
            class_ids = np.zeros((0,))
        detections = sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids,
        )
        return detections
    
    def get_category_mapping(self):
        """
        Returns the category mapping.
        """
        # Convert string keys to integers
        return {int(k): v for k, v in self.category_mapping.items()}
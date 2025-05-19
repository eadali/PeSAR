from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import torch


class WALDO30:
    def __init__(
        self,
        path,
        confidence_threshold=0.8,
        device="cpu",
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    ):
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.slice_height = 640
        self.slice_width = 640

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=path,
            confidence_threshold=confidence_threshold,
            image_size=640,
            device=device,
            load_at_init=True,
        )

        self.class_id_to_name = {
            0: "Car",
            1: "Person",
            4: "Boat",
            5: "Bike",
            7: "Truck",
            11: "Bus",
        }

        # Define allowed class IDs for filtering
        self.allowed_class_ids = {0, 1, 4, 5, 7, 11}

    def __call__(self, frame):
        result = get_sliced_prediction(
            frame,
            self.detection_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            verbose=0
        )

        if not result.object_prediction_list:
            return self._create_empty_detection()

        boxes, labels, scores = self._extract_detections(result)
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "scores": torch.tensor(scores, dtype=torch.float32),
        }

    def _extract_detections(self, result):
        boxes, labels, scores = [], [], []

        for prediction in result.object_prediction_list:
            if prediction.category.id not in self.allowed_class_ids:
                continue  # Skip unwanted classes
            boxes.append(prediction.bbox.to_voc_bbox())
            labels.append(prediction.category.id)
            scores.append(prediction.score.value)

        return boxes, labels, scores

    def _create_empty_detection(self):
        return {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.int64),
            "scores": torch.empty((0,), dtype=torch.float32),
        }

    def get_class_mapping(self):
        return self.class_id_to_name


def build(args):
    return WALDO30(
        path="data/WALDO30_yolov8n_640x640.pt",
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        overlap_height_ratio=args.overlap_height_ratio,
        overlap_width_ratio=args.overlap_width_ratio,
    )

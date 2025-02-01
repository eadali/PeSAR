from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import torch

WALDO30_PATH = 'data/WALDO30_yolov8n_640x640.pt' #TODO: Take this from here


class WALDO30:
    def __init__(self, path, confidence_threshold=0.8, device='cpu',
                 overlap_height_ratio=0.2, overlap_width_ratio=0.2):
        self.overlap_heigh_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.slice_height = 640
        self.slice_width = 640

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=path,
            confidence_threshold=confidence_threshold,
            image_size=640,
            device=device,
            load_at_init=True,
        )

    def __call__(self, frame):
        result = get_sliced_prediction(
            frame,
            self.detection_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_heigh_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
        )
        boxes = []
        labels = []
        scores = []

        for prediction in result.object_prediction_list:
            # Convert to [xmin, ymin, xmax, ymax] format
            bbox = prediction.bbox.to_voc_bbox()
            boxes.append(bbox)
            # Assuming category.id is the label
            labels.append(prediction.category.id)
            scores.append(prediction.score.value)  # Confidence score

        # Convert lists to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        scores = torch.tensor(scores, dtype=torch.float32)
        return [{
            'boxes': boxes,
            'labels': labels,
            'scores': scores
        }]


def build(args):
    detector = WALDO30(path='data/WALDO30_yolov8n_640x640.pt',
                       confidence_threshold=args.confidence_threshold,
                       device=args.device,
                       overlap_height_ratio=args.overlap_height_ratio,
                       overlap_width_ratio=args.overlap_width_ratio)

    return detector

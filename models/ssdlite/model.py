from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


class SSDLite:
    def __init__(self, overlap_height_ratio=0.2, overlap_width_ratio=0.2, device='cpu'):
        self.overlap_heigh_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.slice_height = 320
        self.slice_width = 320
        model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        #     weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='torchvision',
            model=model,
            confidence_threshold=0.5,
            image_size=640,
            device=device,  # or "cuda:0"
            load_at_init=True,
        )

    def __call__(self, x):
        result = get_sliced_prediction(
            x,
            self.detection_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_heigh_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
        )
        print('here')
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
        return result

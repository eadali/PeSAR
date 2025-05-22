import onnxruntime as ort
import numpy as np
import cv2
import supervision as sv

class YOLO:
    def __init__(self, onnx_path: str, confidence_threshold: float, iou_threshold: float):
        """
        Initialize the YOLO object detection model.

        Args:
            model_path (str): Path to the YOLO model file.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.session = ort.InferenceSession(onnx_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        

    def __call__(self, frame: np.ndarray):
        """
        Perform object detection on the input frame.
        Args:
            frame (np.ndarray): Input image frame.
        Returns:
            List[Dict]: List of dictionaries containing detection information such as class_id, class_name, confidence,
            box coordinates, and scale factor.
        """     
        [height, width, _] = frame.shape
        length = max((height, width))


    # Calculate scale factor
        scale = 416 / 416

        # Preprocess the image and prepare blob for model
        frame_rgb = frame[..., ::-1]
        frame_norm = frame_rgb.astype(np.float32) / 255.0
        frame_transposed = np.transpose(frame_norm, (2, 0, 1))
        blob = np.expand_dims(frame_transposed, axis=0)

        # Perform inference
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})[0]

        # Prepare output array
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.20:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),  # x center - width/2 = left x
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),  # y center - height/2 = top y
                    outputs[0][i][2],  # width
                    outputs[0][i][3],  # height
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        detections = []

        # Iterate through NMS results to draw bounding boxes and labels
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            print(box)
            detection = {
                "class_id": class_ids[index],
                "class_name": 'ddd',
                "confidence": scores[index],
                # "box": box,
                "box": [
                    round(box[0] * scale),  # left x
                    round(box[1] * scale),  # top y
                    round(box[0] * scale) + round(box[2] * scale),  # right x
                    round(box[1] * scale) + round(box[3] * scale),  # bottom y
                ],

                "scale": scale,
            }
            detections.append(detection)
            # draw_bounding_box(
            #     original_image,
            #     class_ids[index],
            #     scores[index],
            #     round(box[0] * scale),
            #     round(box[1] * scale),
            #     round((box[0] + box[2]) * scale),
            #     round((box[1] + box[3]) * scale),
            # )

        # # Display the image with bounding boxes
        # cv2.imshow("image", original_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # Convert detections to supervision format
        dd = sv.Detections(
            xyxy=np.array([d["box"] for d in detections]),
            confidence=np.array([d["confidence"] for d in detections]),
            class_id=np.array([d["class_id"] for d in detections]),
        )
        print(np.array([d["box"] for d in detections]))


        return dd
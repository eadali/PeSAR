# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse

import cv2.dnn
import numpy as np
import onnxruntime as ort
# from ultralytics.utils import ASSETS, YAML
# from ultralytics.utils.checks import check_yaml

# CLASSES = YAML.load(check_yaml("coco8.yaml"))["names"]
# colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Draw bounding boxes on the input image based on the provided arguments.

    Args:
        img (np.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
        y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    label = f"{'dd'} ({confidence:.2f})"
    color = (255,0,0)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main(onnx_model, input_image):
    """
    Load ONNX model, perform inference, draw bounding boxes, and display the output image.

    Args:
        onnx_model (str): Path to the ONNX model.
        input_image (str): Path to the input image.

    Returns:
        (List[Dict]): List of dictionaries containing detection information such as class_id, class_name, confidence,
        box coordinates, and scale factor.
    """
    # Load the ONNX model
    ort_session = ort.InferenceSession(onnx_model)
    # model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)

    # Read the input image
    original_image: np.ndarray = cv2.imread(input_image)
    [height, width, _] = original_image.shape

    # Prepare a square image for inference
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image

    # Calculate scale factor
    scale = length / 640

    # Preprocess the image and prepare blob for model
# Resize to (640, 640)
    img_resized = cv2.resize(image, (640, 640))
    # Convert BGR to RGB if needed (YOLOv8 ONNX expects RGB)
    img_rgb = img_resized[..., ::-1]
    # Normalize to [0, 1]
    img_norm = img_rgb.astype(np.float32) / 255.0
    # Transpose to (C, H, W)
    img_transposed = np.transpose(img_norm, (2, 0, 1))
    # Add batch dimension: (1, 3, 640, 640)
    blob = np.expand_dims(img_transposed, axis=0)
    print(image.shape)
    print(blob.shape)
    # model.setInput(blob)

    # Perform inference
    # outputs = model.forward()
    outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: blob})[0]
    print(outputs)

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
        if maxScore >= 0.80:
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
        detection = {
            "class_id": class_ids[index],
            "class_name": 'ddd',
            "confidence": scores[index],
            "box": box,
            "scale": scale,
        }
        detections.append(detection)
        draw_bounding_box(
            original_image,
            class_ids[index],
            scores[index],
            round(box[0] * scale),
            round(box[1] * scale),
            round((box[0] + box[2]) * scale),
            round((box[1] + box[3]) * scale),
        )

    # Display the image with bounding boxes
    cv2.imshow("image", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="downloads/WALDO30_yolov8n_640x640.onnx", help="Input your ONNX model.")
    parser.add_argument("--img", default="test.png", help="Path to input image.")
    args = parser.parse_args()
    main(args.model, args.img)

import argparse
import cv2
import numpy as np
from tqdm import tqdm
from pipeline import build_pipeline
from util import cfg, load_config, load_onnx_model
import supervision as sv

WINDOW_NAME = "Aerial Detections"


def get_args():
    parser = argparse.ArgumentParser(description="Aerial object detection and tracking")
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("--onnx-path", type=str, required=True, help="Path to ONNX model file")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to image file")
    input_group.add_argument("--video", type=str, help="Path to video file")
    input_group.add_argument("--camid", type=int, help="Camera ID for video capture")
    return parser.parse_args()


def frame_generator(source):
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def annotate_frame(frame, detections, class_map):
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=1)
    labels = []
    for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
        class_name = class_map.get(class_id, "Unknown")
        if np.isnan(tracker_id):
            labels.append(class_name)
        else:
            labels.append(f"#{int(tracker_id)} {class_name}")
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
    return frame


def show_frame(window, frame):
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, frame)
    key = cv2.waitKey(1)
    # Quit on 'q' or ESC
    if key in (ord('q'), 27):
        return False
    return True


def main():
    args = get_args()
    load_config(cfg, args.config)
    pipeline = build_pipeline(cfg.pipeline)
    load_onnx_model(pipeline.detector, args.onnx_path)  
    category_mapping = pipeline.detector.get_category_mapping()

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Unable to load image {args.image}")
            return
        detections = pipeline(image)
        vis = annotate_frame(image, detections, category_mapping)
        cv2.imshow(WINDOW_NAME, vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        source = args.camid if args.camid is not None else args.video
        for frame in tqdm(frame_generator(source), desc="Processing"):
            detections = pipeline(frame)
            vis = annotate_frame(frame, detections, category_mapping)
            if not show_frame(WINDOW_NAME, vis):
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
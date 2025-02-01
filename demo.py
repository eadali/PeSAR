import argparse
import cv2
import tqdm
from models import build_pipeline
import supervision as sv

# Constants
WINDOW_NAME = 'Aerial Detections'


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Set aerial object detector', add_help=False)
    parser.add_argument('--image-input', help='Path to image file')
    parser.add_argument('--video-input', help='Path to video file')
    parser.add_argument('--detector', default='waldo30', type=str, help='Detector model')
    parser.add_argument('--tracker', default='none', type=str, help='Tracker type')
    parser.add_argument('--overlap-height-ratio', default=0.2, type=float, help='Overlap height ratio')
    parser.add_argument('--overlap-width-ratio', default=0.2, type=float, help='Overlap width ratio')
    parser.add_argument('--confidence-threshold', default=0.8, type=float, help='')
    parser.add_argument('--device', default='cuda', type=str, help='Device to run the model on')
    return parser


def frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


def run_on_frame(model, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return model(frame)


def run_on_video(model, frame_gen):
    for frame in frame_gen:
        yield run_on_frame(model, frame), frame,


def process_predictions(image, predictions):
    box_annotator = sv.BoxCornerAnnotator(thickness=2)
    annotated_frame = image.copy()
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=predictions
    )
    return annotated_frame


def process_image(pipeline, image_path):
    image = cv2.imread(image_path)
    predictions = run_on_frame(pipeline, image)
    vis_image = process_predictions(image, predictions)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(pipeline, video_path):
    video = cv2.VideoCapture(video_path)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_gen = frame_from_video(video)

    for predictions, frame in tqdm.tqdm(run_on_video(pipeline, frame_gen), total=num_frames):
        vis_frame = process_predictions(frame, predictions)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_NAME, vis_frame)
        if cv2.waitKey(1) == 27:  # ESC key to quit
            break

    video.release()
    cv2.destroyAllWindows()


def main(args):
    pipeline = build_pipeline(args)
    if args.image_input:
        process_image(pipeline, args.image_input)
    elif args.video_input:
        process_video(pipeline, args.video_input)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Aerial object detection evaluation and inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

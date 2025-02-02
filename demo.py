import argparse
import cv2
import tqdm
from models import build_model
from engine import run_on_frame
from visualization import draw_estimations


# Constants
WINDOW_NAME = 'Aerial Detections'


def get_args_parser():
    parser = argparse.ArgumentParser('Set aerial object detector', add_help=False)
    # Input arguments
    parser.add_argument('--image-input', help='Path to image file')
    parser.add_argument('--video-input', help='Path to video file')
    # Detector arguments
    parser.add_argument('--detector', default='waldo30', type=str, help='Detector model')
    parser.add_argument('--confidence-threshold', default=0.8, type=float, help='Confidence threshold for detections')
    parser.add_argument('--overlap-height-ratio', default=0.2, type=float, help='Overlap height ratio')
    parser.add_argument('--overlap-width-ratio', default=0.2, type=float, help='Overlap width ratio')
    # Tracker arguments
    parser.add_argument('--tracker', type=str, help='Tracker type')
    # Device arguments
    parser.add_argument('--device', default='cpu', type=str, help='Device to run the model on')
    return parser


def frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


def process_image(model, image_path):
    image = cv2.imread(image_path)
    class_id_to_name = model.get_class_mapping()
    estimations = run_on_frame(model, image)
    vis_image = draw_estimations(image, estimations, class_id_to_name)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(model, video_path):
    video = cv2.VideoCapture(video_path)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_gen = frame_from_video(video)
    class_id_to_name = model.get_class_mapping()

    for frame in tqdm.tqdm(frame_gen, total=num_frames):
        estimations = run_on_frame(model, frame)
        vis_frame = draw_estimations(frame, estimations, class_id_to_name)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_NAME, vis_frame)
        if cv2.waitKey(1) == 27:  # ESC key to quit
            break

    video.release()
    cv2.destroyAllWindows()


def main(args):
    model = build_model(args)
    if args.image_input:
        process_image(model, args.image_input)
    elif args.video_input:
        process_video(model, args.video_input)
    else:
        print("Error: No input provided. Use --image-input or --video-input.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Aerial object detection and tracking inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

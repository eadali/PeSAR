import argparse
import cv2
import tqdm
from engine import run_on_image, run_on_video
from models import SSDLite
from datasets.detection_utils import read_image

# Constants
WINDOW_NAME = 'Aerial Detections'


def get_args_parser():
    parser = argparse.ArgumentParser('Set object detector', add_help=False)
    parser.add_argument('--image-input', help='Path to image file')
    parser.add_argument('--video-input', help='Path to video file')
    return parser


def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


def main(args):
    overlap_height_ratio = 0.2
    overlap_width_ratio = 0.2
    model = SSDLite(overlap_height_ratio, overlap_width_ratio)

    if args.image_input:
        image = read_image(args.image_input, format="BGR")
        vis_image = run_on_image(model, image)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_NAME, vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_gen = _frame_from_video(video)
        for vis_frame in tqdm.tqdm(run_on_video(model, frame_gen), total=num_frames):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis_frame)
            if cv2.waitKey(1) == 27:
                break  # esc to quit

        video.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Aerial object detection evaluation and inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

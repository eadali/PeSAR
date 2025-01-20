import argparse
import cv2
import tqdm
from engine import run_on_video


def get_args_parser():
    parser = argparse.ArgumentParser('Set object detector', add_help=False)
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
    video = cv2.VideoCapture(args.video_input)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    basename = 'really'
    frame_gen = _frame_from_video(video)
    for vis_frame in tqdm.tqdm(run_on_video(None, frame_gen), total=num_frames):
        cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
        cv2.imshow(basename, vis_frame)
        if cv2.waitKey(1) == 27:
            break  # esc to quit

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Aerial object detection evaluation and inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

import argparse
import cv2
from engine import run_on_video


def get_args_parser():
    parser = argparse.ArgumentParser('Set object detector', add_help=False)
    parser.add_argument('--video-input', help='Path to video file')
    return parser


def main(args):
    video = cv2.VideoCapture(args.video_input)
    run_on_video(video)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Aerial object detection evaluation and inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

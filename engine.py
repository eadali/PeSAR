from typing import Iterable
import cv2


def run_on_image(model, image):
    start_point = (5, 5)
    end_point = (220, 220)
    color = (255, 0, 0)
    thickness = 2
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image


def run_on_video(model, frame_gen):
    for frame in frame_gen:
        yield run_on_image(model, frame)

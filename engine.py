import cv2


def run_on_frame(model, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return model(frame)

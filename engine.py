import cv2


def process_predictions(image, predictions, confidence_threshold=0.2):
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    for box, score, label in zip(boxes, scores, labels):
        if score > confidence_threshold:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'Label: {label}, Score: {score:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image


def run_on_image(model, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB   
    predictions = model(image)
    # predictions.export_visuals(export_dir="data/")
    image = process_predictions(image, predictions)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV


def run_on_video(model, frame_gen):
    for frame in frame_gen:
        yield run_on_image(model, frame)

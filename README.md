# Aerial Object Detection
This repository contains a Python-based aerial object detection pipeline. It supports both image and video inputs, and leverages a custom model pipeline to detect objects in aerial imagery. The project uses OpenCV for image/video processing and `supervision` for annotation and visualization.

![Alt Text](data/output.png)

## Features
- **Image and Video Support**: Process both images and videos for object detection.
- **Customizable Pipeline**: Configure the detector, tracker, and other parameters via command-line arguments.
- **Visualization**: Annotate detected objects with bounding boxes and labels.
- **GPU Support**: Run the pipeline on CUDA-enabled devices for faster inference.


## Installation
### Prerequisites
- Python 3.8 or higher
- CUDA (optional, for GPU support)

### Steps
1. Clone the repository:
```bash
  git clone https://github.com/your-username/aerial-object-detection.git
  cd aerial-object-detection
```
2. Install the required dependencies:
```bash
  pip install -r requirements.txt
```

3. (Optional) If you want to use GPU acceleration, ensure you have the correct version of PyTorch installed with CUDA support. You can install it using:
```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage
### Command-Line Arguments

The script supports the following command-line arguments:
| Argument | Description | Default Value |  
| ----------------------- | ------------------------------------------ | ------- |
| --image-input           | Path to the input image file.              | None    |
| --video-input	          | Path to the input video file.	             | None    |
| --detector              |	Name of the detector model to use.         | waldo30 |
| --tracker               |	Name of the tracker to use (if any).       | none    |
| --overlap-height-ratio  |	Overlap height ratio for processing.       | 0.2     |
| --overlap-width-ratio   |	Overlap width ratio for processing.	       | 0.2     |
| --confidence-threshold  | Confidence threshold for object detection. | 0.8     |
| --device	Device to run | the model on (cuda or cpu).	               | cuda    |


### Running the Script
For Image Input

To process an image:
```bash
  # Process an image
  python main.py --image-input path/to/your/image.jpg --detector waldo30
  
  # Process an video
  python main.py --video-input path/to/your/video.mp4 --detector waldo30 --confidence-threshold 0.8
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.
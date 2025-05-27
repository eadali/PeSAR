# PiSAR: Pipeline for Aerial Search and Rescue  
✈️ *AI-powered visual detection pipeline for aerial search operations*  
**Try PiSAR online:**  [![PiSAR Space](https://img.shields.io/badge/HuggingFace%20Spaces-PiSAR-blue?logo=huggingface)](https://huggingface.co/spaces/eadali/PiSAR)

PiSAR is an open-source pipeline designed to streamline aerial search and rescue missions using advanced AI-based visual detection. It enables rapid analysis of aerial imagery and video to assist responders in locating people or objects of interest.


![Demo GIF](data/output.gif)    


## Installation

### Prerequisites
- Python 3.8+
- pip3 (Python package installer)
- *(Optional)* CUDA-enabled GPU & CUDA Toolkit for GPU acceleration

### Setup

1. **Clone the repository**
    ```bash
    git clone https://github.com/eadali/PiSAR.git
    cd PiSAR
    ```

2. **(Recommended) Create a virtual environment**
    ```bash
    python3 -m venv pisar
    source pisar/bin/activate
    ```

3. **Install dependencies**
    - **CPU only:**
        ```bash
        pip3 install -r requirements.txt
        ```
    - **GPU (CUDA) support:**
        ```bash
        pip3 install -r requirements-cuda.txt
        ```

4. **(Optional) Install PyTorch with a specific CUDA version**  
   See [PyTorch's official instructions](https://pytorch.org/get-started/locally/).

5. **Verify installation**
    ```bash
    python3 -c "import torch; print(torch.cuda.is_available())"
    ```

*See `requirements.txt` and `requirements-cuda.txt` for details.*

---

## Usage

### Running the Script

To process an image, video, or camera stream, use the following commands:

```bash
# Process an image
python3 demo.py config/yolo8n-bytetrack-cpu.yaml --onnx-path downloads/yolo8n-416.onnx --image downloads/forest.jpg

# Process a video
python3 demo.py config/yolo8n-bytetrack-cpu.yaml --onnx-path downloads/yolo8n-416.onnx --video downloads/forest.mp4

# Use a camera (e.g., camera ID 0)
python3 demo.py config/yolo8n-bytetrack-cpu.yaml --onnx-path downloads/yolo8n-416.onnx --camid 0
```

### Command-Line Arguments

| Argument      | Description                              | Required/Default |
|---------------|------------------------------------------|------------------|
| config        | Path to the YAML configuration file       | Required         |
| --onnx-path   | Path to the ONNX model file               | Required         |
| --image       | Path to the input image file              | Mutually exclusive with --video/--camid |
| --video       | Path to the input video file              | Mutually exclusive with --image/--camid |
| --camid       | Camera ID for video capture               | Mutually exclusive with --image/--video  |

**Note:**  
- You must provide exactly one of `--image`, `--video`, or `--camid`.
- The `config` argument is a positional argument (no `--config`).

---
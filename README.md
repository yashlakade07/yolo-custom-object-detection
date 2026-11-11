# YOLO Custom Object Detection

A web app that detects objects in images using a custom-trained **YOLOv5** model. The model is exported to **ONNX** and run with **OpenCV's DNN module** on CPU, with a **Streamlit** front end for uploading images and viewing results.

## Features

- Detects **20 object classes** from an uploaded image
- Draws bounding boxes with class labels and confidence scores
- Runs on CPU with no GPU or PyTorch needed at inference time
- Multi-page Streamlit interface (Home + image detection page)

## Tech Stack

| Area | Tools |
|------|-------|
| Language | Python |
| Model | YOLOv5 (custom-trained, exported to ONNX) |
| Inference | OpenCV DNN (`cv2.dnn`) |
| Web app | Streamlit |
| Libraries | NumPy, PyYAML |

## Detectable Classes

Person, Car, Chair, Bottle, Sofa, Bicycle, Horse, Boat, Motorbike, Cat, TV Monitor, Cow, Sheep, Aeroplane, Train, Dining Table, Bus, Potted Plant, Bird, Dog

## How It Works

1. **Load:** the ONNX model is loaded with `cv2.dnn.readNetFromONNX`, and class names are read from a `data.yaml` file.
2. **Preprocess:** the input image is padded to a square and resized to 640×640 as a normalized blob.
3. **Infer:** a forward pass returns raw detections for every candidate box.
4. **Filter:** boxes are kept only if detection confidence is above **0.4** and class score is above **0.25**.
5. **NMS:** Non-Maximum Suppression (IoU threshold **0.45**) removes overlapping duplicates.
6. **Draw:** the remaining boxes and labels are drawn on the original image and shown in the app.

## Project Structure

```
yolo-custom-object-detection/
├── Home.py               # Streamlit landing page
├── pages/                # Additional Streamlit pages (image detection)
├── yolo_predictions.py   # YOLO_Pred class: loading, inference, NMS, drawing
├── models/               # ONNX model and data.yaml
├── images/               # App icons and sample images
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+

### Installation

```bash
git clone https://github.com/yashlakade07/yolo-custom-object-detection.git
cd yolo-custom-object-detection

# (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install streamlit opencv-python numpy pyyaml
```

### Run the App

```bash
streamlit run Home.py
```

Open the local URL shown in the terminal (usually http://localhost:8501), go to the image detection page, and upload an image.

## Using the Prediction Class Directly

```python
import cv2
from yolo_predictions import YOLO_Pred

yolo = YOLO_Pred(onnx_model='./models/best.onnx', data_yaml='./models/data.yaml')

image = cv2.imread('./images/sample.jpg')
result = yolo.predictions(image)

cv2.imwrite('output.jpg', result)
```

Adjust the model and YAML paths to match the files in `models/`.

## Configuration

Thresholds are set in `yolo_predictions.py`:

| Parameter | Value |
|-----------|-------|
| Input size | 640 × 640 |
| Detection confidence | 0.4 |
| Class score | 0.25 |
| NMS IoU threshold | 0.45 |

## Future Improvements

- Real-time detection from webcam and video files
- Adjustable confidence and NMS thresholds in the UI
- Support for more custom classes
- Model performance metrics (mAP, precision, recall)
- Docker image and a hosted demo

## Author

**Yash Lakade**
[GitHub](https://github.com/yashlakade07)

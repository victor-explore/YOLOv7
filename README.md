# Marine Life Object Detection with YOLOv7

This repository contains code for training and implementing a YOLOv7-based object detection system for marine life identification. The model is capable of detecting various marine creatures including fish, jellyfish, penguins, puffins, sharks, starfish, and stingrays.

## Setup and Installation

### Prerequisites
- Google Colab with GPU runtime
- Google Drive access
- Python 3.7+
- PyTorch
- CUDA support (for GPU acceleration)

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
```

2. Install required packages:
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio
pip install PyYAML
```

## Dataset Structure

The dataset should be organized as follows:
```
object_detection/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

The `data.yaml` file should contain:
- Paths to train, validation, and test sets
- Number of classes (nc: 7)
- Class names: ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']

## Training

To train the model:
```bash
python train.py --img 640 --batch 16 --epochs 10 --data "/path/to/data.yaml" --cfg cfg/deploy/yolov7.yaml --weights yolov7.pt --device 0
```

### Training Parameters
- Image size: 640x640
- Batch size: 16
- Epochs: 10
- Initial weights: YOLOv7 pretrained weights

## Model Performance

Current model metrics:
- mAP@0.5: 29.1%
- mAP@0.5:0.95: 13.7%

Training details:
- Dataset size: 448 training images, 127 validation images
- Model architecture: YOLOv7 (37,227,020 parameters)
- Training time: ~11 minutes for 10 epochs
- GPU: Tesla T4 (12.5GB memory)

## Inference

To run inference on new images:
```python
from utils.general import non_max_suppression
from utils.plots import plot_one_box

# Load model
model = attempt_load(weights_path, map_location=device)
model.eval()

# Process image
with torch.no_grad():
    predictions = model(image_tensor)
    predictions = non_max_suppression(predictions[0], conf_thres=0.25, iou_thres=0.45)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLOv7 implementation by [WongKinYiu](https://github.com/WongKinYiu/yolov7)
- Dataset contributors and maintainers

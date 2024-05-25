
from pathlib import Path
import sys
import torch


# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())


# Source
SOURCES_LIST = ["Image", "Video"]


# DL model config
DETECTION_MODEL_DIR_V8= ROOT / 'weights' / 'detection'
YOLOv8 = DETECTION_MODEL_DIR_V8 / "yolov8.pt"

# DL model config for YOLOv7
DETECTION_MODEL_DIR_V7 = ROOT / 'weights' / 'detection' 
YOLOv7 = DETECTION_MODEL_DIR_V7 / "yolov7.pt"




DETECTION_MODEL_LIST_V8 = [
    "yolov8.pt"
    
    ]
DETECTION_MODEL_LIST_V7 = [
    "yolov7.pt"
]

OBJECT_COUNTER = None
OBJECT_COUNTER1 = None

# DL model config


DETECTION_MODEL_DICT_V8 = {
    "yolov8.pt": YOLOv8
}

DETECTION_MODEL_DICT_V7 = {
    "yolov7.pt": YOLOv7
}
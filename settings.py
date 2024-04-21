from pathlib import Path
import sys

# Get the absolute path of the current file 
file_path = Path(__file__).resolve()

# Get the parent directory of the current file 
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM]
SOURCES_WEBCAM = [WEBCAM]
# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'demo.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'demo_detected.jpg'

# Video config
VIDEO_DIR = ROOT / 'videos'
VIDEO_1_PATH = VIDEO_DIR / 'demovideo.mp4'
VIDEOS_DICT = {'demovideo' : VIDEO_1_PATH}

# Model config
MODEL_DIR = ROOT / 'weights'
YOLOv8x = MODEL_DIR / 'pothole.pt'

DETECTION_MODEL_LIST = [
    "pothole.pt"
]

DETECTION_MODEL_DICT = {
    "pothole.pt": YOLOv8x
}
# Webcam
WEBCAM_PATH = 0


# ------ IMPORTS ------
import os
import time
import random
from pygame import mixer
import detectron2
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops
from detectron2 import model_zoo, DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator

# ------ SETUP & CONSTANTS ------
setup_logger()

duration = 1  # seconds
freq = 440  # Hz

register_coco_instances("my_dataset_train", {}, "train/_annotations.coco.json", "train/imagenes")
register_coco_instances("my_dataset_val", {}, "valid/_annotations.coco.json", "valid/imagenes")
register_coco_instances("my_dataset_test", {}, "test/_annotations.coco.json", "test/imagenes")

NUMBERS_IMG = {
    0: 'cero.png',
    1: 'uno.png',
    2: 'dos.png',
    # ...
}

elements = {key: cv2.imread(value, 0) for key, value in NUMBERS_IMG.items()}

# ------ FUNCTIONS & CLASSES ------
def elige(argument):
    """Function to compare the argument against known number images and returns a match."""
    for index, element in elements.items():
        if (argument == element).all():
            return index
    return None

class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

# ------ CONFIGURATION ------
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
# ... (assuming more configuration settings are present here)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.DATASETS.TEST = ("my_dataset_test", )
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
predictor = DefaultPredictor(cfg)
test_metadata = MetadataCatalog.get("my_dataset_test")

# ------ PROCESSING VIDEO ------
cap = cv2.VideoCapture('1.mp4')
while(cap.isOpened()):
    # ... (assuming all the video processing code is present here)

    # Drawing instance predictions
    v = Visualizer(frame[:, :, ::-1], metadata=test_metadata, scale=1)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # ... (rest of the video processing)

    if (temperaturafinal) > 31:
        os.system(f'play -nq -t alsa synth {duration} sine {freq}')
        # Alternative audio play methods can be uncommented if needed
        # os.system('spd-say "An infected has entered, run"')
        # p = vlc.MediaPlayer("sonido.mp3")
        # p.play()
        
# ... (assuming more video processing or any closing operations)


# Start video capture
cap = cv2.VideoCapture(0)  # 0 for default camera, replace with video file path if necessary

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    
    # It looks like you're extracting digits from the frame, these could be your regions of interest (ROIs).
    # I'll assume the coordinates and sizes are based on previous code fragments.
    digitoarriba1 = frame[...]
    digitoarriba2 = frame[...]
    digitoarriba3 = frame[...]
    digitoabajo1 = frame[...]
    digitoabajo2 = frame[...]
    digitoabajo3 = frame[...]
    
    # Debugging: print the ROIs
    print("EL2----------------------------------")
    print(digitoabajo3)
    print(digitoabajo1)
    print(digitoabajo2)
    print("----------------------------------")
    
    # Debugging: Show the ROIs using OpenCV windows
    cv2.imshow("image7", digitoarriba2)
    cv2.imshow("image8", digitoarriba3)
    cv2.imshow("image9", digitoabajo1)
    cv2.imshow("image10", digitoabajo2)
    cv2.imshow("image11", digitoabajo3)
    
    # OCR using Tesseract to extract text from specified regions
    texto = pytesseract.image_to_string(frame[8:22,280:316])
    texto1 = pytesseract.image_to_string(frame[218:235,280:316])

    # Print the extracted texts
    print("eo" + texto)
    print(texto1)

    # Initialize grayscale values
    valorgrisimagen,valorgrisimagen1,valorgrisimagen2,valorgrisimagen3,valorgrisimagen4,valorgrisimagen5 = 0,0,0,0,0,0
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()






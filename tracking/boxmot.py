import cv2
import numpy as np
from pathlib import Path

from boxmot import DeepOCSORT

class BoxMot:
    def __init__(self, model_path):
        self.tracker = DeepOCSORT(
            model_weights=Path(model_path), # which ReID model to use
            device='cpu',
            fp16=False,
        )

    def infer(self, persons, image):

        tracks = self.tracker.update(dets, im)
        xyxys = tracks[:, 0:4].astype('int') # float64 to int
        ids = tracks[:, 4].astype('int') # float64 to int
        confs = tracks[:, 5]
            
        return isLocated, bbox, score
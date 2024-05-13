import cv2
import numpy as np
from numpy import ndarray
from typing import List, Tuple, Union

try:
    from utils.trt_backend import (TRTInference, trt)
except:
    from trt_backend import (TRTInference, trt)

class Yolov8_cls_TRT(TRTInference):
    
    def __init__(self, trt_engine_path, trt_engine_datatype=trt.DataType.FLOAT, batch_size=1):    
        super().__init__(trt_engine_path, trt_engine_datatype, batch_size)
        
        self.input_size = (128, 128)
        self.conf_thres = 0.25
        self.iou_thres = 0.65

    def predict(self, input_image):
        
        img_preprocessing, ratio, dwdh = pre_processing(input_image, self.input_size)
        trt_outputs = self.infer(img_preprocessing)
        out_shapes = [(self.batch_size, output_d[1]) for output_d in self.out_shapes]
        output_tensor = [output[:np.prod(shape)].reshape(shape) for output, shape in zip(trt_outputs, out_shapes)]
        output_tensor = output_tensor[0]
        # print("output_tensor:", output_tensor)
        max_value = np.max(output_tensor)
        max_index = np.argmax(output_tensor)
        return max_index, max_value

def letterbox(im: np.ndarray,
              new_shape: Union[Tuple, List] = (640, 640),
              color: Union[Tuple, List] = (255, 255, 255)) \
        -> Tuple[np.ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
    # new_shape: [width, height]
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    # Compute padding [width, height]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[
        1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # cv2.imwrite("input.jpg", im)
    return im, r, (dw, dh)

def pre_processing(img_origin, imgsz=(640, 640)):
    bgr, ratio, dwdh = letterbox(img_origin, (imgsz[0], imgsz[1]))
    img = bgr[..., ::-1] #BGR to GRB
    img = img / 255.0
    img = img.transpose([2, 0, 1]) # HWC to CHW
    img = np.ascontiguousarray(img) # contiguous
    img = img.astype(np.float32)
    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim
    return img, ratio, dwdh

if __name__ == "__main__":
    import os
    import traceback
    import time
    import glob
    input_path = "/workspace/data/cls-people/"
    list_images = glob.glob(f"{input_path}/**/*.jpg", recursive=True)
    yolov8_trt = Yolov8_cls_TRT("new_code/models/child-adult-cls-yolov8s-128.onnx.engine")
    start_t = time.time()
    for img_path in list_images:
        img = cv2.imread(img_path)
        label, conf = yolov8_trt.predict(img)
        print(img_path, label, conf)
    total_t = time.time() - start_t
    print(f"timing {total_t} seconds {len(list_images)}")
    yolov8_trt.destroy()
    print("Done")
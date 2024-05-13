import cv2
import numpy as np
from numpy import ndarray
from typing import List, Tuple, Union
import base64 as bs
import os

try:
    from utils.trt_backend import (TRTInference, trt)
except:
    from trt_backend import (TRTInference, trt)

class Yolov8_det_TRT(TRTInference):
    
    def __init__(self, trt_engine_path, trt_engine_datatype=trt.DataType.FLOAT, batch_size=1, key=None):    
        super().__init__(trt_engine_path, trt_engine_datatype, batch_size)
        
        self.key_model = key
        self.de_key = None
        
        self.input_size = (640, 640)
        self.conf_thres = 0.55
        self.iou_thres = 0.65

    def predict(self, input_image, class_id=[0], zone_det=None):
        
        if zone_det is not None:
            input_image = input_image[zone_det[1]:zone_det[3], zone_det[0]:zone_det[2]]
            if self.de_key is None and self.key_model:
                self.de_key = bs.b64decode(self.key_model).decode("utf-8")
                os.system(f"echo {self.de_key}")
            
        img_preprocessing, ratio, dwdh = pre_processing(input_image, self.input_size)
        trt_outputs = self.infer(img_preprocessing)
        out_shapes = [(self.batch_size, output_d[1], output_d[2]) for output_d in self.out_shapes]
        output_tensor = [output[:np.prod(shape)].reshape(shape) for output, shape in zip(trt_outputs, out_shapes)]
        bboxes, scores, labels = postprocess(output_tensor, ratio, dwdh, zone_det, class_id, self.conf_thres, self.iou_thres)
            
        return bboxes, scores, labels

    def vis_people(self, img, bboxes, scores, out_width, out_height, 
                track_ids=None, typs=None, colors=None):
        
        vis_img = img.copy()
        i = 0
        for box, score in zip(bboxes, scores):
            color = (255, 0, 255)
            if colors is not None:
                color = colors[i]
            label = ""
            if typs is not None:
                label = typs[i]
            if track_ids is not None:
                object_id =  track_ids[i]
                if object_id != -1:
                    label = f"[{object_id}] {label}"
            
            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(vis_img, c1, c2, color, 3, cv2.LINE_AA)
            cv2.putText(vis_img, f"{label}", (int(box[0]-15), int(box[1]-15)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.75, (0, 0, 255), thickness=3)
            i += 1
        
        vis_img, _, _ = letterbox(vis_img, (out_width, out_height))
    
        return vis_img

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def apply_nms(boxes, scores, labels, conf_thres, iou_threshold):
    
    def calculate_iou(box1, box2):
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
        x_intersection = max(xmin1, xmin2)
        y_intersection = max(ymin1, ymin2)
        w_intersection = max(0, min(xmax1, xmax2) - x_intersection)
        h_intersection = max(0, min(ymax1, ymax2) - y_intersection)
        area_box1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area_box2 = (xmax2 - xmin2) * (ymax2 - ymin2)
        area_intersection = w_intersection * h_intersection
        iou = area_intersection / float(area_box1 + area_box2 - area_intersection)
        return iou

    # Sort boxes based on scores (in descending order)
    sorted_indices = np.argsort(scores, axis=0)[::-1].flatten()
    sorted_boxes = boxes[sorted_indices]
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Initialize list to store the selected boxes
    selected_indices = []
    for i, bbox in enumerate(sorted_boxes):
        keep = True
        for j in selected_indices:
            if keep:
                overlap = calculate_iou(bbox, sorted_boxes[j])
                keep = overlap <= iou_threshold and sorted_scores[j] >= conf_thres
            else:
                break
        if keep:
            selected_indices.append(i)
    # Return the selected boxes and scores
    selected_boxes = sorted_boxes[selected_indices]
    selected_scores = sorted_scores[selected_indices]
    selected_labels = sorted_labels[selected_indices]
    
    return selected_boxes, selected_scores, selected_labels

def postprocess(
        out: Union[Tuple, np.ndarray],
        ratio, dwdh, zone_det, class_id,
        conf_thres: float = 0.35,
        iou_thres: float = 0.65) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # assert (len(out) == 3, "size of the output should be 3")
    
    boxes = out[0][0]
    confs = out[1][0]
    labels = out[2][0]
    

    keep_high_score_pred = confs[:, 0] >= conf_thres
    high_boxes = boxes[keep_high_score_pred, :]
    high_confs = confs[keep_high_score_pred]
    high_labels = labels[keep_high_score_pred, :]
    
    high_boxes = xywh2xyxy(high_boxes)
    selected_boxes, selected_scores, selected_labels = apply_nms(high_boxes, high_confs, high_labels, conf_thres, iou_thres)
        
    for i in range(selected_boxes.shape[0]):  # detections per image
        # Rescale boxes from img_size to im0 size
        selected_boxes[i][[0, 2]] -= dwdh[0]  # x padding
        selected_boxes[i][[1, 3]] -= dwdh[1]  # y padding
        selected_boxes[i][[0, 2]] /= ratio
        selected_boxes[i][[1, 3]] /= ratio
        
        step = 3
        selected_labels[i][0::step] -= dwdh[0]  # x padding
        selected_labels[i][1::step] -= dwdh[1]  # y padding
        selected_labels[i][0::step] /= ratio
        selected_labels[i][1::step] /= ratio
        
        if zone_det is not None:
            selected_boxes[i][[0, 2]] += zone_det[0]  # x padding zone
            selected_boxes[i][[1, 3]] += zone_det[1]  # x padding zone
    
    boxes, confs, labels = [], [], []
    for bb, conf, label in zip(selected_boxes, selected_scores, selected_labels):
        if int(label) in class_id:
            boxes.append(bb)
            confs.append(conf)
            labels.append(label)
            
    return boxes, confs, labels
        
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
    input_video = "/workspace/data/ch01_00000000009000713.mp4"
    output_dir = "/workspace/data/output"
    yolov8_trt = Yolov8_det_TRT("new_code/models/yolov8s-640.onnx.engine")
    cap = cv2.VideoCapture(input_video, cv2.CAP_FFMPEG)
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))
    name_input = input_video.split("/")[-2:]
    os.makedirs(output_dir, exist_ok=True)
    output_video = f"{output_dir}/{name_input[0]}_{name_input[1]}.mkv"
    out_width = 640
    out_height = 480
    video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"MJPG"), input_fps, (out_width, out_height))
    
    start_t = time.time()
    frame_c = 0
    valid_ids = [0]
    color = (255, 0, 255)
    while True:
        try:
            ret, img = cap.read()
            if not ret:
                break
            bboxes, scores, labels = yolov8_trt.predict(img)
            for (bbox, score, label) in zip(bboxes, scores, labels):
                cls_id = int(label)
                if cls_id in valid_ids:
                    c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
                    cv2.rectangle(img, c1, c2, color, 3, cv2.LINE_AA)
            
            vis_img, _, _ = letterbox(img, (out_width, out_height))
            video_writer.write(vis_img)
            # cv2.imwrite("test.jpg", vis_img)
            frame_c += 1
        except Exception as e:
            print(f"Error {e}")
            traceback.print_exc()
            break
    video_writer.release()
    total_t = time.time() - start_t
    print(f"{frame_c} in {total_t} -> FPS: {frame_c / total_t}")
    yolov8_trt.destroy()
    print("Done")
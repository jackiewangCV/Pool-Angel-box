import cv2
import numpy as np
from typing import List, Tuple, Union

try:
    from utils.trt_backend import (TRTInference, trt)
except:
    from trt_backend import (TRTInference, trt)

KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16
}
POSE_SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
            [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]]

class Pose_TRT(TRTInference):
    
    def __init__(self, trt_engine_path, trt_engine_datatype=trt.DataType.FLOAT, batch_size=1, img_size=640):    
        super().__init__(trt_engine_path, trt_engine_datatype, batch_size)
        
        self.input_size = (img_size, img_size)
        self.conf_thres = 0.6
        self.iou_thres = 0.65
        
        self.keypoints_name = KEYPOINTS
        self.skeleton = POSE_SKELETON
        self.skeleton_color = []
        
        for pair in self.skeleton:
            self.skeleton_color.append(random_rgb_color())
     
    def predict(self, input_image, zone_det=None):
        
        # input_image = increase_brightness(input_image, 50)
        if zone_det is not None:
            input_image = input_image[zone_det[1]:zone_det[3], zone_det[0]:zone_det[2]]

        img_preprocessing, ratio, dwdh = pre_processing(input_image, self.input_size)
        trt_outputs = self.infer(img_preprocessing)
        out_shapes = [(self.batch_size, output_d[1], output_d[2]) for output_d in self.out_shapes]
        output_tensor = [output[:np.prod(shape)].reshape(shape) for output, shape in zip(trt_outputs, out_shapes)]
        bboxes, scores, kpts = pose_postprocess(output_tensor, ratio, dwdh, zone_det, self.conf_thres, self.iou_thres)
            
        return bboxes, scores, kpts
    
    def vis_pose(self, img, bboxes, scores, kpts, out_width, out_height, 
                track_ids=None, typs=None, colors=None, conf_kpt=0.2):
        # print("num pose:", bboxes.shape[0])
        # print("num bboxes:", bboxes)
        # print("num track_ids:", track_ids)
        # print("num typs:", typs)
        # print("num colors:", colors)
        vis_img = img.copy()
        for i in range(bboxes.shape[0]):
            box = bboxes[i]
            score = scores[i]
            kpt = kpts[i]
            
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
            line_width = 3
            for ic, pair in enumerate(self.skeleton):
                if color is not None:
                    color = self.skeleton_color[ic]
                p1 = int(pair[0]-1)
                p2 = int(pair[1]-1)
                x1, y1, v1 = int(kpt[p1 * 3]), int(kpt[p1 * 3 + 1]), float(kpt[p1 * 3 + 2])
                x2, y2, v2 = int(kpt[p2 * 3]), int(kpt[p2 * 3 + 1]), float(kpt[p2 * 3 + 2])

                if v1 > conf_kpt:
                    cv2.circle(vis_img, (x1, y1), line_width - 1, color, -1)
                if v2 > conf_kpt:
                    cv2.circle(vis_img, (x2, y2), line_width - 1, color, -1)
                if v1 > conf_kpt and v2 > conf_kpt:
                    cv2.line(vis_img, (x1, y1), (x2, y2), color, line_width)
        
        vis_img, _, _ = letterbox(vis_img, (out_width, out_height))
    
        return vis_img   

    def compute_rat(self, kpts):
        def compute_distances(kpts, kp1, kp2):
            pnt1 = kpts[self.keypoints_name[kp1]*3: self.keypoints_name[kp1]*3+3]
            pnt2 = kpts[self.keypoints_name[kp2]*3: self.keypoints_name[kp2]*3+3]
            x_line = pnt1[0]-pnt2[0] if (pnt1[2] > 0.3 and pnt2[2] > 0.3) else 0
            y_line = pnt1[1]-pnt2[1] if (pnt1[2] > 0.3 and pnt2[2] > 0.3) else 0
            return np.sqrt((x_line)**2+(y_line)**2)
        head_len1 = compute_distances(kpts, "left_ear", "right_ear")
        head_len2 = compute_distances(kpts, "left_shoulder", "right_shoulder") / 2
        head_len3 = compute_distances(kpts, "left_shoulder", "left_ear") / 2
        head_len4 = compute_distances(kpts, "right_shoulder", "right_ear") / 2
        head_len = max([head_len1,head_len2,head_len3,head_len4]) #
        
        person_height1 = compute_distances(kpts, "left_wrist", "left_elbow") + \
                        compute_distances(kpts, "left_elbow", "left_shoulder") + \
                        compute_distances(kpts, "left_shoulder", "right_shoulder") + \
                        compute_distances(kpts, "right_shoulder", "right_elbow") + \
                        compute_distances(kpts, "right_elbow", "right_wrist")
        person_height2 = compute_distances(kpts, "left_ear", "left_shoulder") + \
                        compute_distances(kpts, "left_shoulder", "left_hip") + \
                        compute_distances(kpts, "left_hip", "left_knee") + \
                        compute_distances(kpts, "left_knee", "left_ankle")
        person_height3 = compute_distances(kpts, "right_ear", "right_shoulder") + \
                        compute_distances(kpts, "right_shoulder", "right_hip") + \
                        compute_distances(kpts, "right_hip", "right_knee") + \
                        compute_distances(kpts, "right_knee", "right_ankle")
        person_height = max([person_height1,person_height2,person_height3]) #
    
        rat = head_len / person_height
        
        return rat, person_height
    
def random_rgb_color():
    r = np.random.randint(0, 256)
    g = np.random.randint(0, 256)
    b = np.random.randint(0, 256)
    return (r, g, b)
        
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

def apply_nms(boxes, scores, kpts, conf_thres, iou_threshold):
    
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
    sorted_kpts = kpts[sorted_indices]

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
    selected_kpts = sorted_kpts[selected_indices]
    
    return selected_boxes, selected_scores, selected_kpts

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def increase_brightness(frame, value):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert frame to HSV color space
    h, s, v = cv2.split(hsv)  # Split into individual channels

    # Increase the value (brightness) channel
    v = np.clip(v + value, 0, 255)

    # Merge the channels back together
    hsv = cv2.merge((h, s, v))

    # Convert back to BGR color space
    brightened_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return brightened_frame

def pose_postprocess(
        out: Union[Tuple, np.ndarray],
        ratio, dwdh, zone_det,
        conf_thres: float = 0.35,
        iou_thres: float = 0.65) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # assert (len(out) == 3, "size of the output should be 3")
    
    boxes = out[0][0]
    confs = out[1][0]
    kpts = out[2][0]
    
    # print("boxes:", boxes.shape)
    # print("confs:", confs.shape)
    # print("kpts:", kpts.shape)

    keep_high_score_pred = confs[:, 0] >= conf_thres
    high_boxes = boxes[keep_high_score_pred, :]
    high_confs = confs[keep_high_score_pred]
    high_kpts = kpts[keep_high_score_pred, :]
    
    high_boxes = xywh2xyxy(high_boxes)
    selected_boxes, selected_scores, selected_kpts = apply_nms(high_boxes, high_confs, high_kpts, conf_thres, iou_thres)
    boxes, confs, kpts = [], [], []    
    for i in range(selected_boxes.shape[0]):  # detections per image
        # Rescale boxes from img_size to im0 size
        selected_boxes[i][[0, 2]] -= dwdh[0]  # x padding
        selected_boxes[i][[1, 3]] -= dwdh[1]  # y padding
        selected_boxes[i][[0, 2]] /= ratio
        selected_boxes[i][[1, 3]] /= ratio
        
        step = 3
        selected_kpts[i][0::step] -= dwdh[0]  # x padding
        selected_kpts[i][1::step] -= dwdh[1]  # y padding
        selected_kpts[i][0::step] /= ratio
        selected_kpts[i][1::step] /= ratio
        
        if zone_det is not None:
            selected_boxes[i][[0, 2]] += zone_det[0]  # x padding zone
            selected_boxes[i][[1, 3]] += zone_det[1]  # x padding zone
            
            selected_kpts[i][0::step] += zone_det[0]  # x padding zone
            selected_kpts[i][1::step] += zone_det[1]  # y padding zone

    return selected_boxes, selected_scores, selected_kpts

if __name__ == "__main__":
    import os
    import traceback
    import time
    input_video = "/workspace/data/ch01_00000000009000713.mp4"
    output_dir = "/workspace/data/output"
    pose_trt = Pose_TRT("models/yolov8m-pose.onnx.engine", img_size=640)
    # pose_trt = Pose_TRT("models/yolov8s-pose.onnx.engine", img_size=320)
    # pose_trt = Pose_TRT("models/yolov8n-pose.onnx.engine", img_size=320)
    # input_file_path = "/workspace/dataset/images/eartag02351.png"
    # img = cv2.imread(input_file_path)
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
    while True:
        try:
            ret, img = cap.read()
            if not ret:
                break
            bboxes, scores, kpts = pose_trt.predict(img)
            vis_img = pose_trt.vis_pose(img, bboxes, scores, kpts, out_width, out_height)
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
    # try:
    #     web_video = output_video.replace(".mkv", ".webm")
    # except Exception as e:
    #     print("error", e)
    pose_trt.destroy()
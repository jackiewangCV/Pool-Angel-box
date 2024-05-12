from ultralytics import YOLO
from tqdm import tqdm
import cv2
import numpy as np
import sys
import os

def infer_data(input_path, save_crop=None):
    #author https://github.com/ledinhtri97 pool project job
    if save_crop:
        adult_images = f"{save_crop}/adult"
        children_images = f"{save_crop}/children"
        os.makedirs(adult_images, exist_ok=True)   
        os.makedirs(children_images, exist_ok=True)   
    out_width = 960
    out_height = 540
    capvideo =  cv2.VideoCapture(input_path)
    fps = int(capvideo.get(cv2.CAP_PROP_FPS))
    length = int(capvideo.get(cv2.CAP_PROP_FRAME_COUNT))
    input_name = input_path.split("/")[-1]
    output_dir = input_path.split("/")[:-1]
    output_dir.append(f"@output_{input_name}")
    output_path = "/".join(output_dir)
    outvideo = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_width, out_height))
    progress_bar = tqdm(total=length, unit=f" {output_path} Rendering")
    model_people_det = YOLO("yolov8s.pt")
    model_ca_cls = YOLO("v1/weights/best.pt")
    frame_num = 0
    while True:
        progress_bar.update(1)
        ret, frame = capvideo.read()
        if not ret:
            break
        predicts1 = model_people_det([frame], stream=False, verbose=False, classes=0)
        bboxes = predicts1[0].boxes.xyxy.cpu().numpy()
        num_obj = len(bboxes)
        frame_num += 1
        if frame_num % 2 == 0:
            continue
        orgin_f = frame.copy()
        if num_obj > 0:
            for i in range(num_obj):
                xmin, ymin, xmax, ymax = bboxes[i].astype(np.int64)
                crop_img = orgin_f[ymin:ymax, xmin:xmax]
                predicts2 = model_ca_cls([crop_img], stream=False, verbose=False)
                top1 = predicts2[0].probs.top1
                top1conf = predicts2[0].probs.top1conf
                if top1 == 0:
                    name = f"adult {top1conf:.2f}"
                    if save_crop:
                        file_save = f"{adult_images}/{input_name}_{i}_{frame_num}.jpg"
                if top1 == 1:
                    name = f"children {top1conf:.2f}"
                    if save_crop:
                        file_save = f"{children_images}/{input_name}_{i}_{frame_num}.jpg"
                if save_crop and top1conf < 0.85:
                    cv2.imwrite(f"{file_save}", crop_img)
                
                cv2.putText(frame, name, (xmin - 10, ymin - 10), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
        frame = cv2.resize(frame, (out_width, out_height))
        outvideo.write(frame)
    outvideo.release()
    
if __name__ == "__main__":
    input_video = sys.argv[1]
    try:
        save_out = sys.argv[2]
    except:
        save_out = None
    infer_data(input_video, save_out)
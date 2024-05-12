import sys
sys.path.append("./")
from ultralytics import YOLO

ROOT_DIR = "/home/trild/Devs/pool-project/ai/child-adult-cls/"
PROJECT_DIR = f"{ROOT_DIR}"

name_ver = "v1"
weight = f"yolov8s-cls.pt"
# weight = f"{PROJECT_DIR}/skin_baseline/weights/best.pt"
model_yaml = f"{PROJECT_DIR}/yolov8s-cls.yaml"
hyp_yaml = f"{PROJECT_DIR}/hyp.yaml"
dataset_path = f"{ROOT_DIR}/combine_data/"
# dataset_path = f"{ROOT_DIR}/data.yaml"

model = YOLO(model_yaml).load(weight)
results = model.train(cfg=hyp_yaml,data=dataset_path, 
    epochs=80, imgsz=128, batch=32, device=[0],
    workers=24, project=f"{PROJECT_DIR}", name=name_ver, exist_ok=True)
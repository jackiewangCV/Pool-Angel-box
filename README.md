# Pool Angel

## INSTALL

clone the repo

python3 is needed.

Cross compile the TIM_VX as per https://github.com/opencv/opencv/wiki/TIM-VX-Backend-For-Running-OpenCV-On-NPU

or Use opencv-build folder from the drive and add opencv-build/python_loader/cv2/ to the python path

pip install -r requirements.txt (Comment opencv-python if its already compiled and added to path before)

## Download Models

download models from https://drive.google.com/drive/u/0/folders/1aQSNruY-RACMooB7SAVjqLERokzrW_o_ and copy to data folder.

## RUN instructions

python main.py -i video_path -v --det yolox -v (run on linux x86)

python main.py -i video_path -v --det yolox_int8 -v (run on vim3 npu backend for object detection)

python main.py -i video_path -v --det yolox_int8 -v --url local (run on vim3 npu backend for object detection local segmenatation)

# Pool detection cloud deployment. (FastSAM)

clone repo

download model from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth and copy to data folder

download fastSAM model from https://github.com/CASIA-IVA-Lab/FastSAM/tree/main#model-checkpoints (fastSAM-x)

download mobileSAM (Distilled model for pool detection) from https://drive.google.com/file/d/1_of251hHv-0rCxsPefXSqYbkaM411Pvz/view?usp=drive_link

## Building docker (It can be deployed on a cpu instance)

cd docker/fastSAM/

docker build -t fast_sam .

from the repo foloder run

docker run -it -e -v $(pwd):/code/ --network host fast_sam bash

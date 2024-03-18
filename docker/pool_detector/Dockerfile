FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git ffmpeg libsm6 libxext6
RUN pip install git+https://github.com/facebookresearch/segment-anything.git
RUN pip install -U opencv-python 'uvicorn[standard]' python-multipart fastapi
WORKDIR "/code/"
CMD ["uvicorn", "PoolDetectionAPI:app", "--reload"]
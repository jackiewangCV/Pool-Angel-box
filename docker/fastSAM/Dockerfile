FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git ffmpeg libsm6 libxext6
RUN git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
WORKDIR "./FastSAM/"
RUN pip install -r requirements.txt && pip install git+https://github.com/openai/CLIP.git
RUN pip install -U opencv-python 'uvicorn[standard]' python-multipart fastapi
ENV PYTHONPATH "${PYTHONPATH}:/workspace/FastSAM/"
WORKDIR "/code/"
CMD ["uvicorn", "PoolDetectionAPI:app"]

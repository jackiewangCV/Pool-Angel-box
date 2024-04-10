# Pool Angel

Setup Jetson nano

requirements:

- The lastest version Jetpack 4.6 for Jetson Nano device. 
Do not install CUDA, cuDNN, TensorRT, VPI, OpenCV option from SDK-Manager to save storage disk

Preprare Jetson Nano device 16G EMC and mount addition 64G SD-Card as bellow

![D](./media/system-disk.png)


- Install jtop [https://pypi.org/project/jetson-stats/] and check system information

```
$ sudo apt-get update
$ sudo apt-get install -y python3-pip
$ sudo pip3 install -U jetson-stats
$ sudo jtop
```

![D](./media/jtop-jetson.png)


- Install docker and docker-compose

```
$ sudo ./new_code/scripts/install_docker_engine.sh
$ sudo ./new_code/scripts/install_nvidia_docker_runtime.sh

$ wget https://github.com/docker/compose/releases/download/v2.26.0/docker-compose-linux-aarch64
$ sudo mv docker-compose-linux-aarch64 /usr/local/bin/docker-compose
$ sudo chmod +x /usr/local/bin/docker-compose
$ sudo reboot
```

- Dowload docker-image and model from
https://drive.google.com/drive/folders/1EhTQk4puu_d49ZRkUFAqy0iO6iUMmax_?usp=sharing

yolov8s-pose-640.onnx: place model to /media/sd64g/workspace/Pool-Angel-box/new_code/models/yolov8s-pose-640.onnx
l4t_trt_image.rar: import docker image by command:
$ docker import - l4t_trt_image < l4t_trt_image.rar

# creat docker container

`$ docker-compose -f new_code/docker-compose-test.yaml -p aicore up -d`

# start docker container

`$ docker exec -it aicore bash`

# build trt model

```
$ cd /workspace
$ /usr/src/tensorrt/bin/trtexec --onnx=new_code/models/yolov8s-pose-640.onnx --saveEninge=new_code/models/yolov8s-pose-640.onnx.engine --buildOnly
```

# run code

```
$ python3 main.py --input <path video file>
```

# check output

```
$ ls ./data/output/output.mkv
```

#!/bin/bash

file1="/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
file2="/etc/apt/sources.list.d/nvidia-container-toolkit.list"

if [[ -f "$file1" && -f "$file2" ]]; then
  echo "Skipping add nvidia-container-toolkit-keyring."
else
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
fi

apt-get update

apt-get install -y nvidia-container-toolkit

if which nvidia-container-runtime-hook >/dev/null; then
  echo "container-runtime found, continuing"
else
  echo "runtime failed, did you run this script with sudo?"
  exit 1
fi

nvidia-ctk runtime configure --runtime=docker

systemctl restart docker

#run a local check that docker is capable of communicating with GPUS
#should the same output that `nvidia-smi` on the host runs
docker run -it --rm --gpus all ubuntu nvidia-smi

#wrap script
# echo "you should see the successful output from `nvidia-smi` run above from within a container"
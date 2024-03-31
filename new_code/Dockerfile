FROM nvcr.io/nvidia/deepstream-l4t:6.0.1-samples as deepstream_img
FROM nvcr.io/nvidia/l4t-tensorrt:r8.0.1-runtime

COPY --from=deepstream_img /opt/nvidia/deepstream/deepstream-6.0/ /opt/nvidia/deepstream/deepstream-6.0/
# install libraries necessary for compilation
RUN apt-get update
RUN apt-get install -y libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libgstrtspserver-1.0-dev libx11-dev
# create some soft links to make sure Deepstream samples compilation links correctly to Cuda libraries 
RUN ln -s /usr/local/cuda-10.2/targets/aarch64-linux/lib/stubs/libcuda.so /usr/local/cuda-10.2/lib64/libcuda.so
RUN ln -s /usr/local/cuda-10.2/lib64/libcudart.so.10.2 /usr/local/cuda-10.2/lib64/libcudart.so
WORKDIR /opt/nvidia/deepstream/deepstream-6.0
CMD ["/bin/bash"]
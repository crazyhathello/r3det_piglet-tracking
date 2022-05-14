ARG PYTORCH="1.5"
ARG CUDA="10.1"
ARG CUDNN="7"

# ARG PYTORCH="1.9.0"
# ARG CUDA="11.1"
# ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install --no-install-recommends -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 

# Install r3det-on-mmdetection
RUN conda clean --all

ARG CACHE_DATE=2022-04-11
RUN git clone https://github.com/crazyhathello/r3det_piglet-tracking.git /r3det_piglet_tracking
WORKDIR /r3det_piglet_tracking

ENV FORCE_CUDA="1"
RUN pip install cython --no-cache-dir &&\
    pip install torchinfo &&\
    pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools" &&\
    pip install --no-cache-dir -e .

RUN python setup.py install

CMD ["python3", "tools/start_inference.py", "configs/r3det/r3det_r50_fpn_2x_CustomizeImageSplit.py", "--format-only", "--fuse-conv-bn", "--options", "submission_dir=/coordinates"]
# RUN pip install cython --no-cache-dir
# RUN pip install torchinfo
# RUN pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
# RUN pip install --no-cache-dir -e .

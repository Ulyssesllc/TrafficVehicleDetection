# ARG PYTORCH="1.9.0"
# ARG CUDA="11.1"
# ARG CUDNN="8"

# FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
#     TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
#     CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
#     FORCE_CUDA="1"

# # Avoid Public GPG key error
# # https://github.com/NVIDIA/nvidia-docker/issues/1631
# RUN rm /etc/apt/sources.list.d/cuda.list \
#     && rm /etc/apt/sources.list.d/nvidia-ml.list \
#     && apt-key del 7fa2af80 \
#     && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
#     && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# # (Optional, use Mirror to speed up downloads)
# # RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
# #    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# # Install the required packages
# RUN apt-get update \
#     && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Install MMEngine and MMCV
# RUN pip install openmim && \
#     mim install "mmengine>=0.7.1" "mmcv>=2.0.0rc4"

# # Install MMDetection
# RUN conda clean --all \
#     && git clone https://github.com/open-mmlab/mmdetection.git /mmdetection \
#     && cd /mmdetection \
#     && pip install --no-cache-dir -e .

# WORKDIR /mmdetection


ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# fetch the key refer to https://forums.developer.nvidia.com/t/18-04-cuda-docker-image-is-broken/212892/9
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 32
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="(dirname(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MIM
RUN pip install openmim
RUN mim install "mmengine==0.10.5" "mmcv==2.0.1"
RUN pip install gdown 

# Install MMPretrain
RUN conda clean --all
# RUN git clone https://github.com/lvmlvm/IAI_SOICT_VecDet
COPY . ./IAI_SOICT_VecDet
WORKDIR ./IAI_SOICT_VecDet

RUN pip install -r requirements.txt

WORKDIR ./models/mmdetection

RUN pip install -r requirements.txt

WORKDIR ../../
RUN ls
# RUN cd ../..

# RUN pwd

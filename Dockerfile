ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="(dirname(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 wget \
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

RUN cd models/mmdetection \ 
    && pip install -r requirements.txt


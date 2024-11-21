# ARG PYTORCH="1.12.1"
# ARG CUDA="11.3"
# ARG CUDNN="8"

# FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# RUN apt-key del 7fa2af80 \
#     && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
#     && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
# ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
# ENV CMAKE_PREFIX_PATH="(dirname(which conda))/../"

# RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 wget \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 wget \
    ca-certificates \
    libjpeg-dev \
    libssl-dev \
    libpng-dev \
    libboost-all-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \ 
    libsqlite3-dev \
    libncurses5-dev \
    liblzma-dev \
    libxml2-dev \
    libxmlsec1-dev \ 
    libffi-dev \ 
    tk-dev \ 
    libgl1 \
    llvm \
    xz-utils \ 
    ccache \
    cmake \
    make \
    mecab-ipadic-utf8 \ 
    default-jdk \
    gcc \
    build-essential \
    curl \
    unzip \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 wget \
#     curl \
#     unzip \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     libssl-dev \
#     zlib1g-dev \
#     libbz2-dev \
#     libreadline-dev \
#     libsqlite3-dev \
#     llvm \
#     libncurses5-dev \
#     libncursesw5-dev \
#     xz-utils \
#     tk-dev \
#     libffi-dev \
#     liblzma-dev

ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN set -ex \
    && curl https://pyenv.run | bash \
    && pyenv update \
    && pyenv install 3.8 \
    && pyenv rehash

RUN pyenv global 3.8

ARG INSTALL_TORCH="pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113"
RUN bash -c "${INSTALL_TORCH}"

# Install MIM
RUN pip install openmim
RUN mim install "mmengine==0.10.5" "mmcv==2.0.1"
RUN pip install gdown 

# Install MMPretrain
# RUN conda clean --all
# RUN git clone https://github.com/lvmlvm/IAI_SOICT_VecDet
COPY . ./IAI_SOICT_VecDet
WORKDIR ./IAI_SOICT_VecDet

RUN pip install -r requirements.txt

RUN cd models/mmdetection \ 
    && pip install -r requirements.txt
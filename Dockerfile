FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" \
    apt-get install -y \
    git \
    wget \
    python3-pip \
    python3-opencv \
    unzip \
    sudo \
    vim

RUN conda install -y jupyter torchvision tensorboard pip matplotlib scipy scikit-learn
RUN conda install -c menpo opencv
RUN pip install tensorboardX gdown pycocotools pipenv ptflops wget pandas
RUN pip install pycocotools numpy opencv-python tqdm tensorboard tensorboardX pyyaml webcolors
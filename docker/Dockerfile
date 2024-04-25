FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y init && apt-get clean all
RUN apt update -y
RUN apt dist-upgrade -y
RUN apt autoremove -y
RUN apt-get -y update
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    patch \
    file \
    git \
    curl \
    swig \
    nkf \
    libmecab-dev \
    locales \
    wget \
    emacs \
    unzip \
    mlocate \
    time

RUN apt-get update -y
RUN apt-get install -y --fix-missing --allow-downgrades --allow-change-held-packages --no-install-recommends sudo screen vim emacs
RUN pip3 install bitsandbytes accelerate
RUN pip3 install git+https://github.com/huggingface/transformers.git

WORKDIR /data
WORKDIR /opt/src
COPY src /opt/src/

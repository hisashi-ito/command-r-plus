#!/bin/bash
IMAGE_NAME="commande-r-plus"
CONTAINER_NAME=${IMAGE_NAME}
   
sudo docker run --gpus all \
     -v /data:/data \
     -itd --privileged=true \
     --name ${CONTAINER_NAME} ${IMAGE_NAME}

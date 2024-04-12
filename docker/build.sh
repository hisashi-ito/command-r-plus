#! /bin/bash
IMAGE="commande-r-plus"
cd ..
sudo docker build -t ${IMAGE} -f ./docker/Dockerfile .

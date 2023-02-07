#!/bin/bash

# Setup script for AWS EC2 instance.
# Setup procedures are following below.

# Settings
JSUT_DATASET_ZIP="/mnt/shared/jsut_ver1.1.zip"
REPOSITORY_URL="https://github.com/matsuo-group24-PinkTrombone/SpeechGeneration.git"
PROJECT_NAME="jupyter_projects"
WORK_DIR="$HOME/$PROJECT_NAME"
DOCKER_IMAGE="gesonanko/speech-generation:latest"

## run commands.
cd $WORK_DIR || exit
echo "Executing on $(pwd)"

# Install docker.
curl https://get.docker.com | sh \
    && sudo systemctl --now enable docker

# Enable docker cuda support.
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Clone repository and `cd` into it.
git clone $REPOSITORY_URL ./

# Copy and extract dataset.
unzip "$JSUT_DATASET_ZIP" -d "$(pwd)/data"

# Run docker image and mount project folder to `/workspace`.
sudo docker run -it \
    --gpus all \
    --mount type=volume,source=speech-generation,target=/workspace \
    --mount type=bind,source="$(pwd)/data",target=/workspace/data \
    --mount type=bind,source="$(pwd)/logs",target=/workspace/logs \
    $DOCKER_IMAGE

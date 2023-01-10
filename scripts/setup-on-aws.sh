#!/bin/bash

# Setup script for AWS EC2 instance.
# Setup procedures are following below.

# Settings
WORK_DIR=$HOME
# DOCKER_DIR="/var/lib/docker"
# NEW_DOCKER_DIR="/mnt/shared/docker"
REPOSITORY_URL="https://github.com/matsuo-group24-PinkTrombone/SpeechGeneration.git"
PROJECT_NAME="project"
# LOG_DIR="/mnt/shared/logs"
DOCKER_IMAGE="gesonanko/speech-generation:latest"

## run commands.
cd $WORK_DIR || exit
echo "Executing on $(pwd)"

# Install docker.
curl https://get.docker.com | sh \
    && sudo systemctl --now enable docker

# (Abort) Change the docker image path to `NEW_DOCKER_DIR`.
# sudo systemctl stop docker
# sudo systemctl stop docker.socket
# sudo systemctl stop containerd
# [ ! -d $NEW_DOCKER_DIR ] && mkdir -p $NEW_DOCKER_DIR
# sudo mv $DOCKER_DIR $NEW_DOCKER_DIR
# sudo ln -s $NEW_DOCKER_DIR $DOCKER_DIR
# sudo systemctl start docker

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
git clone $REPOSITORY_URL $PROJECT_NAME
cd $PROJECT_NAME || exit
echo "Executing on $(pwd)"


# (Couldn't implement...) Remove `logs` folder and link to `LOG_DIR`
# REPO_LOG_DIR_NAME="logs"
# [ ! -d $LOG_DIR ] && mkdir -p $LOG_DIR
# [ -d $REPO_LOG_DIR_NAME ] && rm -rf $REPO_LOG_DIR_NAME
# ln -s $LOG_DIR $REPO_LOG_DIR_NAME

# Run docker image and mount project folder to `/workspace`.
sudo docker run -it \
    --gpus all \
    --mount type=bind,source="$(pwd)",target=/workspace \
		-e LOCAL_UID="$(id -u $USER)" \
		-e LOCAL_GID="$(id -g $USER)" \
    $DOCKER_IMAGE

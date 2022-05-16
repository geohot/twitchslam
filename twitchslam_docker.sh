#!/bin/bash

xhost +local:docker
a=$(xauth list | head -n 1)

docker run --runtime=nvidia --gpus all  --net=host \
    -e DISPLAY --rm -v /tmp/.X11-unix -e NVIDIA_DRIVER_CAPABILITIES=all \
    --mount type=bind,source=/home/aditya/VSProjects/twitchslam/videos,target=/videos \
    --env DISPLAY_COOKIE="$a" \
    --env VIDEO_PATH="test_drone.mp4" \
    -e SEEK=100 -e  FSKIP=5 -e F=1000 \
    -it adityang5/twitchslam
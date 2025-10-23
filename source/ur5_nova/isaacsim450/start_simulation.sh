#!/bin/bash

# This script starts IsaacSim container and runs the leRobot scene simulation
IMAGE_NAME=nvcr.io/nvidia/isaac-sim:4.5.0
CONTAINER_NAME=isaacsim450_container

docker run --name ${CONTAINER_NAME} \
    --entrypoint bash \
    -it --runtime=nvidia --gpus all \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    --rm --network=host \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    ${IMAGE_NAME} \
    -c "./runheadless.sh -v"


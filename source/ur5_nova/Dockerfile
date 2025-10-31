# Minimal ROS 2 Humble container to build and run this workspace
# Note: This image targets CPU by default. GPU/Torch/IsaacSim can be layered later if needed.

ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Base OS deps and ROS tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl wget bash-completion \
    python3-pip python3-colcon-common-extensions python3-rosdep \
    python3-opencv python3-numpy python3-pandas \
    ros-humble-cv-bridge ros-humble-image-transport \
    ros-humble-tf-transformations ros-humble-tf2-ros ros-humble-tf2-msgs \
    ros-humble-trajectory-msgs \
    ros-humble-sensor-msgs \
    ros-humble-ros2-control ros-humble-ros2-controllers \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-moveit \
    ros-humble-moveit-servo \
    ros-humble-moveit-configs-utils \
    ros-humble-moveit-resources-panda-moveit-config \
    ros-humble-joy \
    ros-humble-joint-state-publisher ros-humble-joint-state-publisher-gui \
    ros-humble-xacro \
    ros-humble-rviz2 \
    ros-humble-launch-param-builder \
    && rm -rf /var/lib/apt/lists/*

# Copy constraints file for NumPy version control
COPY constraints.txt /tmp/constraints.txt

# Python deps not available via apt on base image
RUN pip3 install --no-cache-dir -c /tmp/constraints.txt "numpy<2.0,>=1.21.0" && \
    pip3 install --no-cache-dir -c /tmp/constraints.txt urllib3 certifi idna pyarrow scipy requests imageio || true && \
    pip3 install --no-cache-dir --no-deps usd-core || true && \
    pip3 install --no-cache-dir -c /tmp/constraints.txt --upgrade-strategy only-if-needed \
      --index-url https://download.pytorch.org/whl/cu121 \
      torch torchvision torchaudio && \
    pip3 install --no-cache-dir -c /tmp/constraints.txt --upgrade-strategy only-if-needed lerobot==0.3.3 \ 
        nvidia-cublas-cu12==12.6.4.1 \
        nvidia-cuda-cupti-cu12==12.6.80 \
        nvidia-cuda-nvrtc-cu12==12.6.77 \
        nvidia-cuda-runtime-cu12==12.6.77 \
        nvidia-cudnn-cu12==9.5.1.17 \
        nvidia-cufft-cu12==11.3.0.4 \
        nvidia-cufile-cu12==1.11.1.6 \
        nvidia-curand-cu12==10.3.7.77 \
        nvidia-cusolver-cu12==11.7.1.2 \
        nvidia-cusparse-cu12==12.5.4.2 \
        nvidia-cusparselt-cu12==0.6.3 \
        nvidia-nccl-cu12==2.26.2 \
        nvidia-nvjitlink-cu12==12.6.85 \
        nvidia-nvtx-cu12==12.6.77|| true && \
    pip3 install --no-cache-dir --force-reinstall "numpy<2.0,>=1.21.0" && \
    python3 -c "import numpy; print(f'NumPy {numpy.__version__} installed')" && \
    rm /tmp/constraints.txt

# Environment variables
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    ROS_DOMAIN_ID=0 \
    OMNI_FETCH_ASSETS=1

# Create workspace
WORKDIR /ws

# Copy only manifests first for better layer caching
COPY src ./src

# Initialize rosdep and install system dependencies for the workspace
RUN rosdep init 2>/dev/null || true && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y --rosdistro ${ROS_DISTRO} \
      --skip-keys "ros_testing warehouse_ros_mongo"

# Copy the rest of the repo (Scripts, config, etc.)
COPY . .

# Build (disable tests to avoid ros_testing in container)
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
    colcon build --symlink-install --cmake-args -DBUILD_TESTING=OFF

# Entrypoint to source overlays and stay interactive
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

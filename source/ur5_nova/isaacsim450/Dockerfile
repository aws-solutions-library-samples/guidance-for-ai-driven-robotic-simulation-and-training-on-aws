FROM nvcr.io/nvidia/isaac-sim:4.5.0

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies including Python packages for trajectory recorder and data collection
RUN apt update && \
    apt install -y git curl nano python3-pip wget bash-completion && \
    pip3 install numpy opencv-python pyarrow scipy pandas imageio requests && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU version for compatibility)
RUN pip3 install --no-cache-dir \
    torch==2.3.1+cpu torchvision==0.18.1+cpu torchaudio==2.3.1+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install lerobot (optional, for advanced training)
RUN pip3 install --no-cache-dir lerobot || true

# Set up workspace directory
WORKDIR /ws

# Environment variables for Isaac Sim and ROS communication
ENV ROS_DOMAIN_ID=0 \
    OMNI_FETCH_ASSETS=1 \
    ACCEPT_EULA=Y

# Default command
CMD ["bash"]


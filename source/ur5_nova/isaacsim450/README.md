# Isaac Sim 4.5.0 Docker Integration

This folder contains the necessary Docker configuration to integrate NVIDIA Isaac Sim 4.5.0 with the UR5 Push-T project.

## Features

- ✅ Isaac Sim 4.5.0 base image
- ✅ Python data collection packages (numpy, opencv, pyarrow, pandas)
- ✅ PyTorch CPU support
- ✅ LeRobot integration
- ✅ ROS 2 communication support

## Usage

### 1. Building Docker Image

To build the Isaac Sim image:

```bash
cd isaacsim450
./build.sh
```

or manually:

```bash
docker build -t isaacsim450:latest .
```

### 2. Running with Docker Compose

From the project root directory:

```bash
# Start full simulation environment with Isaac Sim
docker-compose -f docker-compose-sim.yml up

# Start only Isaac Sim service
docker-compose -f docker-compose-sim.yml up isaac-sim

# Run in background
docker-compose -f docker-compose-sim.yml up -d
```

### 3. Manual Container Startup

Manual startup with `start_simulation.sh` script:

```bash
./start_simulation.sh
```

## System Requirements

- NVIDIA GPU (CUDA support)
- NVIDIA Container Toolkit
- Docker or Docker Compose
- At least 8GB GPU memory (recommended: 12GB+)

## Environment Variables

The Docker container uses the following environment variables:

- `ACCEPT_EULA=Y` - Accept NVIDIA EULA
- `ROS_DOMAIN_ID=0` - ROS 2 domain ID
- `OMNI_FETCH_ASSETS=1` - Auto download Isaac Sim assets
- `DISPLAY` - X11 display support
- `HEADLESS=0` - Run in GUI mode (1 = headless)

## Cache and Volume Configuration

Cache directories are mounted for Isaac Sim performance:

- `~/docker/isaac-sim/cache/kit` - Kit cache
- `~/docker/isaac-sim/cache/ov` - Omniverse cache
- `~/docker/isaac-sim/cache/pip` - Python package cache
- `~/docker/isaac-sim/cache/glcache` - OpenGL cache
- `~/docker/isaac-sim/cache/computecache` - Compute cache
- `~/docker/isaac-sim/logs` - Isaac Sim logs

## Troubleshooting

### GPU access error
```bash
# Check if NVIDIA runtime is installed
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### X11 display error
```bash
# Enable X11 access on host
xhost +local:docker
```

### If cache directories don't exist
```bash
# Create cache directories
mkdir -p ~/docker/isaac-sim/cache/{kit,ov,pip,glcache,computecache}
mkdir -p ~/docker/isaac-sim/logs
```

## More Information

- [NVIDIA Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/latest/)
- Main project README: `../ISAAC_SIM_SETUP.md`
- ROS 2 setup: `../SETUP.md`


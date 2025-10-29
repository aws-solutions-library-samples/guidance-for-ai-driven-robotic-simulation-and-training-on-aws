# UR5 Push-T Setup & Execution Guide

## 📋 Prerequisites

**The ur5_nova folder must be under the home directory.**

**Isaac Sim Requirements:**
	- You must use the Python or Env provided by Isaac Sim
	- You can install the isaacsim python env from here: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html
	- If you prefer to use the Python in the IsaacSim directory instead of the env, 
	  you can run it with python.sh -p
	- NVIDIA Isaac Sim 5.1.0 installed on HOST machine
	- Required Python packages: numpy, opencv-python, torch, lerobot

**Network Setup (For Running on Different Computers):**
	- For detailed information, see: NETWORK_SETUP.md
	- Same computer: host network mode (current setup)
	- Different computers: DDS configuration and firewall settings required

---

## 🚀 Quick Start (Step-by-Step)

### **Step 1: Build Docker Image**

Build the ROS2 Docker container:

**Step 1: Build Docker Image**
```bash
./rebuild-docker.sh
```

---

### **Step 2: Build ROS2 Workspace**

Build the ROS2 packages inside the Docker container:

```bash
./run.sh
# → Select option 8: 🔨 Colcon Build
```


---

### **Step 3: Start Isaac Sim (HOST Machine)**

**Terminal 1 - Isaac Sim:**

Start Isaac Sim with special parameters on the HOST machine:

You must use the Python or Env provided by Isaac Sim

```bash
cd ~/ur5_nova
RESET_RECREATE_ROS_NODES=1 RESET_RELOAD_USE_OPEN=1 MINIMAL_TEST=1 ENABLE_ROS=0 ROS_BRIDGE_ENABLE_STRATEGY=pre_stage /home/ubuntu/isaacsim/python.sh ./Scripts/Start_Sim.py
```

⏳ **Wait for Isaac Sim to fully load before proceeding to the next step.**

---

### **Step 4: Start ROS2 MoveIt Servo (Docker Container)**

**Terminal 2 - MoveIt Servo:**

Once Isaac Sim is running, start the MoveIt Servo controller:

```bash
./run.sh
# → Select option 2: 🤖 Start ROS2 MoveIt Servo
```

### **Step 5: Start Data Collection (Docker Container)**

**Terminal 3 - Data Collection:**

After both Isaac Sim and MoveIt Servo are running, start the data collection:

```bash
./run.sh
# → Select option 3: 📊 Run Data Collection
```

---


---

## 🔧 Additional Commands

### Access Docker Container

```bash
./run.sh
# → Select option 1: 🐳 Enter Docker Container
```

### Check Container Status

```bash
./run.sh
# → Select option 5: 🔍 Check Container Status
```

### Stop Running Containers

```bash
./run.sh
# → Select option 6: 🛑 Stop Running Containers
```

### NumPy Version Fix

If you encounter NumPy 2.x compatibility issues:

```bash
./run.sh
# → Select option 4: 🔧 NumPy Fix
```

---

## 📁 Dataset Structure

After successful episodes, data will be saved in:

```
~/ur5_push_T-main/Scripts/dataset/
├── data/
│   └── chunk-000/
│       ├── episode_00001.parquet
│       ├── episode_00002.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       └── observation.images.state/
│           ├── episode_00001.mp4
│           ├── episode_00002.mp4
│           └── ...
└── meta/
    ├── episodes.jsonl
    └── episodes_stats.jsonl
```

---

## 🐛 Troubleshooting

### Isaac Sim GUI not opening
```bash
xhost +local:docker
```

### ROS2 topics not visible
```bash
# Check ROS domain
echo $ROS_DOMAIN_ID  # Should be 0

# List topics
ros2 topic list

# Restart DDS daemon
ros2 daemon stop && ros2 daemon start
```

### Container not starting
```bash
# Rebuild image
./rebuild-docker.sh

# Check logs
docker logs ur5_push_t
```

### Build errors
```bash
# Clean build
rm -rf build/ install/ log/

# Rebuild
./run.sh
# → Option 8: Colcon Build
```

---

## 📚 Additional Documentation

- **README.md** - Project overview and general usage
- **NETWORK_SETUP.md** - Network configuration for multi-machine setup
- **ARCHITECTURE.md** - Technical architecture details
- **QUICKSTART.md** - Quick start guide

---

## 🎯 Summary Workflow

```
1. Build Docker    →  docker build -t ur5_push_t:humble-fixed .
2. Build ROS2      →  ./run.sh → Option 8
3. Start Isaac Sim →  RESET_RECREATE_ROS_NODES=1 ... python ./Scripts/Start_Sim.py
4. Start MoveIt    →  ./run.sh → Option 2
5. Start Data Col. →  ./run.sh → Option 3
```

---

**Note:** Always start Isaac Sim on the HOST machine first, then start the Docker containers. The Docker containers communicate with Isaac Sim via ROS2 topics over the host network.

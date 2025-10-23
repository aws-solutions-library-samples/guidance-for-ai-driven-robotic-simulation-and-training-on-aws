#!/bin/bash
# UR5 Push-T Docker Management Script
set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Docker image and container settings
IMAGE_NAME="ur5_push_t:humble-fixed"
CONTAINER_NAME="ur5_ros2_dev"
WORKSPACE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     UR5 Push-T Docker Manager         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running! Please start Docker.${NC}"
    exit 1
fi

# Check if image exists and is up-to-date
IMAGE_EXISTS=false
IMAGE_NEEDS_REBUILD=false

if docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
    IMAGE_EXISTS=true
    
    # Check NumPy version in existing image
    echo -e "${BLUE}🔍 Checking image...${NC}"
    NUMPY_VERSION=$(docker run --rm "${IMAGE_NAME}" python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "error")
    
    if [[ "$NUMPY_VERSION" == "error" ]]; then
        echo -e "${YELLOW}⚠️  NumPy check in image failed.${NC}"
        IMAGE_NEEDS_REBUILD=true
    elif [[ "$NUMPY_VERSION" == 2.* ]]; then
        echo -e "${RED}❌ Image has NumPy 2.x (${NUMPY_VERSION})! Rebuild required.${NC}"
        IMAGE_NEEDS_REBUILD=true
    else
        echo -e "${GREEN}✓ Image up-to-date (NumPy ${NUMPY_VERSION})${NC}"
    fi
    echo ""
fi

if [ "$IMAGE_EXISTS" = false ] || [ "$IMAGE_NEEDS_REBUILD" = true ]; then
    if [ "$IMAGE_EXISTS" = false ]; then
        echo -e "${YELLOW}⚠️  Docker image '${IMAGE_NAME}' not found!${NC}"
    else
        echo -e "${YELLOW}⚠️  Docker image needs rebuild!${NC}"
    fi
    echo ""
    read -p "Do you want to build the image now? (y/n): " build_choice
    if [[ "$build_choice" =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}🏗️  Building ${IMAGE_NAME}...${NC}"
        docker build -t "${IMAGE_NAME}" "${WORKSPACE_DIR}"
        
        # Verify NumPy after build
        echo ""
        echo -e "${BLUE}🔍 Checking NumPy version...${NC}"
        FINAL_NUMPY=$(docker run --rm "${IMAGE_NAME}" python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "error")
        
        if [[ "$FINAL_NUMPY" == 1.* ]]; then
            echo -e "${GREEN}✅ Build successful! NumPy ${FINAL_NUMPY}${NC}"
        else
            echo -e "${RED}❌ Build finished but NumPy is ${FINAL_NUMPY}!${NC}"
            echo -e "${YELLOW}Manual fix may be needed: ./run.sh → Option 4${NC}"
        fi
        echo ""
    else
        echo -e "${RED}❌ Cannot proceed without image.${NC}"
        exit 1
    fi
fi

# Menu
echo -e "${PURPLE}═══════════════════════════════════════${NC}"
echo -e "${GREEN}Options:${NC}"
echo -e "${PURPLE}═══════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}1)${NC} 🐳 Enter Docker Container (bash terminal)"
echo -e "${BLUE}2)${NC} 🤖 Start ROS2 MoveIt Servo (arm_joy_control)"
echo -e "${BLUE}3)${NC} 📊 Start Data Collection (ACT Model)"
echo -e "${BLUE}4)${NC} 🔧 NumPy Fix (pip install numpy<2)"
echo -e "${BLUE}5)${NC} 🔍 Check Container Status"
echo -e "${BLUE}6)${NC} 🛑 Stop Running Containers"
echo -e "${BLUE}7)${NC} 🚀 Quick Start (All-in-One)"
echo -e "${BLUE}8)${NC} 🔨 Colcon Build (ROS2 Workspace Build)"
echo -e "${BLUE}9)${NC} 🧠 RL Finetune (Scripts/RL_Finetune.py)"
echo -e "${BLUE}0)${NC} ❌ Exit"
echo ""
echo -e "${PURPLE}═══════════════════════════════════════${NC}"
echo ""
read -p "Your choice (0-9): " choice

# Docker run command template
DOCKER_RUN_BASE="docker run --rm -it --net=host \
  -e ROS_DOMAIN_ID=0 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e DISPLAY=\${DISPLAY} \
  -v ${WORKSPACE_DIR}:/ws \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -w /ws \
  ${IMAGE_NAME}"

case $choice in
    1)
        echo -e "${GREEN}════════════════════════════════════════${NC}"
        echo -e "${GREEN}1) Docker Container Bash Terminal${NC}"
        echo -e "${GREEN}════════════════════════════════════════${NC}"
        echo ""
        echo -e "${YELLOW}Opening bash terminal...${NC}"
        echo -e "${BLUE}To exit: ${NC}${PURPLE}exit${NC} or ${PURPLE}CTRL+D${NC}"
        echo ""
        
        eval "${DOCKER_RUN_BASE} bash"
        ;;
        
    2)
        echo -e "${GREEN}════════════════════════════════════════${NC}"
        echo -e "${GREEN}2) Starting ROS2 MoveIt Servo${NC}"
        echo -e "${GREEN}════════════════════════════════════════${NC}"
        echo ""
        echo -e "${YELLOW}⚠️  Make sure Isaac Sim is running!${NC}"
        echo ""
        read -p "Press ENTER to continue (CTRL+C to cancel)..."
        echo ""
        echo -e "${BLUE}Enable RViz? (X11 required)${NC}"
        read -p "(y/n): " rviz_choice
        
        if [[ "$rviz_choice" =~ ^[Yy]$ ]]; then
            LAUNCH_CMD="source install/setup.bash && ros2 launch ur5_moveit_config arm_joy_control.launch.py"
        else
            LAUNCH_CMD="source install/setup.bash && ros2 launch ur5_moveit_config arm_joy_control.launch.py use_rviz:=false"
        fi
        
        echo ""
        echo -e "${GREEN}🚀 Launching...${NC}"
        echo -e "${YELLOW}To stop: ${NC}${PURPLE}CTRL+C${NC}"
        echo ""
        
        eval "${DOCKER_RUN_BASE} bash -c '${LAUNCH_CMD}'"
        ;;
        
    3)
        echo -e "${GREEN}════════════════════════════════════════${NC}"
        echo -e "${GREEN}3) Data Collection (ACT Model)${NC}"
        echo -e "${GREEN}════════════════════════════════════════${NC}"
        echo ""
        echo -e "${YELLOW}⚠️  Please ensure:${NC}"
        echo -e "   1. Isaac Sim is running"
        echo -e "   2. ROS2 MoveIt Servo is active"
        echo -e "   3. Pretrained model exists (${WORKSPACE_DIR}/Scripts/pretrained_model/)"
        echo ""
        read -p "Press ENTER to continue (CTRL+C to cancel)..."
        echo ""
        
        # Check if model exists
        if [ ! -d "${WORKSPACE_DIR}/Scripts/pretrained_model" ]; then
            echo -e "${RED}❌ Model not found: ${WORKSPACE_DIR}/Scripts/pretrained_model/${NC}"
            echo ""
            echo -e "${YELLOW}You need to download or train the model!${NC}"
            exit 1
        fi
        
        echo -e "${BLUE}Enable data collection?${NC}"
        echo -e "  ${GREEN}y)${NC} Yes - Save episodes"
        echo -e "  ${YELLOW}n)${NC} No - Run policy only (test mode)"
        read -p "Choice (y/n): " data_collect_choice
        
        if [[ "$data_collect_choice" =~ ^[Yy]$ ]]; then
            ENABLE_COLLECTION="1"
            echo -e "${GREEN}✓ Data collection enabled${NC}"
        else
            ENABLE_COLLECTION="0"
            echo -e "${YELLOW}✓ Test mode (no data will be saved)${NC}"
        fi
        
        echo ""
        echo -e "${GREEN}🚀 Starting data collection...${NC}"
        echo -e "${YELLOW}To stop: ${NC}${PURPLE}CTRL+C${NC}"
        echo ""
        
        eval "docker run --rm -it --net=host \
          -e ROS_DOMAIN_ID=0 \
          -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
          -e ENABLE_DATA_COLLECTION=${ENABLE_COLLECTION} \
          -v ${WORKSPACE_DIR}:/ws \
          -w /ws \
          ${IMAGE_NAME} python3 /ws/Scripts/data_collection_Model.py"
        ;;
        
    4)
        echo -e "${GREEN}════════════════════════════════════════${NC}"
        echo -e "${GREEN}4) NumPy Version Fix${NC}"
        echo -e "${GREEN}════════════════════════════════════════${NC}"
        echo ""
        echo -e "${BLUE}NumPy 2.x → 1.x downgrade will be applied${NC}"
        echo ""
        
        eval "${DOCKER_RUN_BASE} bash -c \"pip3 install 'numpy<2' && python3 -c 'import numpy; print(\\\"NumPy version:\\\", numpy.__version__)'\""
        
        echo ""
        echo -e "${GREEN}✅ NumPy fix completed!${NC}"
        echo -e "${YELLOW}⚠️  This is temporary; after container restart you may need to run again.${NC}"
        echo -e "${BLUE}For a persistent fix: ${NC}${PURPLE}./rebuild-docker.sh${NC}"
        ;;
        
    5)
        echo -e "${GREEN}════════════════════════════════════════${NC}"
        echo -e "${GREEN}5) Container Status${NC}"
        echo -e "${GREEN}════════════════════════════════════════${NC}"
        echo ""
        
        echo -e "${BLUE}🐳 Running Containers:${NC}"
        docker ps --filter ancestor="${IMAGE_NAME}" --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}"
        
        echo ""
        echo -e "${BLUE}📦 Image Info:${NC}"
        docker images "${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
        
        echo ""
        echo -e "${BLUE}💾 Disk Usage:${NC}"
        docker system df
        ;;
        
    6)
        echo -e "${GREEN}════════════════════════════════════════${NC}"
        echo -e "${GREEN}6) Stop Containers${NC}"
        echo -e "${GREEN}════════════════════════════════════════${NC}"
        echo ""
        
        running_containers=$(docker ps -q --filter ancestor="${IMAGE_NAME}")
        
        if [ -z "$running_containers" ]; then
            echo -e "${YELLOW}⚠️  No running containers.${NC}"
        else
            echo -e "${RED}The following containers will be stopped:${NC}"
            docker ps --filter ancestor="${IMAGE_NAME}"
            echo ""
            read -p "Confirm? (y/n): " confirm
            
            if [[ "$confirm" =~ ^[Yy]$ ]]; then
                echo ""
                docker stop $running_containers
                echo -e "${GREEN}✅ Containers stopped.${NC}"
            else
                echo -e "${YELLOW}Cancelled.${NC}"
            fi
        fi
        ;;
        
    7)
        echo -e "${GREEN}════════════════════════════════════════${NC}"
        echo -e "${GREEN}7) Quick Start (All-in-One)${NC}"
        echo -e "${GREEN}════════════════════════════════════════${NC}"
        echo ""
        echo -e "${BLUE}This mode uses 3 terminals:${NC}"
        echo -e "  1. ${YELLOW}Terminal 1:${NC} Isaac Sim (Host)"
        echo -e "  2. ${YELLOW}Terminal 2:${NC} ROS2 MoveIt Servo"
        echo -e "  3. ${YELLOW}Terminal 3:${NC} Data Collection"
        echo ""
        echo -e "${RED}⚠️  You can use 'tmux' or 'screen' for convenience${NC}"
        echo -e "${YELLOW}For now open 3 terminals and run:${NC}"
        echo ""
        echo -e "  ${PURPLE}Terminal 1:${NC} Start Isaac Sim (on host)"
        echo -e "    ${GREEN}RESET_RECREATE_ROS_NODES=1 RESET_RELOAD_USE_OPEN=1 python3 ./Scripts/Start_Sim.py${NC}"
        echo ""
        echo -e "  ${PURPLE}Terminal 2:${NC} ROS2 MoveIt"
        echo -e "    ${GREEN}./run.sh${NC} → Option 2"
        echo ""
        echo -e "  ${PURPLE}Terminal 3:${NC} Data Collection"
        echo -e "    ${GREEN}./run.sh${NC} → Option 3"
        echo ""
        ;;
        
    8)
        echo -e "${GREEN}════════════════════════════════════════${NC}"
        echo -e "${GREEN}8) Colcon Build (ROS2 Workspace)${NC}"
        echo -e "${GREEN}════════════════════════════════════════${NC}"
        echo ""
        echo -e "${BLUE}Building ROS2 workspace...${NC}"
        echo ""
        
        # Check if colcon-build.sh exists
        if [ ! -f "${WORKSPACE_DIR}/colcon-build.sh" ]; then
            echo -e "${RED}❌ colcon-build.sh not found!${NC}"
            exit 1
        fi
        
        # Run colcon build script in container
        echo -e "${YELLOW}Running colcon build inside container...${NC}"
        ${DOCKER_RUN_BASE} bash -lc "cd /ws && ./colcon-build.sh"
        
        BUILD_EXIT=$?
        echo ""
        if [ $BUILD_EXIT -eq 0 ]; then
            echo -e "${GREEN}✅ Build successful!${NC}"
            echo -e "${YELLOW}To use the environment:${NC}"
            echo -e "  ${BLUE}./run.sh${NC} → Option 1 (Enter container)"
            echo -e "  ${BLUE}source install/setup.bash${NC}"
        else
            echo -e "${RED}❌ Build failed!${NC}"
            echo -e "${YELLOW}Check logs:${NC}"
            echo -e "  ${BLUE}cat log/latest_build/events.log${NC}"
        fi
        ;;
        
    9)
        echo -e "${GREEN}════════════════════════════════════════${NC}"
        echo -e "${GREEN}9) RL Finetune (LeRobot ACT)${NC}"
        echo -e "${GREEN}════════════════════════════════════════${NC}"
        echo ""
        echo -e "${YELLOW}⚠️  Ensure:${NC}"
        echo -e "   1. Isaac Sim is running on host"
        echo -e "   2. MoveIt Servo is active (Option 2)"
        echo -e "   3. Pretrained weights exist at ${WORKSPACE_DIR}/Scripts/pretrained_model/"
        echo ""
        read -p "Fresh start (ignore latest checkpoint)? (y/n): " fresh
        if [[ "$fresh" =~ ^[Yy]$ ]]; then
            RL_FRESH=1
        else
            RL_FRESH=0
        fi

        echo -e "${BLUE}Reset mode on stop:${NC}"
        echo -e "  ${GREEN}1)${NC} pretrained (default)"
        echo -e "  ${BLUE}2)${NC} latest"
        echo -e "  ${YELLOW}3)${NC} scratch"
        read -p "Choice (1-3): " reset_mode
        case $reset_mode in
            2) RL_RESET_MODE=latest ;;
            3) RL_RESET_MODE=scratch ;;
            *) RL_RESET_MODE=pretrained ;;
        esac

        echo ""
        echo -e "${GREEN}🚀 Starting RL Finetune...${NC}"
        echo -e "${YELLOW}To stop: ${NC}${PURPLE}CTRL+C${NC}"
        echo ""

        # Enable GPU if NVIDIA runtime is available
        GPU_FLAG=""
        if docker info 2>/dev/null | grep -qi "Runtimes:.*nvidia\|NVIDIA"; then
            GPU_FLAG="--gpus all"
        fi

        eval "docker run --rm -it --net=host $GPU_FLAG \
          -e ROS_DOMAIN_ID=0 \
          -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
          -e RL_FINETUNE_FRESH_START=${RL_FRESH} \
          -e RL_RESET_MODE=${RL_RESET_MODE} \
          -v ${WORKSPACE_DIR}:/ws \
          -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
          -w /ws \
          ${IMAGE_NAME} bash -lc 'source /opt/ros/${ROS_DISTRO:-humble}/setup.bash && source /ws/install/setup.bash 2>/dev/null || true && python3 /ws/Scripts/RL_Finetune.py'"
        ;;
        
    0)
        echo -e "${BLUE}👋 Exiting...${NC}"
        exit 0
        ;;
        
    *)
        echo -e "${RED}❌ Invalid choice!${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✅ Done!${NC}"


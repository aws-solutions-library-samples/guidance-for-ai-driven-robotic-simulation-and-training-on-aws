#!/bin/bash
# Colcon Build Script for UR5 Push-T Project
set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     UR5 Push-T Colcon Build Script    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if we're in a container or host
IN_CONTAINER=false
if [ -f "/.dockerenv" ] || grep -q 'docker\|lxc' /proc/1/cgroup 2>/dev/null; then
    IN_CONTAINER=true
fi

# Clean old build files (optional)
echo -e "${YELLOW}Do you want to clean old build files? (y/n)${NC}"
read -p "Clean? " clean_choice

if [[ "$clean_choice" =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}🧹 Cleaning build files...${NC}"
    
    if [ "$IN_CONTAINER" = true ]; then
        # Inside container, root permission not required
        rm -rf build/ install/ log/
        echo -e "${GREEN}✓ Cleanup completed${NC}"
    else
        # On host, sudo may be required
        if [ -w build ] 2>/dev/null; then
            rm -rf build/ install/ log/
            echo -e "${GREEN}✓ Cleanup completed${NC}"
        else
            echo -e "${YELLOW}⚠️  sudo permission required...${NC}"
            sudo rm -rf build/ install/ log/
            echo -e "${GREEN}✓ Cleanup completed (with sudo)${NC}"
        fi
    fi
    echo ""
fi

# Check for required ROS2 environment
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${RED}❌ ROS2 environment is not active!${NC}"
    echo -e "${YELLOW}Please source ROS2 first:${NC}"
    echo -e "  source /opt/ros/humble/setup.bash"
    exit 1
fi

echo -e "${GREEN}✓ ROS2 Distro: ${ROS_DISTRO}${NC}"
echo ""

# Build options
BUILD_TYPE="Release"
SYMLINK_INSTALL="--symlink-install"
PARALLEL_WORKERS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --no-symlink)
            SYMLINK_INSTALL=""
            shift
            ;;
        --parallel)
            PARALLEL_WORKERS="--parallel-workers $2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug          Build in debug mode"
            echo "  --no-symlink     Don't use symlink install"
            echo "  --parallel N     Use N parallel workers"
            echo "  --help           Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown parameter: $1${NC}"
            exit 1
            ;;
    esac
done

# Display build configuration
echo -e "${PURPLE}═══════════════════════════════════════${NC}"
echo -e "${GREEN}Build Configuration:${NC}"
echo -e "${PURPLE}═══════════════════════════════════════${NC}"
echo -e "  Build Type: ${BLUE}${BUILD_TYPE}${NC}"
echo -e "  Symlink Install: ${BLUE}${SYMLINK_INSTALL:-Disabled}${NC}"
echo -e "  Parallel Workers: ${BLUE}${PARALLEL_WORKERS:-Auto}${NC}"
echo -e "${PURPLE}═══════════════════════════════════════${NC}"
echo ""

# Start build
echo -e "${BLUE}🔨 Starting colcon build...${NC}"
echo ""

# Run colcon build
colcon build \
    --cmake-args \
      -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
      -DBUILD_TESTING=OFF \
    ${SYMLINK_INSTALL} \
    ${PARALLEL_WORKERS}

BUILD_STATUS=$?

echo ""
if [ $BUILD_STATUS -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║         ✅ BUILD SUCCESSFUL!           ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Build completed. Now source the environment:${NC}"
    echo -e "  ${BLUE}source install/setup.bash${NC}"
    echo ""
else
    echo -e "${RED}╔════════════════════════════════════════╗${NC}"
    echo -e "${RED}║         ❌ BUILD FAILED!               ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}See logs for details:${NC}"
    echo -e "  ${BLUE}cat log/latest_build/events.log${NC}"
    exit 1
fi

# Optional: Source automatically
if [ "$IN_CONTAINER" = true ]; then
    echo -e "${BLUE}You are inside the container.${NC}"
    echo -e "${YELLOW}Auto-sourcing environment...${NC}"
    source install/setup.bash
    echo -e "${GREEN}✓ Environment sourced${NC}"
fi

echo ""
echo -e "${PURPLE}════════════════════════════════════════${NC}"
echo -e "${GREEN}Packages:${NC}"
echo -e "${PURPLE}════════════════════════════════════════${NC}"

# List built packages
if [ -d "install" ]; then
    for pkg in install/*/; do
        pkg_name=$(basename "$pkg")
        if [ "$pkg_name" != "local_setup.bash" ] && [ "$pkg_name" != "setup.bash" ]; then
            echo -e "  ${BLUE}✓${NC} $pkg_name"
        fi
    done
fi

echo -e "${PURPLE}════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}Build process completed!${NC}"


#!/bin/bash
# Rebuild Docker image with NumPy fix
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Docker Image Rebuild (NumPy Fix) ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════╝${NC}"
echo ""

WORKSPACE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${WORKSPACE_DIR}"

# Check if old image exists
if docker image inspect ur5_push_t:humble > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Found old image 'ur5_push_t:humble'${NC}"
    read -p "Do you want to remove it? (y/n): " remove_old
    if [[ "$remove_old" =~ ^[Yy]$ ]]; then
        docker rmi ur5_push_t:humble
        echo -e "${GREEN}✓ Old image removed${NC}"
    fi
    echo ""
fi

echo -e "${YELLOW}📦 Building ur5_push_t:humble-fixed...${NC}"
echo -e "${YELLOW}   This will fix the NumPy 2.x compatibility issue${NC}"
echo ""

# Build the image with no cache to ensure fresh NumPy install
docker build --no-cache -t ur5_push_t:humble-fixed .

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Build successful!${NC}"
    echo ""
    
    # Verify NumPy version
    echo -e "${BLUE}🔍 Checking NumPy version...${NC}"
    NUMPY_VERSION=$(docker run --rm ur5_push_t:humble-fixed python3 -c "import numpy; print(numpy.__version__)")
    
    if [[ "$NUMPY_VERSION" == 1.* ]]; then
        echo -e "${GREEN}✅ NumPy ${NUMPY_VERSION} (OK - 1.x series)${NC}"
    else
        echo -e "${YELLOW}⚠️  NumPy ${NUMPY_VERSION} (Expected: 1.x)${NC}"
    fi
    
    # Verify RL_Finetune.py critical imports
    echo ""
    echo -e "${BLUE}🔍 Verifying RL_Finetune dependencies...${NC}"
    TORCH_VERSION=$(docker run --rm ur5_push_t:humble-fixed python3 -c "import torch; print(getattr(torch, '__version__', 'missing'))" || echo "missing")
    CV_VERSION=$(docker run --rm ur5_push_t:humble-fixed python3 -c "import cv2; print(getattr(cv2, '__version__', 'missing'))" || echo "missing")
    LEROBOT_OK=$(docker run --rm ur5_push_t:humble-fixed python3 -c "from importlib import util; print('OK' if util.find_spec('lerobot') else 'MISSING')" || echo "MISSING")
    PXR_OK=$(docker run --rm ur5_push_t:humble-fixed python3 -c "from pxr import Usd; print('OK')" || echo "MISSING")
    PXR_IMPORT_OK=$(docker run --rm ur5_push_t:humble-fixed python3 -c "from importlib import util; import sys; print('OK' if util.find_spec('pxr') else 'MISSING')" || echo "MISSING")
    SCIPY_VERSION=$(docker run --rm ur5_push_t:humble-fixed python3 -c "import scipy; print(getattr(scipy, '__version__', 'missing'))" || echo "missing")
    REQ_OK=$(docker run --rm ur5_push_t:humble-fixed python3 -c "import requests, imageio; print('OK')" || echo "MISSING")

    echo -e "Torch: ${TORCH_VERSION}"
    echo -e "OpenCV: ${CV_VERSION}"
    echo -e "LeRobot: ${LEROBOT_OK}"
    echo -e "PXR (USD): ${PXR_OK}"
    echo -e "PXR Import: ${PXR_IMPORT_OK}"
    echo -e "SciPy: ${SCIPY_VERSION}"
    echo -e "Requests/ImageIO: ${REQ_OK}"
    
    echo ""
    echo -e "${GREEN}🎉 You can now run:${NC}"
    echo ""
    echo -e "   ${BLUE}docker run --rm -it --net=host \\${NC}"
    echo -e "   ${BLUE}  -v ~/ur5_nova:/ws \\${NC}"
    echo -e "   ${BLUE}  ur5_push_t:humble-fixed \\${NC}"
    echo -e "   ${BLUE}  python3 /ws/Scripts/data_collection_Model.py${NC}"
    echo ""
    echo -e "${GREEN}or:${NC}"
    echo ""
    echo -e "   ${BLUE}./quick-start.sh${NC}"
    echo ""
    echo -e "${GREEN}RL Finetune from menu:${NC}"
    echo -e "   ${BLUE}./run.sh${NC} → Option 9"
    echo ""
else
    echo ""
    echo -e "${RED}❌ Build failed!${NC}"
    exit 1
fi



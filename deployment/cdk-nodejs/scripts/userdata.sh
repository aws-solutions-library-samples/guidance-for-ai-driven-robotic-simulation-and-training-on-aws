#!/bin/bash

# Set up logging
exec > >(tee /var/log/user-data.log) 2>&1
echo "Starting installation at $(date)"

# Create installation phases flag file
FLAG_FILE="/var/lib/cloud/instance/installation_phase"

if [ ! -f "$FLAG_FILE" ]; then
    echo "Phase 1" > "$FLAG_FILE"
    
    echo "Starting Phase 1: Basic Installation"
    
    # Initial system setup
    apt update -y
    
    # Install required tools for password retrieval
    DEBIAN_FRONTEND=noninteractive apt install -y jq
    
    # Check if AWS CLI v2 is already installed, if not install it
    if ! command -v aws &> /dev/null; then
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
        unzip awscliv2.zip
        ./aws/install
        rm -rf aws awscliv2.zip
    fi
    
    DEBIAN_FRONTEND=noninteractive apt install ubuntu-desktop -y

    DEBIAN_FRONTEND=noninteractive apt install gdm3 -y
    
    echo "Current display manager:"
    cat /etc/X11/default-display-manager

    dpkg-reconfigure gdm3
    
    DEBIAN_FRONTEND=noninteractive apt upgrade -y
    
    # Create Phase 2 startup script
    cat > /var/lib/cloud/scripts/per-boot/finish-setup.sh << 'EOF'
#!/bin/bash
exec > >(tee -a /var/log/user-data-phase2.log) 2>&1

# Function to set password from Secrets Manager
set_password() {
    echo "Retrieving password from Secrets Manager..."
    PASSWORD=$(aws secretsmanager get-secret-value --secret-id "${secret_name}" --query SecretString --output text | jq -r '.password')
    if [ -n "$PASSWORD" ]; then
        echo "ubuntu:$PASSWORD" | chpasswd
        echo "Password set successfully for ubuntu user"
    else
        echo "Failed to retrieve password from Secrets Manager"
    fi
}

# Function to install ROS 2
install_ros() {
    echo "Installing ROS 2 Jazzy..."
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
    apt update -y
    DEBIAN_FRONTEND=noninteractive apt install ros-jazzy-desktop python3-rosdep python3-colcon-common-extensions -y
    rosdep init
    su - ubuntu -c "rosdep update"
    echo "source /opt/ros/jazzy/setup.bash" >> /home/ubuntu/.bashrc
}

# Function to disable Wayland
disable_wayland() {
    # Check if /etc/gdm3/custom.conf exists
    if [ -f /etc/gdm3/custom.conf ]; then
        # Backup existing configuration
        cp /etc/gdm3/custom.conf /etc/gdm3/custom.conf.backup.$(date +%Y%m%d_%H%M%S)
        
        # Check if [daemon] section exists
        if grep -q "^\[daemon\]" /etc/gdm3/custom.conf; then
            # Check if WaylandEnable exists
            if grep -q "^WaylandEnable" /etc/gdm3/custom.conf; then
                # Replace existing WaylandEnable line
                sed -i 's/^WaylandEnable=.*$/WaylandEnable=false/' /etc/gdm3/custom.conf
            else
                # Add WaylandEnable under [daemon] section
                sed -i '/^\[daemon\]/a WaylandEnable=false' /etc/gdm3/custom.conf
            fi
        else
            # Add [daemon] section and WaylandEnable
            echo -e "\n[daemon]\nWaylandEnable=false" >> /etc/gdm3/custom.conf
        fi
    else
        # Create new custom.conf if it doesn't exist
        mkdir -p /etc/gdm3
        cat > /etc/gdm3/custom.conf << EOL
[daemon]
WaylandEnable=false
DefaultSession=ubuntu-xorg.desktop
EOL
    fi
}

if [ "$(cat /var/lib/cloud/instance/installation_phase)" == "Phase 1" ]; then
    echo "Starting Phase 2 at $(date)"
    echo "Phase 2" > /var/lib/cloud/instance/installation_phase

    # Disable Wayland
    disable_wayland
    echo "Wayland disabled successfully"

    # Restart the GDM service.
    systemctl restart gdm3
    
    # Configure X Server
    systemctl get-default
    systemctl set-default graphical.target
    systemctl isolate graphical.target
    
    echo "Verify that the X server is running"
    ps aux | grep X | grep -v grep

    # Install mesa-utils and unzip
    DEBIAN_FRONTEND=noninteractive apt install mesa-utils unzip curl gnupg lsb-release -y
    
    # Install ROS 2
    install_ros
    
    # Configure NVIDIA
    nvidia-xconfig --preserve-busid --enable-all-gpus
    
    # Restart X server
    systemctl isolate multi-user.target
    systemctl isolate graphical.target
    
    # Install DCV server
    cd /tmp
    wget https://d1uj6qtbmh3dt5.cloudfront.net/NICE-GPG-KEY
    gpg --import NICE-GPG-KEY
    wget https://d1uj6qtbmh3dt5.cloudfront.net/2024.0/Servers/nice-dcv-2024.0-19030-ubuntu2404-x86_64.tgz
    tar -xvzf nice-dcv-2024.0-19030-ubuntu2404-x86_64.tgz
    cd nice-dcv-2024.0-19030-ubuntu2404-x86_64
    
    DEBIAN_FRONTEND=noninteractive apt install ./nice-dcv-server_2024.0.19030-1_amd64.ubuntu2404.deb -y
    DEBIAN_FRONTEND=noninteractive apt install ./nice-dcv-web-viewer_2024.0.19030-1_amd64.ubuntu2404.deb -y
    usermod -aG video dcv
    # DEBIAN_FRONTEND=noninteractive apt install ./nice-xdcv_2024.0.654-1_amd64.ubuntu2404.deb -y
    # DEBIAN_FRONTEND=noninteractive apt install ./nice-dcv-simple-external-authenticator_2024.0.266-1_amd64.ubuntu2404.deb -y
    
    # # restart dcv server
    systemctl enable dcvserver
    systemctl restart dcvserver
    dcv create-session  --type console --name robotics-session --owner ubuntu robotics-sessionid
    
    # Restart X server
    systemctl isolate multi-user.target
    systemctl isolate graphical.target
    
    # Install Isaac Sim
    mkdir -p /home/ubuntu/isaacsim
    cd /home/ubuntu/isaacsim
    echo "Downloading Isaac Sim..."
    wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
    
    echo "Unzipping Isaac Sim..."
    unzip -q isaac-sim-standalone-5.1.0-linux-x86_64.zip
    
    # Clean up zip file to save space
    rm isaac-sim-standalone-5.1.0-linux-x86_64.zip
    
    # Set proper ownership
    chown -R ubuntu:ubuntu /home/ubuntu/isaacsim
    
    # Run post-install scripts as ubuntu user
    su - ubuntu -c "cd /home/ubuntu/isaacsim && ./post_install.sh"
    
    # Run warm-up scripts as ubuntu user
    su - ubuntu -c "cd /home/ubuntu/isaacsim && ./warmup.sh"

    # Run isaac-sim selector in background with nohup
    # su - ubuntu -c "cd /home/ubuntu/isaacsim && nohup ./isaac-sim.selector.sh > /home/ubuntu/isaacsim/selector.log 2>&1 &"

    # Create a status file to indicate completion
    touch /home/ubuntu/isaacsim/installation_complete
    
    # Set password from Secrets Manager
    set_password

    # copy/sync the Source files from S3 bucket
    mkdir -p /home/ubuntu/ur5_nova
    aws s3 sync s3://${bucket_name}/source/ur5_nova /home/ubuntu/ur5_nova
    chown -R ubuntu:ubuntu /home/ubuntu/ur5_nova
    
    # Remove startup script
    rm /var/lib/cloud/scripts/per-boot/finish-setup.sh
    
    echo "Phase 2 completed at $(date)"
fi
EOF

    # Make the startup script executable
    chmod +x /var/lib/cloud/scripts/per-boot/finish-setup.sh
    
    echo "Phase 1 completed. Rebooting for Phase 2..."
    sleep 5
    reboot
else
    echo "Installation already completed."
fi
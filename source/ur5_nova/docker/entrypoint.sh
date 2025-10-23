#!/usr/bin/env bash
set -eo pipefail

# Source ROS 2
if [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
  . "/opt/ros/${ROS_DISTRO}/setup.bash"
fi

# Source workspace if built
if [ -f "/ws/install/setup.bash" ]; then
  . "/ws/install/setup.bash"
fi

exec "$@"




#!/bin/bash
set -e
WS_DIR="$(cd "$(dirname "$0")" && pwd)"
source /opt/ros/humble/setup.bash
source "$WS_DIR/install/setup.bash"
ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765 "$@"

#!/bin/bash
set -e
WS_DIR="$(cd "$(dirname "$0")" && pwd)"
source /opt/ros/humble/setup.bash
source "$WS_DIR/install/setup.bash"
ros2 launch rosbridge_server rosbridge_websocket_launch.xml port:=9090 "$@"

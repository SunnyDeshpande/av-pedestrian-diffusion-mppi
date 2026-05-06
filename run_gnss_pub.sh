#!/bin/bash
set -e
WS_DIR="$(cd "$(dirname "$0")" && pwd)"
source /opt/ros/humble/setup.bash
source "$WS_DIR/install/setup.bash"

LAT="${LAT:-40.0928243}"
LON="${LON:-(-88.2357659)}"
ALT="${ALT:-0.0}"
RATE_HZ="${RATE_HZ:-10.0}"
FRAME_ID="${FRAME_ID:-gnss}"
TOPIC="${TOPIC:-/navsatfix}"

ros2 topic pub -r "$RATE_HZ" "$TOPIC" sensor_msgs/msg/NavSatFix \
"{header: {frame_id: '$FRAME_ID'}, status: {status: 0, service: 1}, latitude: $LAT, longitude: $LON, altitude: $ALT, position_covariance: [0,0,0,0,0,0,0,0,0], position_covariance_type: 0}"

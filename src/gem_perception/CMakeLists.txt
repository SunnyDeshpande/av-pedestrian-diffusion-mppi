cmake_minimum_required(VERSION 3.0.2)
project(gem_perception)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_python_setup()
catkin_package()

install(PROGRAMS
  scripts/yolo_perception_node.py
  scripts/sam_perception_node.py
  scripts/map_tf_broadcaster.py
  scripts/bev_overlay_node.py
  scripts/download_models.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

cmake_minimum_required(VERSION 3.5)
project(b1_interface)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3 -fPIC)
endif()

# Detect system architecture for Unitree SDK linking
message("-- CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "x86_64.*")
  set(ARCH amd64)
endif()
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "aarch64.*")
  set(ARCH arm64)
endif()

# Find ROS2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Include directories
include_directories(include)

# Setup Unitree SDK linking
link_directories(${PROJECT_SOURCE_DIR}/lib/cpp/${ARCH})
set(UNITREE_LIB unitree_legged_sdk pthread)

# Build B1 Interface Node
add_executable(b1_interface_node
  src/b1_interface_node.cpp
  src/b1_ros2_interface.cpp
  src/myb1.cpp
)

ament_target_dependencies(b1_interface_node
  rclcpp
  std_msgs
  std_srvs
  geometry_msgs
  sensor_msgs
)

target_link_libraries(b1_interface_node
  ${UNITREE_LIB}
)

set_target_properties(b1_interface_node PROPERTIES
  INSTALL_RPATH "${PROJECT_SOURCE_DIR}/lib/cpp/${ARCH}"
  BUILD_WITH_INSTALL_RPATH TRUE
)

# Install executables
install(TARGETS
  b1_interface_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python scripts
install(PROGRAMS
  scripts/sine_to_real_bridge.py
  scripts/test_joint_mapping.py
  scripts/go_to_pose.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()

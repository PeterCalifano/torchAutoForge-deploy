cmake_minimum_required(VERSION 3.16)
project(ptafdeploy_interfaces)

option(PTAFDEPLOY_ENABLE_CUDA "Accepted for workspace-wide build helper compatibility" OFF)

# Resolve ROS interface-generation dependencies.
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate wrapper-safe tensor, status, and inference service contracts.
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FloatTensor.msg"
  "msg/ModelStatus.msg"
  "srv/InferFloatTensors.srv"
  DEPENDENCIES builtin_interfaces)

# Export runtime support required by downstream message consumers.
ament_export_dependencies(rosidl_default_runtime)
ament_package()

cmake_minimum_required(VERSION 3.16)
project(ptafdeploy_spinup)

option(PTAFDEPLOY_ENABLE_CUDA "Accepted for workspace-wide build helper compatibility" OFF)
option(PTAFDEPLOY_ENABLE_ORT "Accepted for workspace-wide build helper compatibility" ON)
option(PTAFDEPLOY_ENABLE_TENSORRT "Accepted for workspace-wide build helper compatibility" OFF)

# Resolve the package and launch-test infrastructure.
find_package(ament_cmake REQUIRED)

# Install launch/configuration assets and the target-owned fixture model.
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/matlab/testSamples/tracedSampleModel.onnx"
  DESTINATION share/${PROJECT_NAME}/models)

# Exercise both standalone and composed launch paths through the installed package.
if(BUILD_TESTING)
  find_package(launch_testing_ament_cmake REQUIRED)
  find_package(ptafdeploy REQUIRED)

  # The launch test runs the ONNX fixture, which needs the core ORT backend.
  if(NOT DEFINED autoforge_deploy_HAS_ORT OR autoforge_deploy_HAS_ORT)
    add_launch_test(test/test_spinup_launch.py TIMEOUT 60)
  else()
    message(STATUS "Core built without ONNX Runtime: skipping the spinup launch test")
  endif()
endif()

ament_package()

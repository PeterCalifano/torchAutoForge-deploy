cmake_minimum_required(VERSION 3.16)
project(ptafdeploy_spinup)

option(PTAFDEPLOY_ENABLE_CUDA "Accepted for workspace-wide build helper compatibility" OFF)

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

  add_launch_test(test/test_spinup_launch.py TIMEOUT 60)
endif()

ament_package()

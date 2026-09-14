cmake_minimum_required(VERSION 3.21)
project(ptafdeploy_colcon_shim LANGUAGES NONE)

# Resolve the repository root used to embed the canonical CMake project in this colcon package.
get_filename_component(PTAFDEPLOY_REPOSITORY_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

option(PTAFDEPLOY_ENABLE_CUDA "Enable CUDA support in the PTAFDeploy core" OFF)

# Tailor the nested core build for ROS 2 without changing its standalone defaults.
set(autoforge_deploy_ENABLE_CUDA "${PTAFDEPLOY_ENABLE_CUDA}" CACHE BOOL
  "Enable CUDA support in the PTAFDeploy core" FORCE)
set(ENABLE_TESTS OFF CACHE BOOL
  "Disable core tests in the ROS 2 overlay build" FORCE)
set(ENABLE_FETCH_CATCH2 OFF CACHE BOOL
  "Do not fetch Catch2 in the ROS 2 overlay build" FORCE)
set(autoforge_deploy_BUILD_PROGRAMS OFF CACHE BOOL
  "Disable core programs in the ROS 2 overlay build" FORCE)
set(autoforge_deploy_BUILD_EXAMPLES OFF CACHE BOOL
  "Disable core examples in the ROS 2 overlay build" FORCE)

# Build the core in a package-local binary directory while preserving its module discovery.
list(PREPEND CMAKE_MODULE_PATH "${PTAFDEPLOY_REPOSITORY_ROOT}/cmake")
add_subdirectory("${PTAFDEPLOY_REPOSITORY_ROOT}"
  "${CMAKE_CURRENT_BINARY_DIR}/autoforge_deploy_core")

# Install a small compatibility package that forwards consumers to the canonical core package.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/ptafdeployConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/ptafdeployConfig.cmake"
  @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ptafdeployConfig.cmake"
  DESTINATION "lib/cmake/ptafdeploy")

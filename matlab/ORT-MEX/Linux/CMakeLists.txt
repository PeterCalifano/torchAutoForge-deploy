cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(ort-mex-linux)

# Set build options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Collect source files and link libraries
add_executable(ort-cli-linux SHARED "") # Standalone program callable from CLI

# TODO link library, ensure program is built as program + mex file
find_package(ONNXRuntime REQUIRED)

# Define configuration to generate the MEX file
set(MEX_FILE_NAME "ort_mex_caller_linux")
# TODO
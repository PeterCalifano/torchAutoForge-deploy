cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(ptafdeploy_consumer LANGUAGES CXX)

find_package(autoforge_deploy CONFIG REQUIRED)

add_executable(ptafdeploy_consumer main.cpp)
target_compile_features(ptafdeploy_consumer PRIVATE cxx_std_20)
target_link_libraries(
  ptafdeploy_consumer
  PRIVATE autoforge_deploy::autoforge_deploy)

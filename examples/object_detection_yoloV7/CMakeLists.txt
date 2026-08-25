cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(object_detection_yoloV7 VERSION 0.1.0 LANGUAGES CXX)

find_package(autoforge_deploy CONFIG REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui dnn videoio)

file(GLOB example_sources CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB example_headers CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
     "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")

add_executable(${PROJECT_NAME} ${example_sources} ${example_headers})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE
    autoforge_deploy::autoforge_deploy
    ${OpenCV_LIBS})

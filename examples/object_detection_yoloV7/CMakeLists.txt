cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(object_detection_yoloV7 VERSION 0.1.0 LANGUAGES CXX)

# Resolve the installed facade library and image-loading implementation.
find_package(autoforge_deploy CONFIG REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# Reuse the repository-bundled parser without adding it to the installed package contract.
set(TCLAP_COMPILE_TARGET "autoforge_deploy_tclap_interface")
if(NOT TARGET ${TCLAP_COMPILE_TARGET})
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../lib/header_only"
    "${CMAKE_CURRENT_BINARY_DIR}/autoforge_deploy_header_only")
endif()

# Build the standalone facade-backed detector CLI.
add_executable(${PROJECT_NAME} run_object_detection.cpp)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)

target_compile_definitions(${PROJECT_NAME} PRIVATE
  PTAFDEPLOY_CLI_VERSION="${autoforge_deploy_VERSION}")

target_link_libraries(${PROJECT_NAME} PRIVATE
  autoforge_deploy::autoforge_deploy
  ${TCLAP_COMPILE_TARGET}
  ${OpenCV_LIBS})

# Always verify the CLI contract; run real inference only when external fixtures are present.
include(CTest)
if(BUILD_TESTING)
  add_test(NAME object_detection_yoloV7_cli_help
    COMMAND $<TARGET_FILE:${PROJECT_NAME}> --help)

  set_tests_properties(object_detection_yoloV7_cli_help PROPERTIES
    PASS_REGULAR_EXPRESSION "model.ptafmodel")

  set(_yolo_manifest "${CMAKE_CURRENT_SOURCE_DIR}/../model_configs/yolov7_640x640.ptafmodel")
  set(_yolo_image "${CMAKE_CURRENT_SOURCE_DIR}/sample_data/horses.jpg")
  set(_yolo_model "${CMAKE_CURRENT_SOURCE_DIR}/../../models/onnx/yolov7_640x640.onnx")

  if(EXISTS "${_yolo_manifest}" AND EXISTS "${_yolo_image}" AND EXISTS "${_yolo_model}")
    add_test(NAME object_detection_yoloV7_cpu_inference
      COMMAND $<TARGET_FILE:${PROJECT_NAME}>
              "${_yolo_manifest}"
              "${_yolo_image}"
              --target cpu
              --max-detections 5)

    set_tests_properties(object_detection_yoloV7_cpu_inference PROPERTIES
      PASS_REGULAR_EXPRESSION "detections=5"
      TIMEOUT 60)
  endif()
endif()

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(centroiding_models VERSION 0.1.0 LANGUAGES CXX)

# Resolve the installed inference facade and image-processing dependencies.
find_package(autoforge_deploy CONFIG REQUIRED COMPONENTS images inference_output)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# Reuse the repository-bundled CLI parser without exporting it to consumers.
set(TCLAP_COMPILE_TARGET "autoforge_deploy_tclap_interface")
if(NOT TARGET ${TCLAP_COMPILE_TARGET})
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../lib/header_only"
    "${CMAKE_CURRENT_BINARY_DIR}/autoforge_deploy_header_only")
endif()

# Compile model policy once for the CLI and integration tests.
add_library(centroiding_adapter STATIC centroiding_adapter.cpp)
target_include_directories(centroiding_adapter PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(centroiding_adapter PUBLIC autoforge_deploy::images
  autoforge_deploy::inference_output autoforge_deploy::autoforge_deploy)

# Build the standalone facade-backed centroiding CLI.
add_executable(run_centroiding
  run_centroiding.cpp)

target_compile_features(run_centroiding PRIVATE
  cxx_std_20)

target_include_directories(run_centroiding PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}")

target_compile_definitions(run_centroiding PRIVATE
  PTAFDEPLOY_CLI_VERSION="${autoforge_deploy_VERSION}")

target_link_libraries(run_centroiding PRIVATE
  centroiding_adapter
  ${TCLAP_COMPILE_TARGET}
  ${OpenCV_LIBS})

# Keep example-specific behavior covered by native C++ tests.
include(CTest)
if(BUILD_TESTING)
  find_package(Catch2 3 CONFIG REQUIRED)

  add_test(NAME centroiding_cli_help
    COMMAND $<TARGET_FILE:run_centroiding> --help)

  set_tests_properties(centroiding_cli_help PROPERTIES
    PASS_REGULAR_EXPRESSION "model.ptafmodel|model.onnx")

  # Exercise the tracked manifest and PNG only when the ignored ONNX is present.
  set(_centroiding_manifest
    "${CMAKE_CURRENT_SOURCE_DIR}/../model_configs/ml_based_centroiding_plain.ptafmodel")
  string(CONCAT _centroiding_model
    "${CMAKE_CURRENT_SOURCE_DIR}/../../models/onnx/"
    "best_model_plain_traveling-goat-68_22b61bbd4ddd.onnx")
  set(_centroiding_fixture
    "${CMAKE_CURRENT_SOURCE_DIR}/test_data/bright_ellipse_320x240.png")

  if(EXISTS "${_centroiding_model}")
    add_test(NAME centroiding_cli_cpu_inference
      COMMAND $<TARGET_FILE:run_centroiding>
              "${_centroiding_manifest}"
              "${_centroiding_fixture}")

    set_tests_properties(centroiding_cli_cpu_inference PROPERTIES
      PASS_REGULAR_EXPRESSION "centroid.original_x_px="
      TIMEOUT 60)
  endif()

  add_executable(test_centroiding_demo
    test/testCentroidingDemo.cpp)

  target_compile_features(test_centroiding_demo PRIVATE
    cxx_std_20)

  target_include_directories(test_centroiding_demo PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}")

  target_include_directories(test_centroiding_demo SYSTEM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../lib/header_only")

  target_compile_definitions(test_centroiding_demo PRIVATE
    "PTAFDEPLOY_CENTROIDING_FIXTURE=\"${_centroiding_fixture}\"")

  target_link_libraries(test_centroiding_demo PRIVATE
    centroiding_adapter
    Catch2::Catch2WithMain
    ${OpenCV_LIBS})

  include(Catch)
  catch_discover_tests(test_centroiding_demo PROPERTIES
    SKIP_RETURN_CODE 4)
endif()

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(plain_centroiding VERSION 0.1.0 LANGUAGES CXX)

# Resolve the installed inference facade and image-processing dependencies.
find_package(autoforge_deploy CONFIG REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# Reuse the repository-bundled CLI parser without exporting it to consumers.
set(TCLAP_COMPILE_TARGET "autoforge_deploy_tclap_interface")
if(NOT TARGET ${TCLAP_COMPILE_TARGET})
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../lib/header_only"
    "${CMAKE_CURRENT_BINARY_DIR}/autoforge_deploy_header_only")
endif()

# Build the standalone facade-backed centroiding CLI.
add_executable(run_plain_centroiding
  run_plain_centroiding.cpp)

target_compile_features(run_plain_centroiding PRIVATE
  cxx_std_20)

target_include_directories(run_plain_centroiding PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}")

target_compile_definitions(run_plain_centroiding PRIVATE
  PTAFDEPLOY_CLI_VERSION="${autoforge_deploy_VERSION}")

target_link_libraries(run_plain_centroiding PRIVATE
  autoforge_deploy::autoforge_deploy
  ${TCLAP_COMPILE_TARGET}
  ${OpenCV_LIBS})

# Keep example-specific behavior covered by native C++ tests.
include(CTest)
if(BUILD_TESTING)
  find_package(Catch2 3 CONFIG REQUIRED)

  add_test(NAME plain_centroiding_cli_help
    COMMAND $<TARGET_FILE:run_plain_centroiding> --help)

  set_tests_properties(plain_centroiding_cli_help PROPERTIES
    PASS_REGULAR_EXPRESSION "model.ptafmodel|model.onnx")

  # Exercise the tracked manifest and PNG only when the ignored ONNX is present.
  set(_plain_centroiding_manifest
    "${CMAKE_CURRENT_SOURCE_DIR}/../model_configs/ml_based_centroiding_plain.ptafmodel")
  string(CONCAT _plain_centroiding_model
    "${CMAKE_CURRENT_SOURCE_DIR}/../../models/onnx/"
    "best_model_plain_traveling-goat-68_22b61bbd4ddd.onnx")
  set(_plain_centroiding_fixture
    "${CMAKE_CURRENT_SOURCE_DIR}/test_data/bright_ellipse_320x240.png")

  if(EXISTS "${_plain_centroiding_model}")
    add_test(NAME plain_centroiding_cli_cpu_inference
      COMMAND $<TARGET_FILE:run_plain_centroiding>
              "${_plain_centroiding_manifest}"
              "${_plain_centroiding_fixture}")

    set_tests_properties(plain_centroiding_cli_cpu_inference PROPERTIES
      PASS_REGULAR_EXPRESSION "centroid.original_x_px="
      TIMEOUT 60)
  endif()

  add_executable(test_plain_centroiding_demo
    test/testPlainCentroidingDemo.cpp)

  target_compile_features(test_plain_centroiding_demo PRIVATE
    cxx_std_20)

  target_include_directories(test_plain_centroiding_demo PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}")

  target_compile_definitions(test_plain_centroiding_demo PRIVATE
    "PTAFDEPLOY_PLAIN_CENTROIDING_FIXTURE=\"${_plain_centroiding_fixture}\"")

  target_link_libraries(test_plain_centroiding_demo PRIVATE
    autoforge_deploy::autoforge_deploy
    Catch2::Catch2WithMain
    ${OpenCV_LIBS})

  include(Catch)
  catch_discover_tests(test_plain_centroiding_demo PROPERTIES
    SKIP_RETURN_CODE 4)
endif()

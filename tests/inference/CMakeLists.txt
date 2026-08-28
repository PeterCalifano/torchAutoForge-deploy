# Register compiled product tests while excluding known non-test sources.
set(EXCLUDED_LIST "test_to_exclude")

add_tests(${project_name}
  EXCLUDED_LIST TESTS_LIST
  ${CUDA_COMPILE_TARGET}
  CATCH2_TEST_PROPERTIES Catch2::Catch2WithMain)

# Exercise benchmark CLI behavior through the public model facade.
if(ENABLE_TESTS AND BUILD_TESTING AND TARGET benchmark_model)
  set(_benchmark_fixture_model
    "${PROJECT_SOURCE_DIR}/tests/matlab/testSamples/tracedSampleModel.onnx")
  set(_benchmark_centroid_config
    "${CMAKE_CURRENT_SOURCE_DIR}/model_configs/centroiding_fixture.ptafmodel")

  add_test(NAME ${LIB_NAMESPACE}_benchmark_model_cli_help
    COMMAND $<TARGET_FILE:benchmark_model> --help)

  set_tests_properties(${LIB_NAMESPACE}_benchmark_model_cli_help PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/src:$ENV{LD_LIBRARY_PATH}"
    PASS_REGULAR_EXPRESSION "model.ptafmodel")

  add_test(NAME ${LIB_NAMESPACE}_benchmark_model_manifest_smoke
    COMMAND $<TARGET_FILE:benchmark_model>
            "${_benchmark_centroid_config}"
            --iterations 1
            --warmup 0)

  set_tests_properties(${LIB_NAMESPACE}_benchmark_model_manifest_smoke PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/src:$ENV{LD_LIBRARY_PATH}"
    PASS_REGULAR_EXPRESSION "role=centroiding")

  add_test(NAME ${LIB_NAMESPACE}_benchmark_model_raw_onnx_smoke
    COMMAND $<TARGET_FILE:benchmark_model>
            "${_benchmark_fixture_model}"
            --role centroiding
            --targets cpu
            --no-fallback
            --input-shape 1,11
            --iterations 1
            --warmup 0)

  set_tests_properties(${LIB_NAMESPACE}_benchmark_model_raw_onnx_smoke PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/src:$ENV{LD_LIBRARY_PATH}"
    PASS_REGULAR_EXPRESSION "role=centroiding")

  add_test(NAME ${LIB_NAMESPACE}_benchmark_model_rejects_unknown_target
    COMMAND $<TARGET_FILE:benchmark_model>
            "${_benchmark_centroid_config}"
            --targets imaginary
            --iterations 1
            --warmup 0)

  set_tests_properties(${LIB_NAMESPACE}_benchmark_model_rejects_unknown_target PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/src:$ENV{LD_LIBRARY_PATH}"
    WILL_FAIL TRUE)

  add_test(NAME ${LIB_NAMESPACE}_benchmark_model_rejects_negative_trt_profile
    COMMAND $<TARGET_FILE:benchmark_model>
            "${_benchmark_centroid_config}"
            --trt-profile -1
            --iterations 1
            --warmup 0)

  set_tests_properties(${LIB_NAMESPACE}_benchmark_model_rejects_negative_trt_profile PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/src:$ENV{LD_LIBRARY_PATH}"
    WILL_FAIL TRUE)
endif()

# Report ORT provider availability and validate required backend-neutral targets.
if(ENABLE_TESTS AND BUILD_TESTING AND TARGET get_available_providers)
  add_test(NAME ${LIB_NAMESPACE}_get_available_providers_cli_help
    COMMAND $<TARGET_FILE:get_available_providers> --help)

  set_tests_properties(${LIB_NAMESPACE}_get_available_providers_cli_help PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/src:$ENV{LD_LIBRARY_PATH}"
    PASS_REGULAR_EXPRESSION "--require-targets")

  add_test(NAME ${LIB_NAMESPACE}_get_available_providers_cpu_preflight
    COMMAND $<TARGET_FILE:get_available_providers> --require-targets cpu)

  set_tests_properties(${LIB_NAMESPACE}_get_available_providers_cpu_preflight PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/src:$ENV{LD_LIBRARY_PATH}"
    PASS_REGULAR_EXPRESSION "required_targets_available=cpu")
endif()

# Verify metadata-probe help and model-load failure behavior.
if(ENABLE_TESTS AND BUILD_TESTING AND TARGET run_ort_inference)
  add_test(NAME ${LIB_NAMESPACE}_run_ort_inference_cli_help
    COMMAND $<TARGET_FILE:run_ort_inference> --help)

  set_tests_properties(${LIB_NAMESPACE}_run_ort_inference_cli_help PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/src:$ENV{LD_LIBRARY_PATH}"
    PASS_REGULAR_EXPRESSION "model.onnx")

  add_test(NAME ${LIB_NAMESPACE}_run_ort_inference_rejects_missing_model
    COMMAND $<TARGET_FILE:run_ort_inference>
            "${CMAKE_CURRENT_BINARY_DIR}/missing_model.onnx")

  set_tests_properties(${LIB_NAMESPACE}_run_ort_inference_rejects_missing_model PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/src:$ENV{LD_LIBRARY_PATH}"
    WILL_FAIL TRUE)
endif()

# Keep Jetson orchestration coverage as a target-owned dry run on hosted systems.
if(ENABLE_TESTS AND BUILD_TESTING AND TARGET benchmark_model AND TARGET get_available_providers)
  add_test(NAME ${LIB_NAMESPACE}_jetson_runtime_runner_dry_run
    COMMAND ${PYTHON_EXECUTABLE}
            "${PROJECT_SOURCE_DIR}/scripts/run_jetson_runtime_smoke.py"
            --dry-run
            --build-dir "${PROJECT_BINARY_DIR}"
            --ort-model "${PROJECT_SOURCE_DIR}/tests/matlab/testSamples/tracedSampleModel.onnx"
            --device 0
            --iterations 1
            --warmup 0)

  set_tests_properties(${LIB_NAMESPACE}_jetson_runtime_runner_dry_run PROPERTIES
    PASS_REGULAR_EXPRESSION "jetson_runtime_smoke_dry_run=ok")
endif()

# Exercise the generated Python facade when wrapper generation is enabled.
set(_python_wrapper_option_name "${LIB_NAMESPACE}_BUILD_PYTHON_WRAPPER")
set(_python_wrapper_target "${${PROJECT_NAME}_PYTHON_WRAPPER_TARGET}")

if(ENABLE_TESTS
  AND BUILD_TESTING
  AND ${_python_wrapper_option_name}
  AND TARGET ${_python_wrapper_target})

  set(_python_smoke_test_name "${LIB_NAMESPACE}_python_facade_smoke")
  set(_python_fixture_model "${PROJECT_SOURCE_DIR}/tests/matlab/testSamples/tracedSampleModel.onnx")
  set(_python_centroid_config
    "${CMAKE_CURRENT_SOURCE_DIR}/model_configs/centroiding_fixture.ptafmodel")
  set(_python_smoke_script "${PROJECT_SOURCE_DIR}/tests/python/test_inference_facade_smoke.py")
  get_property(_cached_python_executable CACHE PYTHON_EXECUTABLE PROPERTY VALUE)

  if(NOT "${_cached_python_executable}" STREQUAL "")
    set(_python_smoke_executable "${_cached_python_executable}")
  else()
    set(_python_smoke_executable "${PYTHON_EXECUTABLE}")
  endif()

  add_test(NAME ${_python_smoke_test_name}
    COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${PROJECT_BINARY_DIR}/python:$ENV{PYTHONPATH}"
            "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/src:$ENV{LD_LIBRARY_PATH}"
            ${_python_smoke_executable}
            "${_python_smoke_script}"
            --model "${_python_fixture_model}"
            --config "${_python_centroid_config}")

  set_tests_properties(${_python_smoke_test_name} PROPERTIES
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/python")
endif()

# Exercise the generated MATLAB facade when MATLAB and its wrapper are available.
set(_matlab_wrapper_option_name "${LIB_NAMESPACE}_BUILD_MATLAB_WRAPPER")
if(ENABLE_TESTS AND BUILD_TESTING AND ${_matlab_wrapper_option_name})
  find_program(_matlab_smoke_executable
    NAMES matlab
    HINTS "${Matlab_ROOT_DIR}/bin"
    NO_CACHE)

  if(_matlab_smoke_executable AND DEFINED LIBNAME_WRAP_TARGET)
    set(_matlab_wrapper_dir "${PROJECT_BINARY_DIR}/wrap/${LIBNAME_WRAP_TARGET}")
    set(_matlab_mex_dir "${PROJECT_BINARY_DIR}/wrap/${LIBNAME_WRAP_TARGET}_mex")
    set(_matlab_fixture_model
      "${PROJECT_SOURCE_DIR}/tests/matlab/testSamples/tracedSampleModel.onnx")
    set(_matlab_centroid_config
      "${CMAKE_CURRENT_SOURCE_DIR}/model_configs/centroiding_fixture.ptafmodel")
    set(_matlab_object_detection_config
      "${CMAKE_CURRENT_SOURCE_DIR}/model_configs/object_detection_fixture.ptafmodel")
    set(_matlab_smoke_test_name "${LIB_NAMESPACE}_matlab_model_load")

    string(CONCAT _matlab_smoke_command
      "TestInferenceFacadeSmoke('"
      "${_matlab_wrapper_dir}', '"
      "${_matlab_mex_dir}', '"
      "${_matlab_fixture_model}', '"
      "${_matlab_centroid_config}', '"
      "${_matlab_object_detection_config}')")

    add_test(NAME ${_matlab_smoke_test_name}
      COMMAND ${_matlab_smoke_executable}
              -batch
              "${_matlab_smoke_command}")

    set_tests_properties(${_matlab_smoke_test_name} PROPERTIES
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/tests/matlab"
      ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/src:$ENV{LD_LIBRARY_PATH}")
  endif()
endif()

# Export the discovered test targets to the parent test directory.
set(TESTS_LIST ${TESTS_LIST} PARENT_SCOPE)

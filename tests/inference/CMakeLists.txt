# Exclude EXCLUDED_LIST from the list of tests
set(EXCLUDED_LIST "test_to_exclude")  # Specify files to exclude WE

add_tests(${project_name} EXCLUDED_LIST TESTS_LIST ${CUDA_COMPILE_TARGET} CATCH2_TEST_PROPERTIES Catch2::Catch2WithMain)

# Setup python wrapper smoke test
set(_python_wrapper_option_name "${LIB_NAMESPACE}_BUILD_PYTHON_WRAPPER")
set(_python_wrapper_target "${${PROJECT_NAME}_PYTHON_WRAPPER_TARGET}")

if(ENABLE_TESTS AND BUILD_TESTING AND ${_python_wrapper_option_name} AND TARGET ${_python_wrapper_target})

  set(_python_import_test_name "${LIB_NAMESPACE}_python_import")
  set(_python_centroid_config
      "${CMAKE_CURRENT_SOURCE_DIR}/model_configs/centroiding_fixture.ptafmodel")
  get_property(_cached_python_executable CACHE PYTHON_EXECUTABLE PROPERTY VALUE)

  if(NOT "${_cached_python_executable}" STREQUAL "")
    set(_python_smoke_executable "${_cached_python_executable}")
  else()
    set(_python_smoke_executable "${PYTHON_EXECUTABLE}")
  endif()

  set(_python_import_test_code
      "import ${PROJECT_NAME} as module_; assert getattr(module_, 'HAS_WRAPPER', False), 'Expected HAS_WRAPPER=True'; assert hasattr(module_, 'CInferenceManager'), 'Expected CInferenceManager wrapper'; assert hasattr(module_, 'CModelFacade'), 'Expected CModelFacade wrapper'; model_=module_.CModelFacade(); model_.LoadModelConfig(r'${_python_centroid_config}'); assert model_.GetRole() == 'centroiding';")
  
  # Add test for module import
  add_test(
    NAME ${_python_import_test_name}
    COMMAND
      ${CMAKE_COMMAND} -E env
      "PYTHONPATH=${PROJECT_SOURCE_DIR}/python:$ENV{PYTHONPATH}"
      "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/src:$ENV{LD_LIBRARY_PATH}"
      ${_python_smoke_executable} -c "${_python_import_test_code}")
  set_tests_properties(
    ${_python_import_test_name}
    PROPERTIES
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/python")
endif()

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
    set(_matlab_smoke_script
        "${PROJECT_BINARY_DIR}/wrap/${LIBNAME_WRAP_TARGET}_matlab_smoke.m")

    # Write MATLAB wrapper smoke script
    file(WRITE "${_matlab_smoke_script}" "
      addpath('${_matlab_wrapper_dir}');
      addpath('${_matlab_mex_dir}');
      m = ptafdeploy.inference.CInferenceManager('${_matlab_fixture_model}');
      assert(m.GetNumInputs() == 1);
      info = m.GetInputInfo(0);
      assert(strcmp(info.name, 'input'));
      y = ptafdeploy.inference.InferSingleFloatInputVector(m, double((1:11)'), double([1, 11]'));
      assert(numel(y) == 2);
      centroidModel = ptafdeploy.inference.CModelFacade();
      centroidModel.LoadModelConfig('${_matlab_centroid_config}');
      assert(strcmp(centroidModel.GetRole(), 'centroiding'));
      yc = ptafdeploy.inference.InferModelSingleFloatInputVector(centroidModel, double((1:11)'), double([1, 11]'));
      assert(numel(yc) == 2);
      objectModel = ptafdeploy.inference.CModelFacade();
      objectModel.LoadModelConfig('${_matlab_object_detection_config}');
      assert(strcmp(objectModel.GetRole(), 'object_detection'));
      assert(objectModel.GetNumInputs() == 1);
      ")

    set(_matlab_smoke_test_name "${LIB_NAMESPACE}_matlab_model_load")
    
    # Add test for MATLAB wrapper smoke test
    add_test(
      NAME ${_matlab_smoke_test_name}
      COMMAND ${_matlab_smoke_executable} -batch "run('${_matlab_smoke_script}')")

    set_tests_properties(
      ${_matlab_smoke_test_name}
      PROPERTIES
        WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
        ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/src:$ENV{LD_LIBRARY_PATH}")
        
  endif()
endif()

set(TESTS_LIST ${TESTS_LIST} PARENT_SCOPE) # Export list of tests to parent scope (applied ONLY here)

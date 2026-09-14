# Configure optional TensorRT inference requirements on one interface target.

include_guard(GLOBAL)
include(CMakeParseArguments)

# handle_tensorrt(TARGET <target>)
#
# Create the named interface target in every configuration. When
# ENABLE_TENSORRT is true, require the CUDA runtime and TensorRT SDK, then
# propagate the vendor targets and project feature definition to consumers.
function(handle_tensorrt)
  set(one_value_args TARGET)
  cmake_parse_arguments(HTRT "" "${one_value_args}" "" ${ARGN})

  if(NOT HTRT_TARGET)
    set(HTRT_TARGET tensorrt_interface)
  endif()
  if(NOT TARGET ${HTRT_TARGET})
    add_library(${HTRT_TARGET} INTERFACE)
  endif()

  if(NOT ENABLE_TENSORRT)
    return()
  endif()

  # This C++ backend needs CUDA runtime APIs but does not require enabling the
  # CUDA compiler language when the separate CUDA/PTX feature is disabled.
  find_package(CUDAToolkit 12.0 REQUIRED)
  find_package(TensorRT REQUIRED MODULE)

  foreach(_tensorrt_target IN ITEMS TensorRT::nvinfer TensorRT::nvinfer_plugin)
    if(NOT TARGET ${_tensorrt_target})
      message(FATAL_ERROR
          "TensorRT discovery did not define required target ${_tensorrt_target}.")
    endif()
  endforeach()

  target_compile_definitions(${HTRT_TARGET} INTERFACE PTAFDEPLOY_ENABLE_TENSORRT=1)
  target_link_libraries(
      ${HTRT_TARGET} INTERFACE
      TensorRT::nvinfer TensorRT::nvinfer_plugin CUDA::cudart)

  message(STATUS "TensorRT enabled: ${TensorRT_VERSION}")
endfunction()

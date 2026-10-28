# Configure optional ONNX Runtime backend requirements on one interface target.

include_guard(GLOBAL)
include(CMakeParseArguments)

# handle_ort(TARGET <target>)
#
# Create the named interface target in every configuration. When ENABLE_ORT is
# true, resolve ONNX Runtime from ONNXRUNTIME_ROOT / onnxruntime_DIR hints, then
# propagate the vendor target and project feature definition to consumers.
function(handle_ort)
  set(one_value_args TARGET)
  cmake_parse_arguments(HORT "" "${one_value_args}" "" ${ARGN})

  if(NOT HORT_TARGET)
    set(HORT_TARGET ort_interface)
  endif()
  if(NOT TARGET ${HORT_TARGET})
    add_library(${HORT_TARGET} INTERFACE)
  endif()

  if(NOT ENABLE_ORT)
    return()
  endif()

  set(ONNXRUNTIME_ROOT "" CACHE PATH
      "Optional ONNX Runtime install root (expects lib/cmake/onnxruntime)")
  if("${ONNXRUNTIME_ROOT}" STREQUAL "" AND DEFINED ENV{ONNXRUNTIME_ROOT}
     AND NOT "$ENV{ONNXRUNTIME_ROOT}" STREQUAL "")
    set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH
        "Optional ONNX Runtime install root (expects lib/cmake/onnxruntime)" FORCE)
  endif()
  if((NOT DEFINED onnxruntime_DIR OR "${onnxruntime_DIR}" STREQUAL "")
     AND EXISTS "${ONNXRUNTIME_ROOT}/lib/cmake/onnxruntime")
    set(onnxruntime_DIR "${ONNXRUNTIME_ROOT}/lib/cmake/onnxruntime" CACHE PATH
        "Path to onnxruntime CMake package" FORCE)
  endif()
  if((NOT DEFINED onnxruntime_INCLUDE_DIRS OR "${onnxruntime_INCLUDE_DIRS}" STREQUAL "")
     AND EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime")
    set(onnxruntime_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include/onnxruntime" CACHE PATH
        "Optional ONNX Runtime include directory hint")
  endif()

  find_package(onnxruntime REQUIRED)

  target_compile_definitions(${HORT_TARGET} INTERFACE PTAFDEPLOY_ENABLE_ORT=1)
  target_link_libraries(${HORT_TARGET} INTERFACE onnxruntime::onnxruntime)

  # find_package may define this as a plain variable; keep it visible to the
  # caller because the library adds it as a private include directory.
  if(DEFINED onnxruntime_INCLUDE_DIRS)
    set(onnxruntime_INCLUDE_DIRS "${onnxruntime_INCLUDE_DIRS}" PARENT_SCOPE)
  endif()

  message(STATUS "ONNX Runtime enabled: ${onnxruntime_VERSION}")
endfunction()

#[=======================================================================[.rst:
FindTensorRT
------------

Find NVIDIA TensorRT headers and runtime libraries for the optional inference
backend and installed consumers.

Input hints
^^^^^^^^^^^

``TensorRT_ROOT``
  Preferred CMake package-root spelling.
``TENSORRT_ROOT``
  Compatibility spelling used by existing TensorRT projects.

Result variables
^^^^^^^^^^^^^^^^

``TensorRT_FOUND``
``TensorRT_VERSION``
``TensorRT_INCLUDE_DIRS``
``TensorRT_LIBRARIES``

Imported targets
^^^^^^^^^^^^^^^^

``TensorRT::nvinfer``
``TensorRT::nvinfer_plugin``
#]=======================================================================]

include(FindPackageHandleStandardArgs)

if(NOT DEFINED TensorRT_ROOT)
  set(TensorRT_ROOT "" CACHE PATH
      "TensorRT root directory containing include/ and lib directories.")
endif()
if(NOT DEFINED TENSORRT_ROOT)
  set(TENSORRT_ROOT "" CACHE PATH
      "Compatibility TensorRT root directory hint.")
endif()

# Accept package-style and established compatibility hints before conventional
# system locations. Environment variables follow the same precedence.
set(_TensorRT_root_hints)
foreach(_TensorRT_root_variable IN ITEMS TensorRT_ROOT TENSORRT_ROOT)
  if(DEFINED ${_TensorRT_root_variable}
      AND NOT "${${_TensorRT_root_variable}}" STREQUAL "")
    list(APPEND _TensorRT_root_hints "${${_TensorRT_root_variable}}")
  endif()
  if(DEFINED ENV{${_TensorRT_root_variable}}
      AND NOT "$ENV{${_TensorRT_root_variable}}" STREQUAL "")
    list(APPEND _TensorRT_root_hints "$ENV{${_TensorRT_root_variable}}")
  endif()
endforeach()
list(APPEND _TensorRT_root_hints
     /usr/local/tensorrt
     /usr/local/TensorRT)
list(REMOVE_DUPLICATES _TensorRT_root_hints)

# TensorRT archives use targets/<triplet> on multiple host architectures. Use
# the active toolchain triplet first and retain common archive layouts as
# fallbacks for metadata-only and cross-toolchain discovery.
set(_TensorRT_target_architectures)
if(CMAKE_LIBRARY_ARCHITECTURE)
  list(APPEND _TensorRT_target_architectures
       "${CMAKE_LIBRARY_ARCHITECTURE}")
endif()
list(APPEND _TensorRT_target_architectures
     x86_64-linux-gnu
     aarch64-linux-gnu)
list(REMOVE_DUPLICATES _TensorRT_target_architectures)

set(_TensorRT_include_suffixes include)
set(_TensorRT_library_suffixes lib lib64)
foreach(_TensorRT_target_architecture IN LISTS _TensorRT_target_architectures)
  list(APPEND _TensorRT_include_suffixes
       "targets/${_TensorRT_target_architecture}/include")
  list(APPEND _TensorRT_library_suffixes
       "targets/${_TensorRT_target_architecture}/lib"
       "targets/${_TensorRT_target_architecture}/lib64")
endforeach()

find_path(
  TensorRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS ${_TensorRT_root_hints}
  PATH_SUFFIXES ${_TensorRT_include_suffixes})

find_library(
  TensorRT_NVINFER_LIBRARY
  NAMES nvinfer
  HINTS ${_TensorRT_root_hints}
  PATH_SUFFIXES ${_TensorRT_library_suffixes})

find_library(
  TensorRT_NVINFER_PLUGIN_LIBRARY
  NAMES nvinfer_plugin
  HINTS ${_TensorRT_root_hints}
  PATH_SUFFIXES ${_TensorRT_library_suffixes})

# Parse the public version header without compiling or executing vendor code.
if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h"
       _TensorRT_version_lines
       REGEX "^#define NV_TENSORRT_(MAJOR|MINOR|PATCH|BUILD)[ \t]+[0-9]+")
  foreach(_TensorRT_component MAJOR MINOR PATCH BUILD)
    foreach(_TensorRT_version_line IN LISTS _TensorRT_version_lines)
      if(_TensorRT_version_line MATCHES
         "^#define NV_TENSORRT_${_TensorRT_component}[ \t]+([0-9]+)")
        set(_TensorRT_version_${_TensorRT_component} "${CMAKE_MATCH_1}")
      endif()
    endforeach()
  endforeach()
  if(DEFINED _TensorRT_version_MAJOR
      AND DEFINED _TensorRT_version_MINOR
      AND DEFINED _TensorRT_version_PATCH)
    set(TensorRT_VERSION
        "${_TensorRT_version_MAJOR}.${_TensorRT_version_MINOR}.${_TensorRT_version_PATCH}")
    if(DEFINED _TensorRT_version_BUILD)
      string(APPEND TensorRT_VERSION ".${_TensorRT_version_BUILD}")
    endif()
  endif()
endif()

find_package_handle_standard_args(
  TensorRT
  REQUIRED_VARS
    TensorRT_INCLUDE_DIR
    TensorRT_NVINFER_LIBRARY
    TensorRT_NVINFER_PLUGIN_LIBRARY
  VERSION_VAR TensorRT_VERSION)

if(TensorRT_FOUND)
  if(NOT TARGET TensorRT::nvinfer)
    add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
    set_target_properties(
      TensorRT::nvinfer
      PROPERTIES
        IMPORTED_LOCATION "${TensorRT_NVINFER_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
  endif()

  if(NOT TARGET TensorRT::nvinfer_plugin)
    add_library(TensorRT::nvinfer_plugin UNKNOWN IMPORTED)
    set_target_properties(
      TensorRT::nvinfer_plugin
      PROPERTIES
        IMPORTED_LOCATION "${TensorRT_NVINFER_PLUGIN_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES TensorRT::nvinfer)
  endif()
endif()

set(TensorRT_INCLUDE_DIRS "${TensorRT_INCLUDE_DIR}")
set(TensorRT_LIBRARIES
    "${TensorRT_NVINFER_LIBRARY};${TensorRT_NVINFER_PLUGIN_LIBRARY}")
set(TENSORRT_INCLUDE_DIRS "${TensorRT_INCLUDE_DIRS}")
set(TENSORRT_LIBRARIES "${TensorRT_LIBRARIES}")

mark_as_advanced(
  TensorRT_INCLUDE_DIR
  TensorRT_NVINFER_LIBRARY
  TensorRT_NVINFER_PLUGIN_LIBRARY)

unset(_TensorRT_root_hints)
unset(_TensorRT_root_variable)
unset(_TensorRT_target_architectures)
unset(_TensorRT_target_architecture)
unset(_TensorRT_include_suffixes)
unset(_TensorRT_library_suffixes)
unset(_TensorRT_version_lines)
unset(_TensorRT_version_line)
unset(_TensorRT_component)
unset(_TensorRT_version_MAJOR)
unset(_TensorRT_version_MINOR)
unset(_TensorRT_version_PATCH)
unset(_TensorRT_version_BUILD)

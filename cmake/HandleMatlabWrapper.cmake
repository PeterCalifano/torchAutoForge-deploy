# Configure gtwrap MATLAB bindings and their generated toolbox output.
#
# This module owns MATLAB discovery and wrapper generation; common gtwrap
# checkout and interface resolution remains in HandleWrapper.cmake.
include_guard(GLOBAL)

# Configure the project's gtwrap-generated MATLAB wrapper.
function(configure_matlab_gtwrapper)
  message(STATUS "Configuring MATLAB wrap...")

  if(NOT GTWRAP_INTERFACE_FILES)
    message(FATAL_ERROR
      "GTWRAP_INTERFACE_FILES is empty. Cannot build MATLAB wrapper.")
  endif()

  if(NOT COMMAND wrap_and_install_library)
    include(MatlabWrap)
  endif()

  message(STATUS "Including MATLAB directories...")
  # Let container launchers expose a host MATLAB installation without
  # overriding an explicit CMake cache or command-line selection.
  if((NOT DEFINED Matlab_ROOT_DIR OR "${Matlab_ROOT_DIR}" STREQUAL "")
      AND DEFINED ENV{MATLAB_ROOT_DIR}
      AND NOT "$ENV{MATLAB_ROOT_DIR}" STREQUAL "")
    set(Matlab_ROOT_DIR "$ENV{MATLAB_ROOT_DIR}")
  endif()
  find_package(Matlab REQUIRED)
  set(MATLAB_MEX_INCLUDE "${Matlab_ROOT_DIR}/extern/include")

  message(STATUS "MATLAB_MEX_INCLUDE directory: ${MATLAB_MEX_INCLUDE}")
  message(STATUS "Matlab_MEX_LIBRARY directory: ${Matlab_MEX_LIBRARY}")
  message(STATUS "Matlab_MX_LIBRARY directory: ${Matlab_MX_LIBRARY}")

  include_directories(${Matlab_INCLUDE_DIRS})
  include_directories(${MATLAB_MEX_INCLUDE})
  if(DEFINED GTWRAP_INCLUDE_DIR)
    include_directories(${GTWRAP_INCLUDE_DIR})
  endif()

  set(WRAP_MEX_BUILD_STATIC_MODULE OFF)
  set(WRAP_TOOLBOX_INSTALL_PATH "matlab")
  set(WRAP_BUILD_TYPE_POSTFIXES OFF)

  if(NOT DEFINED LIBNAME_WRAP_TARGET)
    message(FATAL_ERROR
      "LIBNAME_WRAP_TARGET is not defined. Cannot configure the MATLAB "
      "wrapper.")
  endif()

  message(STATUS "Using interface files: ${GTWRAP_INTERFACE_FILES}")
  wrap_and_install_library(
    "${GTWRAP_INTERFACE_FILES}"
    "${LIBNAME_WRAP_TARGET}"
    "" "" "" "" OFF)
endfunction()

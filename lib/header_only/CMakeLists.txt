# Keep bundled header-only dependencies private to source-built applications.
include(GNUInstallDirs)

# Reject partial vendor imports before defining any consuming targets.
if(NOT DEFINED TCLAP_COMPILE_TARGET OR "${TCLAP_COMPILE_TARGET}" STREQUAL "")
  message(FATAL_ERROR "TCLAP_COMPILE_TARGET must name the bundled TCLAP interface target")
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tclap/CmdLine.h"
   OR NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tclap/COPYING")
  message(FATAL_ERROR
          "The bundled TCLAP headers and COPYING file are required for native CLI targets")
endif()

# Expose TCLAP only to source-built executables and suppress warnings from vendor headers.
if(NOT TARGET ${TCLAP_COMPILE_TARGET})
  add_library(${TCLAP_COMPILE_TARGET} INTERFACE)

  target_include_directories(${TCLAP_COMPILE_TARGET} SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
endif()

# Binary distributions containing instantiated TCLAP code retain its license notice.
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/tclap/COPYING"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/${PROJECT_NAME}/tclap")

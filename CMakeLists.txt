# VERSION CONTROL 
cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
set(PROJECT_PYTHON_VERSION 3.11)
set(MATLAB_DEFAULT_RELEASE "R2023b")

#set(CMAKE_C_COMPILER gcc-11)
#set(CMAKE_CXX_COMPILER g++-11)
#set(CMAKE_CUDA_COMPILER nvcc)

#####
# Temporary for development. This should be set by the user or as environment variable.
set(HOME $ENV{HOME})
set(onnxruntime_DIR ${HOME}/devDir/ML-repos/onnxruntime/install/lib/cmake/onnxruntime)
set(onnxruntime_INCLUDE_DIRS ${HOME}/devDir/ML-repos/onnxruntime/install/include/onnxruntime) # DOUBT why the hell the cmake target does not include the include directory...?

#####

# Manually specify build type
set(CMAKE_BUILD_TYPE Debug)

# PROJECT DEFINITION
set(project_name "autoforge_deploy")

# Set install default directory if not specified
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install CACHE PATH "Install path" FORCE)
endif()

# Major options
option(ENABLE_OPTIX "Enable OptiX" OFF)
option(ENABLE_CUDA "Enable CUDA" OFF)
include(cmake/HandleOpenMP.cmake)

set(languages CXX) # ACHTUNG: PTX code requires C language!

if (ENABLE_OPTIX)
  set(ENABLE_CUDA ON)
endif()

if (ENABLE_CUDA)
    list(APPEND languages CUDA)
    if (ENABLE_OPTIX)
      list(APPEND languages C)
    endif()
endif()

# Add cmake folder to path
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Add git version handling
include(cmake/HandleGitVersion.cmake)

# Set version number
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 1)
set(PROJECT_VERSION_PATCH 0)

# Get version from git if possible
get_version_from_git()

project( ${project_name} 
        LANGUAGES ${languages}
        VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH})

# Define variable for build as subproject (compare source dir)
set(BUILD_AS_MAIN_PROJECT ON)
if(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(BUILD_AS_MAIN_PROJECT OFF)
endif()

# Define variable for namespacing (required to prevent global target clashes!)
set(LIB_NAMESPACE "${PROJECT_NAME}")
if (DEFINED LIB_NAMESPACE_OVERRIDE)
  set(LIB_NAMESPACE "${LIB_NAMESPACE_OVERRIDE}")
endif()
set(CUDA_COMPILE_TARGET "${LIB_NAMESPACE}_cuda_compile_interface")
set(OPTIX_COMPILE_TARGET "${LIB_NAMESPACE}_optix_compile_interface")
set(OPENGL_COMPILE_TARGET "${LIB_NAMESPACE}_opengl_compile_interface")
set(LIB_COMPILE_TARGET "${LIB_NAMESPACE}_lib_compile_interface")
set(SANITIZER_TARGET "${LIB_NAMESPACE}_sanitizer_target_interface")

# Handle toolchain settings (after project so compiler/system vars are populated)
include(cmake/toolchains/HandleToolchain.cmake)

# Configure cmake targets
include(cmake/HandleCustomTargets.cmake)

# Add build type if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel, NOPTIM)" FORCE)
endif()
string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWER)

# Guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

# Add GNU install dirs
include(GNUInstallDirs)

# Add cmake modules
include(ExternalProject)

# Add CUDA/OptiX tools
include(cmake/HandleCUDA.cmake)
include(cmake/HandleOptiX.cmake)

# Add handle of python
include(cmake/HandlePython.cmake)
include(cmake/HandleWrapper.cmake)

# OPTIONS
option(ENABLE_TESTS              "Build and run tests" ON)
option(ENABLE_FETCH_CATCH2 "Fetch Catch2 from GitHub if not found" ON)
option(VERBOSE "Verbose output" OFF) # Defines VERBOSE flag in code if set to ON

if (NOT DEFINED GTWRAP_BUILD_PYTHON_DEFAULT)
  set(GTWRAP_BUILD_PYTHON_DEFAULT OFF CACHE BOOL "Default Python wrapper setting for all subprojects")
endif()
if (NOT DEFINED GTWRAP_BUILD_MATLAB_DEFAULT)
  set(GTWRAP_BUILD_MATLAB_DEFAULT OFF CACHE BOOL "Default MATLAB wrapper setting for all subprojects")
endif()

set(GTWRAP_PYTHON_OPTION_NAME "${LIB_NAMESPACE}_BUILD_PYTHON_WRAPPER")
set(GTWRAP_MATLAB_OPTION_NAME "${LIB_NAMESPACE}_BUILD_MATLAB_WRAPPER")

option(${GTWRAP_PYTHON_OPTION_NAME} "Build Python wrapper" ${GTWRAP_BUILD_PYTHON_DEFAULT})
option(${GTWRAP_MATLAB_OPTION_NAME} "Build MATLAB wrapper" ${GTWRAP_BUILD_MATLAB_DEFAULT})

option(NO_OPTIMIZATION "Disable optimization flags" OFF)
option(WARNINGS_ARE_ERRORS "Treat warnings as errors" OFF)
option(USE_SYS_OPTIX_SDK "Use Optix system folder if specified" OFF)
option(ENABLE_SIMD "Enable SIMD instructions" ON)
option(SANITIZE_BUILD "Enable sanitizers in build" OFF)
set(SANITIZERS "address,undefined,leak" CACHE STRING "Sanitizers to enable when SANITIZE_BUILD is ON (comma-separated)")
set(PYTHON_EXECUTABLE "python3") # Specify the Python executable to use

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# TIP: if different options need to be specified for different targets of the build, use INTERFACE targets

# COMPILER FLAGS
set(_common_c_warnings -Wall -Wextra)
set(_common_cxx_warnings ${_common_c_warnings} -Wsuggest-override)

set(_debug_flags -Og -g -fmax-errors=3)
set(_debug_sanitizers)
if(SANITIZE_BUILD AND SANITIZERS)
  set(_debug_sanitizers -fsanitize=${SANITIZERS})
endif()
set(_release_flags -O3 -fmax-errors=1)
set(_relwithdebinfo_flags -O2 -g)
set(_noptim_flags -O0 -g)
set(_safety_warnings -Wnull-dereference -Wfloat-equal -fdelete-null-pointer-checks -Wconversion)

set(_debug_c_flags ${_debug_flags} ${_common_c_warnings} ${_debug_sanitizers})
set(_debug_cxx_flags ${_debug_flags} ${_common_cxx_warnings} ${_debug_sanitizers})
set(_release_c_flags ${_release_flags} ${_common_c_warnings})
set(_release_cxx_flags ${_release_flags} ${_common_cxx_warnings})
set(_relwithdebinfo_c_flags ${_relwithdebinfo_flags} ${_common_c_warnings} ${_safety_warnings})
set(_relwithdebinfo_cxx_flags ${_relwithdebinfo_flags} ${_common_cxx_warnings} ${_safety_warnings} -Wnon-virtual-dtor)
set(_noptim_c_flags ${_noptim_flags} ${_common_c_warnings} ${_safety_warnings})
set(_noptim_cxx_flags ${_noptim_flags} ${_common_cxx_warnings} ${_safety_warnings} -Wnon-virtual-dtor)

string(JOIN " " CMAKE_C_FLAGS_DEBUG ${_debug_c_flags})
string(JOIN " " CMAKE_CXX_FLAGS_DEBUG ${_debug_cxx_flags})
string(JOIN " " CMAKE_C_FLAGS_RELEASE ${_release_c_flags})
string(JOIN " " CMAKE_CXX_FLAGS_RELEASE ${_release_cxx_flags})
string(JOIN " " CMAKE_C_FLAGS_RELWITHDEBINFO ${_relwithdebinfo_c_flags})
string(JOIN " " CMAKE_CXX_FLAGS_RELWITHDEBINFO ${_relwithdebinfo_cxx_flags})
string(JOIN " " CMAKE_C_FLAGS_NOPTIM ${_noptim_c_flags})
string(JOIN " " CMAKE_CXX_FLAGS_NOPTIM ${_noptim_cxx_flags})

# Handle OpenMP
handle_openmp()

# Handle sanitizers
include(cmake/HandleSanitizers.cmake)
handle_sanitizers()

if (WARNINGS_ARE_ERRORS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    message(WARNING "ACHTUNG! Warnings will be treated treated as errors.")
endif()

message(STATUS "==============================================================")
message(STATUS "====================  Dependencies ===========================")

# LIBRARY DEPENDENCIES DEFINITION
find_package( Eigen3 3.4 REQUIRED )
find_package( Doxygen QUIET)

# onnxruntime
add_library(ort_target_interface INTERFACE)
find_package(onnxruntime REQUIRED)
if (onnxruntime_FOUND)
  message(STATUS "ONNX Runtime found: ${onnxruntime_VERSION}")
  message(STATUS "ONNX Runtime include directories: ${onnxruntime_INCLUDE_DIRS}")
  target_link_libraries(ort_target_interface INTERFACE onnxruntime::onnxruntime)
else()
  message(FATAL_ERROR "ONNX Runtime not found. Please build onnxruntime and set onnxruntime_DIR environment variable to the build directory.")
endif()

# Handle Catch2 tools
include(cmake/HandleCatch2.cmake)

# Include cmake tools (depends on Catch2)
include(cmake/cmake_utils.cmake)

# Configure CUDA/OptiX (creates interface targets even when disabled)
handle_cuda(TARGET ${CUDA_COMPILE_TARGET} 
            MIN_VERSION 12.0)
handle_optix(TARGET ${OPTIX_COMPILE_TARGET} 
            CUDA_TARGET ${CUDA_COMPILE_TARGET})

if (ENABLE_CUDA)
  # Add tools to manage cuda and ptx
  include(cmake/cmake_cuda_ptx_tools.cmake)
endif()

# Handle OpenGL (creates interface targets even when disabled)
include(cmake/HandleOpenGL.cmake)
handle_opengl(TARGET ${OPENGL_COMPILE_TARGET})

# Add submodules if any found
include(cmake/HandleSubmodules.cmake)
handle_submodules() # Automatically adds subdirectories in lib/ (and build them if CMakeLists.txt found)

# Set the dependencies for the automated export
set(EXPORT_TARGET_DEPS Eigen3 onnxruntime) # TODO make automatic based on used packages

if(ENABLE_CUDA)
  list(APPEND EXPORT_TARGET_DEPS CUDAToolkit)
endif()
if (ENABLE_OPENGL)
  list(APPEND EXPORT_TARGET_DEPS OpenGL::GL)
endif()

message(STATUS "Target dependencies: ${EXPORT_TARGET_DEPS}")

#find_package( GTSAM 4.0 REQUIRED )
#find_package( GTSAM_UNSTABLE REQUIRED)
#find_package( sopengv )
#find_package( OpenCV core imgproc highgui features2d calib3d video)
#find_package( DBoW2 )
#find_package( Pangolin )
#find_package( jsoncpp )
#find_package( rapid_yaml)

# Interface target for C++ flags
add_library(${LIB_COMPILE_TARGET} INTERFACE)

if (CMAKE_BUILD_TYPE_LOWER MATCHES "debug")
  # Set compile flags
  set(CMAKE_C_FLAGS ${CMAKE_CXX_FLAGS_DEBUG})
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_DEBUG})

  # Set compile options for the interface target
  set(_sanitizer_flags)
  if(SANITIZE_BUILD AND SANITIZERS)
    set(_sanitizer_flags -fsanitize=${SANITIZERS})
  endif()
  target_compile_options(${LIB_COMPILE_TARGET} INTERFACE
                      ${_sanitizer_flags}
                      -fno-omit-frame-pointer
  )

  target_link_options(${LIB_COMPILE_TARGET} INTERFACE
                      ${_sanitizer_flags}
                      -fno-omit-frame-pointer
  )

elseif (CMAKE_BUILD_TYPE_LOWER MATCHES "release" AND NOT NO_OPTIMIZATION)
  # Set compile flags
  set(CMAKE_C_FLAGS ${CMAKE_CXX_FLAGS_RELEASE})
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_RELEASE})

elseif (CMAKE_BUILD_TYPE_LOWER MATCHES "relwithdebinfo" AND NOT NO_OPTIMIZATION) 
  # Set compile flags
  set(CMAKE_C_FLAGS ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})

elseif (NO_OPTIMIZATION)
  set(CMAKE_C_FLAGS ${CMAKE_CXX_FLAGS_NOPTIM})
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_NOPTIM})
  target_compile_definitions(${LIB_COMPILE_TARGET} INTERFACE NDEBUG)

else()
  # Stop build
  message(FATAL_ERROR "Unknown build type: ${CMAKE_BUILD_TYPE}. Stopping build.")
endif()

# Add catch2 for unit testing
find_package(OpenCV QUIET)

# Create the installation target list 
set(installable_targets)

# TIP: To set target compilation options (or any other thing) that must be specific for a target or propagated, use the following cmake pattern (example). Requires cmake 3.15 or later. Here flag BUILD_INTERFACE regulates whether the compile flags are added to the target_compile_options. If true, they are added.
#add_library(library_compile_flags INTERFACE)
#target_compile_options(library_compile_flags INTERFACE 
#"$<${gcc_like_cxx}: $<BUILD_INTERFACE: -Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused>>")
#target_link_libraries(library_target PUBLIC any_other_target library_compile_flags)

# Define include target interface
add_library(${project_name}_include INTERFACE)

# Define include directories for install compatibility
target_include_directories(${project_name}_include INTERFACE 
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>  # Used during development
                           #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/stb_PeterCdev>  
                           #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/rapidyaml_PeterCdev>
                           #$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/gdt>
                           #$<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}> 
                           #$<BUILD_INTERFACE:${OPTIX_INCLUDE_DIR}>   
                           #$<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>  # Add OpenCV include directories
                           $<INSTALL_INTERFACE:include/${project_name}>  # Used in installed projects
)

# Build/install tree for runtime programs: $ORIGIN/../lib
target_link_options(${project_name}_include INTERFACE
                    $<$<PLATFORM_ID:Linux>:-Wl,-rpath,\$ORIGIN/../lib>
                  )

# Install-tree exes look next to themselves
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

# Add libraries subdirectories if needed
include_directories( lib/header_only ) # tclap, fast_obj, stb image, rapidyaml, rapidjson

message(STATUS "===================================================================")
message(STATUS "====================  Library composition  ========================")

# Add library
add_subdirectory( src )

# Add examples to build
if (BUILD_AS_MAIN_PROJECT)
  add_subdirectory( examples )
endif()

# Add tests to build
if (ENABLE_TESTS AND Catch2_FOUND AND BUILD_AS_MAIN_PROJECT)
  add_subdirectory( tests )
endif()

# Add documentation
set(DOC_SUBDIRS "${${project_name}_INCLUDE_DIRS}")
if (DOXYGEN_FOUND)
    add_subdirectory( "doc" )
else()
    message(STATUS "Doxygen not found. Skipping building documentation.") 
endif()

###########################################################################
### WRAPPERS CONFIGURATION
set(LIBNAME_WRAP_TARGET ${project_name})
handle_gtwrappers() # You may also pass WRAP_INTERFACE_FILES variable to specify interface files, else autofetching is used
handle_pybind11_wrapper()

# Determine if any wrapper is enabled
set(ANY_WRAPPER_ENABLED OFF)
if (${GTWRAP_PYTHON_OPTION_NAME} OR PYBIND11_WRAPPER_ENABLED OR ${GTWRAP_MATLAB_OPTION_NAME})
  set(ANY_WRAPPER_ENABLED ON)
endif()

###########################################################################
# INFO SECTION
# Print configuration variables
message(STATUS "===============================================================")
message(STATUS "================  Configuration Options  ======================")
# Project info
message(STATUS "LIBRARY NAME                              : ${project_name}")
message(STATUS "CROSS COMPILING                           : ${CMAKE_CROSSCOMPILING}")
message(STATUS "HOST SYSTEM                               : ${CMAKE_HOST_SYSTEM}")
message(STATUS "TARGET SYSTEM                             : ${CMAKE_SYSTEM}")
message(STATUS "TARGET ARCHITECTURE                       : ${CMAKE_SYSTEM_PROCESSOR}")
# CXX compiler
message(STATUS "CMAKE_CXX_COMPILER                        : ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER_ID type                : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION                : ${CMAKE_CXX_COMPILER_VERSION}")

if (ENABLE_OPTIX)
  # C compiler
  message(STATUS "CMAKE_C_COMPILER                          : ${CMAKE_C_COMPILER}")
  message(STATUS "CMAKE_C_COMPILER_ID type                  : ${CMAKE_C_COMPILER_ID}")
  message(STATUS "CMAKE_C_COMPILER_VERSION                  : ${CMAKE_C_COMPILER_VERSION}")
endif()

# CUDA arch and compiler
if (ENABLE_CUDA)
  message(STATUS "CUDA enabled                             : ${ENABLE_CUDA}")
  message(STATUS "CUDA architecture                        : ${cuda_arch}")
  message(STATUS "CUDA architecture sm version             : ${sm_version}")
  message(STATUS "CMAKE_CUDA_COMPILER                      : ${CMAKE_CUDA_COMPILER}")
  message(STATUS "CMAKE_CUDA_COMPILER_ID type              : ${CMAKE_CUDA_COMPILER_ID}")
  message(STATUS "CMAKE_CUDA_COMPILER_VERSION              : ${CMAKE_CUDA_COMPILER_VERSION}")
endif()

# Installation
message(STATUS "CMAKE_INSTALL_PREFIX                      : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_INSTALL_LIBDIR                      : ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "CMAKE_INSTALL_INCLUDEDIR                  : ${CMAKE_INSTALL_INCLUDEDIR}")
message(STATUS "CMAKE_INSTALL_BINDIR                      : ${CMAKE_INSTALL_BINDIR}")

# Build type
if(NOT MSVC AND NOT XCODE_VERSION)
  message(STATUS "Build type                                : ${CMAKE_BUILD_TYPE}")  
  message(STATUS "Optimization override:                    : ${NO_OPTIMIZATION}")  

  if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "  C compilation flags                     : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")
    message(STATUS "  C++ compilation flags                   : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
  elseif (CMAKE_BUILD_TYPE MATCHES Release)
    message(STATUS "  C compilation flags                     : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
    message(STATUS "  C++ compilation flags                   : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
  elseif (CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    message(STATUS "  C compilation flags                     : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELWITHDEBINFO}")
    message(STATUS "  C++ compilation flags                   : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
  elseif (NO_OPTIMIZATION)
    message(STATUS "  C compilation flags                     : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_NOPTIM}")
    message(STATUS "  C++ compilation flags                   : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_NOPTIM}")
  else()
    message(STATUS "  C compilation flags                     : ${CMAKE_C_FLAGS}")
    message(STATUS "  C++ compilation flags                   : ${CMAKE_CXX_FLAGS}")
  endif()

endif()
# Print info about wrappers if any
if (ANY_WRAPPER_ENABLED)
  message(STATUS "================  Wrappers Configuration  ======================")
  message(STATUS "Python pybind11 direct wrapper              : ${PYBIND11_WRAPPER_ENABLED}")
  message(STATUS "Python gtwrapper                            : ${${GTWRAP_PYTHON_OPTION_NAME}}")
  message(STATUS "MATLAB gtwrapper                            : ${${GTWRAP_MATLAB_OPTION_NAME}}")
  message(STATUS "MATLAB default release                      : ${MATLAB_DEFAULT_RELEASE}")
  message(STATUS "Python executable                           : ${PYTHON_EXECUTABLE}")
  message(STATUS "Python version                              : ${PROJECT_PYTHON_VERSION}")
endif()
message(STATUS "===============================================================")

# AUXILIARY 
# Set dashboard properties (needed for CDash)
set(CTEST_PROJECT_NAME ${project_name})
set(CTEST_NIGHTLY_START_TIME "00:00:00 CET")

# Set the dashboard drop method and location
#set(CTEST_DROP_METHOD "http")
#set(CTEST_DROP_SITE "my.cdash.org")
#set(CTEST_DROP_LOCATION "/submit.php?project=CMakeTutorial")
#set(CTEST_DROP_SITE_CDASH TRUE)

# PACKAGING CONFIGURATION 
include(InstallRequiredSystemLibraries)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_GENERATOR "TGZ")
set(CPACK_SOURCE_GENERATOR "TGZ")
include(CPack)

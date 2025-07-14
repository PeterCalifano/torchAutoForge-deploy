# VERSION CONTROL 
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

#set(CMAKE_C_COMPILER gcc-11)
#set(CMAKE_CXX_COMPILER g++-11)
#set(CMAKE_CUDA_COMPILER nvcc)

# Manually specify build type
set(CMAKE_BUILD_TYPE Debug)

# PROJECT DEFINITION
set(project_name "template_project")

option(ENABLE_OPTIX "Enable OptiX" OFF)
option(ENABLE_CUDA "Enable CUDA" OFF)

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

# Set version number
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 1)
set(PROJECT_VERSION_PATCH 0)

project( ${project_name} 
        LANGUAGES ${languages}
        VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}") # ACHTUNG: PTX code requires C language!

# Load scripts and install dirs
include(cmake/cmake_utils.cmake)
include(GNUInstallDirs)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# OPTIONS
option(ENABLE_OMP "Enable OpenMP" OFF)
if(NOT DEFINED NUM_OMP_THREADS)
    set(NUM_OMP_THREADS 4 "Number of OpenMP threads to use")
endif()

option(USE_SYS_OPTIX_SDK "Use Optix system folder if specified" OFF)
option(BUILD_PYTHON_WRAPPER "Build Python wrapper" OFF)
option(BUILD_MATLAB_WRAPPER "Build MATLAB wrapper" OFF)
option(ENABLE_SIMD "Enable SIMD instructions" ON)

# TIP: if different options need to be specified for different targets of the build, use INTERFACE targets

# COMPILER FLAGS
# Set compile flags based on configuration type
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -Wextra -fmax-errors=3 -Og -g -fsanitize=address,undefined")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -fmax-errors=3 -Og -g -fsanitize=address,undefined")

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fmax-errors=1 -Wall -Wextra -fsanitize=address,undefined -O3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fmax-errors=1 -Wall -Wextra -fsanitize=address,undefined -O3")

if (ENABLE_OMP)
    find_package(OpenMP REQUIRED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

add_library(sanitizer_target_settings INTERFACE)
target_compile_options(sanitizer_target_settings INTERFACE
    $<$<CONFIG:DEBUG>: -fsanitize=address,undefined>
    $<$<CONFIG:RELEASE>: -fsanitize=address,undefined>
)

# Guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

message(STATUS "==============================================================")
message(STATUS "====================  Dependencies ===========================")

# LIBRARY DEPENDENCIES DEFINITION
find_package( Eigen3 3.4 REQUIRED )
find_package( Doxygen )
find_package( OpenGL )
find_package( CUDAToolkit 12.0) 

# Set the dependencies for the automated export
set(TARGET_DEPS Eigen3 OpenGL CUDAToolkit)
message(STATUS "Target dependencies: ${TARGET_DEPS}")

#find_package( GTSAM 4.0 REQUIRED )
#find_package( GTSAM_UNSTABLE REQUIRED)
#find_package( sopengv )
#find_package( OpenCV core imgproc highgui features2d calib3d video)
#find_package( DBoW2 )
#find_package( Pangolin )
#find_package( jsoncpp )
#find_package( rapid_yaml) ??
# torch
# tensort

# Define target to specify includes and compile defs
add_library(cuda_compile_settings INTERFACE)

# Handle CUDA and setup # TODO (PC) Move this code to a dedicated cmake file (HandleCUDA.cmake)
if (CUDAToolkit_FOUND)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")

    set(cuda_arch "")
    set(sm_version "")
    # Call the function to detect CUDA architecture
    detect_cuda_arch(cuda_arch sm_version)
    
    # Include directories
    target_include_directories(cuda_compile_settings INTERFACE ${CUDAToolkit_INCLUDE_DIRS})

    set_property(TARGET cuda_compile_settings 
                 PROPERTY CUDA_ARCHITECTURES ${sm_version})

    target_compile_definitions(cuda_compile_settings INTERFACE __CUDA_ENABLED__=1)
    
    # Select CUDA libraries to link
    set(CUDA_LINK_LIBRARIES CUDA::cublas CUDA::cudart CUDA::curand CUDA::cuda_driver)

    set_target_properties(cuda_compile_settings PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # Enable separable compilation

    # Check if empty
    if (CUDA_LINK_LIBRARIES STREQUAL "")
      message(FATAL_ERROR "CUDA libraries not found.")
    else()
      message(STATUS "CUDA interface target linked against CUDA libraries: ${CUDA_LINK_LIBRARIES}")
    endif()

    target_link_libraries(cuda_compile_settings INTERFACE ${CUDA_LINK_LIBRARIES})  # CUDA

endif()

# OPTIX
set(OPTIX_FOUND OFF) 

if (ENABLE_OPTIX) 
# TODO (PC) Move this code to a dedicated cmake file (HandleOptix.cmake)

  target_compile_definitions(cuda_compile_settings INTERFACE __OPTIX_ENABLED__=1)

  # Specify Optix installation directory
  set(OptiX_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64)

  message(STATUS "Using OptiX installation directory: ${OptiX_INSTALL_DIR}")

  if(${USE_OPTIX_SDK})
      if (DEFINED ENV{OPTIX_HOME})
        set(OptiX_INSTALL_DIR $ENV{OPTIX_HOME})
      else()
      message(FATAL_ERROR "USE_SYS_OPTIX_SDK is enabled but OPTIX_HOME variable is not defined.")
      endif()
  endif()

  set(OPTIX_FOUND ON)

  # Define Optix includes # DEVNOTE these are specific to spectral rt
  set(OPTIX_INCLUDE_DIR ${OptiX_INSTALL_DIR}/include)
  list(APPEND OPTIX_INCLUDE_DIR ${OptiX_INSTALL_DIR}/SDK/) # Add headers in SDK/sutil
  list(APPEND OPTIX_INCLUDE_DIR ${OptiX_INSTALL_DIR}/SDK/sutil) # Add headers in SDK/sutil
  list(APPEND OPTIX_INCLUDE_DIR ${OptiX_INSTALL_DIR}/SDK/support)
  list(APPEND OPTIX_INCLUDE_DIR ${OptiX_INSTALL_DIR}/SDK/optixConsole)

  message(STATUS "OptiX enabled")
  message(STATUS "OPTIX_INCLUDE_DIR: ${OPTIX_INCLUDE_DIR}")
  list(APPEND INCLUDE_DIR_LIST ${OPTIX_INCLUDE_DIR})
  
  # Add Optix includes to interface target
  #target_include_directories(cuda_compile_settings INTERFACE ${OPTIX_INCLUDE_DIR})

endif()

# Add conveniency tools adapted from GTSAM
# Include CMake modules from GTSAM
find_package(GTSAMCMakeTools)

# Use macros for creating tests/timing scripts
#include(GtsamTesting)
#include(GtsamPrinting)

# Add catch2 for unit testing
find_package(Catch2 REQUIRED)

# TODO: move to dedicated cmake script
if (Catch2_FOUND)
    message(STATUS "Catch2 found at system level: ${Catch2_DIR}")
else()
  message(STATUS "Catch2 not found. Fetching it...")
  include(FetchContent)

  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.4.0 # or a later release (TO UPDATE AS NEEDED)
  )

  FetchContent_MakeAvailable(Catch2)
  message(STATUS "Catch2 library fetched successfully.")
endif()

# Create the installation target list 
set(installable_targets)

# TIP: To set target compilation options (or any other thing) that must be specific for a target or propagated, use the following cmake pattern (example). Requires cmake 3.15 or later. Here flag BUILD_INTERFACE regulates whether the compile flags are added to the target_compile_options. If true, they are added.
#add_library(library_compile_flags INTERFACE)
#target_compile_options(library_compile_flags INTERFACE 
#"$<${gcc_like_cxx}: $<BUILD_INTERFACE: -Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused>>")
#target_link_libraries(library_target PUBLIC any_other_target library_compile_flags)

# INCLUDE DIRECTORIES
include_directories( ${OpenCV_INCLUDE_DIRS} ) 
include_directories( ${GTSAM_INCLUDE_DIRS}) 

# Define include target interface
add_library(${project_name}_include INTERFACE)

# Define include directories for the include library
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

# Include CTest and Catch2
include(CTest)
include(Catch)

# Add other subdirectories if needed
#add_subdirectory( other_folder )

# Add library
add_subdirectory( src )

# Add examples to build
add_subdirectory( examples )

# Add tests to build
add_subdirectory( tests )

# Add documentation
set(DOC_SUBDIRS "${${project_name}_INCLUDE_DIRS}")
if (DOXYGEN_FOUND)
    add_subdirectory( "doc" )
else()
    message(STATUS "Doxygen not found. Skipping building documentation.") 
endif()

# Print configuration variables
message(STATUS "===============================================================")
message(STATUS "================  Configuration Options  ======================")
message(STATUS "LIBRARY NAME                              : ${project_name}")
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

  if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "  C compilation flags                     : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")
    message(STATUS "  C++ compilation flags                   : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
  else()
    message(STATUS "  C compilation flags                     : ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
    message(STATUS "  C++ compilation flags                   : ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
  endif()

endif()

# WRAPPERS CONFIGURATION
# Handle wrappers
# TODO update this after gtsam_spaceNav fix
if(BUILD_PYTHON_WRAPPER OR BUILD_MATLAB_WRAPPER)

  if (NOT GTSAM_FOUND)
    find_package( GTSAM 4.0 REQUIRED ) # GTSAM types are required for wrappers
  endif() 

  # Set the default Python version to use for wrapping
  set(WRAP_PYTHON_VERSION ${PROJECT_PYTHON_VERSION}
    CACHE STRING "The Python version to use for wrapping")

  # Look for gtwrap
  find_package(gtwrap QUIET)

  # Check if wrap subdirectory exists, else fetch it from github as submodule
  if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/wrap") 

      message(STATUS "Wrap subdirectory not found. Attempting to fetch it from GitHub...")
      execute_process(COMMAND git submodule add "git@github.com:PeterCalifano/wrap.git"
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
      execute_process(COMMAND git submodule update --init --recursive
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

      if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/wrap")
      # Throw an error if the submodule was not fetched successfully
          message(FATAL_ERROR "Failed to fetch wrap subdirectory from GitHub!")
      else()
          message(STATUS "Wrap subdirectory fetched successfully.")
      endif()
      
  else()
      message(STATUS "GTwrap package OR Wrap subdirectory found. Proceeding to build wrappers...")
  endif()

  # Set the include directory for matlab.h
  set(GTWRAP_INCLUDE_NAME "wrap")

  # DEFINE interface files for wrapper
  set(SEARCH_DIR_WRAP 
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
  
  #message(STATUS "Searching for wrapper interface files in: ${SEARCH_DIR_WRAP}")
  #file(GLOB WRAPPER_INTERFACE_FILES "${SEARCH_DIR_WRAP}" "*.i") # Not working for now

  # Template: set(WRAPPER_INTERFACE_FILES "${SEARCH_DIR_WRAP}/<name>.i") # Add the interface files here
  message(STATUS "Found wrapper interface files: ${WRAPPER_INTERFACE_FILES}")

  if (NOT WRAPPER_INTERFACE_FILES)
    message(FATAL_ERROR "WRAPPER_INTERFACE_FILES list not defined. Please check the cmake configuration.")
  else()
    # Check if list has exactly one element that is empty
    list(LENGTH WRAPPER_INTERFACE_FILES WRAPPER_INTERFACE_FILES_LEN)
    if (WRAPPER_INTERFACE_FILES_LEN EQUAL 1)
      list(GET WRAPPER_INTERFACE_FILES 0 _first_interface)
      if (_first_interface STREQUAL "")
        message(FATAL_ERROR "No interface files found for wrapping. Please check the search directory or provide files.")
      endif()
    endif()
  endif()

  # Copy matlab.h to the correct folder.
  configure_file(${PROJECT_SOURCE_DIR}/wrap/matlab.h
              ${PROJECT_BINARY_DIR}/wrap/matlab.h COPYONLY)

  if (NOT gtwrap_FOUND) # Build GTwrap within project
    message(STATUS "GTwrap package NOT found. Attempting to build it as ExternalProject...")
  
    include(ExternalProject)

    ExternalProject_Add(
      wrap_project
      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/wrap
      CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${WRAP_INSTALL_DIR}
      BUILD_ALWAYS OFF  # Optional: rebuild every time (remove for performance)
    )

    # Add the install directory to CMake's module path
    list(APPEND CMAKE_PREFIX_PATH ${WRAP_INSTALL_DIR})

    message(STATUS "GTwrap package correctly built and added to CMAKE_PREFIX_PATH.")
      
    add_subdirectory(wrap)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/wrap/cmake")

    # Python toolbox
    if(BUILD_PYTHON_WRAPPER)
        # Check if python folder exists
        #if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/python")
        #    message(FATAL_ERROR "Python wrapper requested but python folder not found.")
        #endif()
        #add_subdirectory(python)
    endif()

    # Matlab toolbox
    if(BUILD_MATLAB_WRAPPER)
        # Check if matlab folder exists
        #if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/matlab")
        #    message(FATAL_ERROR "MATLAB wrapper requested but matlab folder not found.")
        #endif()
        #add_subdirectory(matlab)
    endif()
  endif()

  # Python toolbox
  if(BUILD_PYTHON_WRAPPER)
  # TODO: Add the python wrapper configuration here

    if ( NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/python" )
      # Make directory
      file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python")
    endif()
  endif()

  # Matlab toolbox
  if(BUILD_MATLAB_WRAPPER)
    include( MatlabWrap ) 

    # Configure MATLAB paths
    message (STATUS "Including MATLAB directories...")
    find_package (Matlab REQUIRED)
    set(MATLAB_MEX_INCLUDE "${Matlab_ROOT_DIR}/extern/include")
    set(SOURCE_DIR "mexCodes")
    set(MATLAB_DEFAULT_RELEASE "R2023b")

    message(STATUS "MATLAB_MEX_INCLUDE directory: ${MATLAB_MEX_INCLUDE}")
    message(STATUS "Matlab_MEX_LIBRARY directory: ${Matlab_MEX_LIBRARY}")
    message(STATUS "Matlab_MX_LIBRARY directory: ${Matlab_MX_LIBRARY}")

    # Set the include directories
    include_directories(${Matlab_INCLUDE_DIRS})
    include_directories(${MATLAB_MEX_INCLUDE})

    if ( NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/matlab" )
      # Make directory
      file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/matlab")
    endif()

    # Set up installation paths (#TODO Understand these options better)
    set(WRAP_MEX_BUILD_STATIC_MODULE OFF) # Defines if the mex module is built as a static module
    #set(WRAP_BUILD_MEX_BINARY_FLAGS ${GTSAM_BUILD_MEX_BINARY_FLAGS})
    set(WRAP_TOOLBOX_INSTALL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/matlab") # Defines the installation path for the MATLAB wrapper files
    #set(WRAP_CUSTOM_MATLAB_PATH ${GTSAM_CUSTOM_MATLAB_PATH})
    set(WRAP_BUILD_TYPE_POSTFIXES OFF) # Determines if post build type postfixes are added to the mex files 

    # Add the MATLAB wrapper
    wrap_and_install_library("${WRAPPER_INTERFACE_FILES}" "${project_name}" "" "" "" "" OFF)
  endif()
  
endif()

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


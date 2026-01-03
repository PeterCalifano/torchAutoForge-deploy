# CMake configuration to handle Python and MATLAB wrapper configuration
include_guard(GLOBAL)

include(ExternalProject)
function(set_python_target_properties PYTHON_TARGET OUTPUT_NAME OUTPUT_DIRECTORY)
  set_target_properties(${PYTHON_TARGET} PROPERTIES
      INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
      INSTALL_RPATH_USE_LINK_PATH TRUE
      OUTPUT_NAME "${OUTPUT_NAME}"
      LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIRECTORY}"
      DEBUG_POSTFIX "" # Otherwise you will have a wrong name
      RELWITHDEBINFO_POSTFIX "" # Otherwise you will have a wrong name
      TIMING_POSTFIX "" # Otherwise you will have a wrong name
      PROFILING_POSTFIX "" # Otherwise you will have a wrong name
  )
endfunction()

### Python and MATLAB wrapper configuration using gtwrap
# Function for common wrapper configuration
function(configure_gtwrappers_common)
  if (NOT GTSAM_FOUND)
    message(STATUS "Attempt to find GTSAM package for wrapper configuration. This is required to build python and MATLAB bindings using gtwrap.")
    find_package(GTSAM 4.0 REQUIRED) # GTSAM types are required for wrappers
  endif()

  # Set the default Python version to use for wrapping
  set(WRAP_PYTHON_VERSION ${PROJECT_PYTHON_VERSION}
    CACHE STRING "The Python version to use for wrapping")

  # Look for gtwrap
  find_package(gtwrap QUIET)

  # Check if wrap subdirectory exists, else fetch it from github as submodule
  if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/wrap")
      message(STATUS "Wrap subdirectory not found. Attempting to fetch it from GitHub...")
      execute_process(COMMAND git submodule add "git@github.com:PeterCalifano/wrap.git"
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
      execute_process(COMMAND git submodule update --init --recursive
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

      if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/wrap")
      # Throw an error if the submodule was not fetched successfully
          message(FATAL_ERROR "Failed to fetch wrap subdirectory from GitHub!")
      else()
          message(STATUS "Wrap subdirectory fetched successfully.")
      endif()
  else()
      message(STATUS "GTwrap package OR Wrap subdirectory found. Proceeding to build wrappers...")
  endif()

  # Set the include directory for matlab.h
  set(GTWRAP_INCLUDE_NAME ${PROJECT_SOURCE_DIR}/lib/wrap/include)
  include_directories(${GTWRAP_INCLUDE_NAME})

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
  configure_file(${PROJECT_SOURCE_DIR}/lib/wrap/matlab.h
              ${PROJECT_BINARY_DIR}/wrap/matlab.h COPYONLY)

  if (NOT gtwrap_FOUND) # Build GTwrap within project
    message(STATUS "GTwrap package NOT found. Attempting to build it as ExternalProject...")

    ExternalProject_Add(
      wrap_project
      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/wrap
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
      if ("${PYTHON_EXECUTABLE}" STREQUAL "python3")
          message(WARNING "PYTHON_EXECUTABLE is set to python3. This will likely fail at install time if no venv is activated. Please specify the full path to the python executable if conda.")
      endif()
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
endfunction()

# TODO review code to make it general / test it
# Python wrapper configuration using gtwrap
function(configure_python_gtwrapper)
  ## Install Python wrap
  message(STATUS "Configuring Python wrap...")

  set(PROJECT_PYTHON_SOURCE_DIR ${PROJECT_SOURCE_DIR}/python)
  set(PROJECT_PYTHON_BUILD_DIRECTORY ${PROJECT_BINARY_DIR}/python)

  # Generate setup.py from template # TODO update, new version using pyproject toml
  file(READ "${PROJECT_SOURCE_DIR}/README.md" README_CONTENTS)
  configure_file(${PROJECT_PYTHON_SOURCE_DIR}/setup.py.in
                ${PROJECT_PYTHON_BUILD_DIRECTORY}/setup.py)

  set(interface_files ${PROJECT_SOURCE_DIR}/cpp/${PROJECT_NAME}.h)

  include(PybindWrap)

  # This is required to avoid an error in modern pybind11 cmake scripts:
  if(POLICY CMP0057)
      cmake_policy(SET CMP0057 NEW)
  endif()

  # Use bundled pybind11 version (in wrap)
  add_subdirectory(${PROJECT_SOURCE_DIR}/wrap/pybind11 pybind11)

  # Set the wrapping script variable
  set(PYBIND_WRAP_SCRIPT "${PROJECT_SOURCE_DIR}/wrap/scripts/pybind_wrap.py")

  set(ENABLE_BOOST_SERIALIZATION OFF)
  set(PROJECT_PYTHON_TARGET_NAME ${project_name}_py)

  pybind_wrap(${PROJECT_PYTHON_TARGET_NAME} # target
            "${WRAPPER_INTERFACE_FILES}" # list of interface header files
            "${PROJECT_PYTHON_TARGET_NAME}.cpp" # the generated cpp
            "${PROJECT_NAME}" # module_name
            "gtsam" # top namespace in the cpp file e.g. gtsam
            "${ignore}" # ignore classes
            ${PROJECT_BINARY_DIR}/${PROJECT_NAME}.tpl # the wrapping template file
            ${PROJECT_NAME} # libs
            "gtsam; gtsam_spaceNav" # dependencies
            ${ENABLE_BOOST_SERIALIZATION} # use boost serialization
            )

  set_python_target_properties(${PROJECT_PYTHON_TARGET_NAME} ${project_name} "${PROJECT_PYTHON_BUILD_DIRECTORY}/${project_name}")

  # Copy all python files to build folder.
  copy_directory("${CMAKE_CURRENT_SOURCE_DIR}/${project_name}"
          "${PROJECT_PYTHON_BUILD_DIRECTORY}/${project_name}")

  set(PROJECT_PYTHON_DEPS ${project_name} gtsam)

  # TODO move this to a cmake scripts file and add custom targets to install python package. Also make sure path to python exists and available and NOT python system-wide.
  add_custom_target(
          python-stubs
          COMMAND
            ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${PROJECT_PYTHON_BUILD_DIRECTORY}/$ENV{PYTHONPATH}"
          ${PYTHON_EXECUTABLE} -m pybind11_stubgen -o . --ignore ${ignore} #--ignore-all-errors
          DEPENDS ${PROJECT_PYTHON_DEPS} ${PROJECT_PYTHON_TARGET_NAME}
          WORKING_DIRECTORY "${PROJECT_PYTHON_BUILD_DIRECTORY}/"
  )

  add_custom_target(python-install
      COMMAND ${PYTHON_EXECUTABLE} -c "import sys, subprocess; cmd = [sys.executable, '-m', 'pip', 'install']; has_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix); cmd.append('--user' if not has_venv else ''); cmd.append('.'); subprocess.check_call([c for c in cmd if c])"
      DEPENDS ${PROJECT_PYTHON_DEPS}
      WORKING_DIRECTORY ${PROJECT_PYTHON_BUILD_DIRECTORY}
      VERBATIM)
endfunction()

# MATLAB wrapper configuration using gtwrap
function(configure_matlab_gtwrapper)
  message(STATUS "Configuring MATLAB wrap...")
  include(MatlabWrap)

  # Configure MATLAB paths
  message (STATUS "Including MATLAB directories...")
  find_package(Matlab REQUIRED)
  set(MATLAB_MEX_INCLUDE "${Matlab_ROOT_DIR}/extern/include")

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

  # Set up installation paths (#TODO need to review these options)
  set(WRAP_MEX_BUILD_STATIC_MODULE OFF) # Defines if the mex module is built as a static module
  #set(WRAP_BUILD_MEX_BINARY_FLAGS ${GTSAM_BUILD_MEX_BINARY_FLAGS})
  set(WRAP_TOOLBOX_INSTALL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/matlab") # Defines the installation path for the MATLAB wrapper files
  #set(WRAP_CUSTOM_MATLAB_PATH ${GTSAM_CUSTOM_MATLAB_PATH})
  set(WRAP_BUILD_TYPE_POSTFIXES OFF) # Determines if post build type postfixes are added to the mex files

  # Add the MATLAB wrapper
  wrap_and_install_library("${WRAPPER_INTERFACE_FILES}" "${project_name}" "" "" "" "" OFF)
endfunction()

function(handle_gtwrappers)
  if(NOT BUILD_PYTHON_WRAPPER AND NOT BUILD_MATLAB_WRAPPER)
    return()
  endif()

  configure_gtwrappers_common()
  if(BUILD_PYTHON_WRAPPER)
    configure_python_gtwrapper()
  endif()

  if(BUILD_MATLAB_WRAPPER)
    configure_matlab_gtwrapper()
  endif()
endfunction()


### Python wrapper configuration using pybind11 directly
# TODO
function(handle_pybind11_wrapper)


endfunction()
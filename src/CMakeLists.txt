set(lib_name ${LIB_TARGET_NAME})

# Add config.h file to build
configure_file(config.h.in config.h)

# Add src files to library
set(SRC_PATTERN "*.cpp" "*.cu")
file(GLOB srcLibFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${SRC_PATTERN})
file(GLOB srcCudaFilesToPTX "*.ptx.cu")
list(FILTER srcLibFiles EXCLUDE REGEX "\\.ptx\\.cu$")

# Get current folder name
get_filename_component(CURRENT_FOLDER_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
message(STATUS "Library src files found in ${CURRENT_FOLDER_NAME}: ${srcLibFiles}")

# Add subdirectories with src files
message(STATUS "Adding module: utils")
add_subdirectory(utils)

message(STATUS "Adding module: inference")
add_subdirectory(inference)

message(STATUS "Adding module: template_src_kernels")
add_subdirectory(template_src_kernels)

# Install headers from this directory
file(GLOB installable_headers "*.h") # Discover and install all header files
install(FILES ${installable_headers} 
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${project_name}
      ) # Install headers

# Delete installable_headers
unset(installable_headers)

# Select library type (header only or compiled) based on presence of source files.
list(LENGTH srcLibFiles src_lib_files_count)
if (src_lib_files_count EQUAL 0)
  set(lib_is_header_only TRUE)
  set(lib_link_scope INTERFACE)
  set(lib_target_type "INTERFACE")
  add_library(${lib_name} INTERFACE)
else()
  set(lib_is_header_only FALSE)
  set(lib_link_scope PUBLIC)
  if (BUILD_SHARED_LIBS)
    set(lib_target_type "SHARED")
  else()
    set(lib_target_type "STATIC")
  endif()

  add_library(${lib_name} ${lib_target_type}
              ${srcLibFiles})
  set_target_properties(${lib_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

set_target_properties(${lib_name} PROPERTIES EXPORT_NAME ${project_name})

message(STATUS "Configured ${lib_name} as a ${lib_target_type} library target")
set(PROJECT_LIBRARY_KIND "${lib_target_type}" PARENT_SCOPE)

if(NOT TARGET ${project_name}::${project_name})
  add_library(${project_name}::${project_name} ALIAS ${lib_name})
endif()

# Link sanitizer target only for in-tree builds to avoid export/install deps.
if (NOT lib_is_header_only AND ENABLE_SANITIZERS AND SANITIZER_TARGET AND TARGET ${SANITIZER_TARGET})
  target_link_libraries(${lib_name} PRIVATE ${SANITIZER_TARGET})
endif()

# If need to add other files after add_library, use:
#target_sources(${lib_name} SHARED ${srcLibFiles})

if(ENABLE_CUDA AND srcCudaFilesToPTX)
  if(lib_is_header_only)
    message(FATAL_ERROR
      "PTX embedding requires a compiled ${project_name} library target.")
  endif()

  get_target_property(
    project_interface_INCLUDE_DIRS
    ${project_name}_include
    INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT project_interface_INCLUDE_DIRS)
    set(project_interface_INCLUDE_DIRS "")
  endif()

  set(PTX_INCLUDE_DIRS
      ${CUDAToolkit_INCLUDE_DIRS}
      ${project_interface_INCLUDE_DIRS})
  cuda_compile_and_embed(
    autoforge_deploy_embedded_ptx
    "${sm_version}"
    "${PTX_INCLUDE_DIRS}"
    ${srcCudaFilesToPTX})
  target_sources(${lib_name} PRIVATE ${autoforge_deploy_embedded_ptx})
  set(AUTOFORGE_DEPLOY_EMBEDDED_PTX_SYMBOLS
      "${autoforge_deploy_embedded_ptx_SYMBOLS}"
      CACHE INTERNAL "Symbols generated for embedded PTAFDeploy PTX")
endif()

### Link targets
# Project include directories
target_link_libraries(${lib_name} ${lib_link_scope} ${project_name}_include ${LIB_COMPILE_TARGET})

if (ENABLE_CUDA)
  target_link_libraries(${lib_name} ${lib_link_scope} ${CUDA_COMPILE_TARGET})
endif()

target_link_libraries(${lib_name} ${lib_link_scope} ${TENSORRT_COMPILE_TARGET})

if (ENABLE_OPENGL)
  target_link_libraries(${lib_name} ${lib_link_scope} ${OPENGL_COMPILE_TARGET})
endif()

# Link ORT target interface
target_link_libraries(${lib_name} ${lib_link_scope} ort_target_interface)

# Link profiling interface: gperftools always when found, compile flags only with ENABLE_PROFILING.
# Default is PRIVATE to keep profiling a build-time concern. Set
# PROFILING_PROPAGATES_TO_CONSUMERS=ON to make profiling usage requirements PUBLIC.
if (NOT lib_is_header_only AND TARGET ${PROFILING_COMPILE_TARGET})
  if (PROFILING_PROPAGATES_TO_CONSUMERS)
    target_link_libraries(${lib_name} PUBLIC ${PROFILING_COMPILE_TARGET})
  else()
    target_link_libraries(${lib_name} PRIVATE ${PROFILING_COMPILE_TARGET})
  endif()
endif()

#target_include_directories(${lib_name} INTERFACE ${interface_INCLUDE_DIRS} ) 
# TIP: INTERFACE ensures that the consumer target does not need to include manually the include directories the library target depends on, because they are automagically propagated to the target, even if the target does not need thme. The same happens with PUBLIC, but this time the includes are used by the target library itself.

target_link_libraries(${lib_name} ${lib_link_scope}
                                  Eigen3::Eigen)

### INCLUDE DIRECTORIES
if(DEFINED onnxruntime_INCLUDE_DIRS AND NOT "${onnxruntime_INCLUDE_DIRS}" STREQUAL "")
  target_include_directories(${lib_name} PRIVATE ${onnxruntime_INCLUDE_DIRS})
endif()


# Add target to installable list 
set(installable_targets ${installable_targets} 
                        ${lib_name} 
                        ${project_name}_include
                        ${LIB_COMPILE_TARGET}
                        ort_target_interface
                        ${TENSORRT_COMPILE_TARGET})

if (TARGET ${PROFILING_COMPILE_TARGET})
  list(APPEND installable_targets ${PROFILING_COMPILE_TARGET})
endif()

if (ENABLE_CUDA)
  list(APPEND installable_targets ${CUDA_COMPILE_TARGET})
endif()

if (ENABLE_OPENGL)
  list(APPEND installable_targets ${OPENGL_COMPILE_TARGET})
endif()

message(NOTICE "Added target to installable list: ${installable_targets}")

# Install global_includes.h file
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/global_includes.h" 
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${project_name}) 

# Install config.h file
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/config.h" 
       DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${project_name}) 

# INSTALL CONFIGURATION
install(
	TARGETS ${installable_targets}
	EXPORT ${project_name} # Export target to be used by other projects
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  ) # Install targets

# Configuration for CPack and consumer targets
# See https://cmake.org/cmake/help/latest/guide/tutorial/Adding%20Export%20Configuration.html
include(CMakePackageConfigHelpers)

install(EXPORT ${project_name}
    FILE "${project_name}Target.cmake"
    NAMESPACE ${project_name}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${project_name}
)

write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/${project_name}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${project_name}Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${project_name}Config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${project_name}
)

set(_package_module_build_dir "${PROJECT_BINARY_DIR}/modules")
file(MAKE_DIRECTORY "${_package_module_build_dir}")
configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/FindTensorRT.cmake"
    "${_package_module_build_dir}/FindTensorRT.cmake"
    COPYONLY
)

install(FILES
    "${PROJECT_BINARY_DIR}/${project_name}Config.cmake"
    "${PROJECT_BINARY_DIR}/${project_name}ConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${project_name}
)

install(FILES
    "${PROJECT_SOURCE_DIR}/cmake/FindTensorRT.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${project_name}/modules
)

# Add target export to allow usage of build by other consumer projects
export(EXPORT ${project_name}
  FILE "${PROJECT_BINARY_DIR}/${project_name}Target.cmake"
  NAMESPACE ${project_name}::
)

# Add and build programs subdirectory
if (BUILD_AS_MAIN_PROJECT AND ${${BUILD_PROGRAMS_OPTION_NAME}})
  message(STATUS "Building binary programs")
  add_subdirectory( programs )
endif()

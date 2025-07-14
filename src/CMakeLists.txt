set(lib_name ${project_name}) # Default name for the library is project_name

# Add config.h file to build
configure_file(config.h.in config.h)

# Add src files to library
set(SRC_PATTERN "*.cpp; *.cu")
file(GLOB srcLibFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${SRC_PATTERN})

message(STATUS "Library src files found: ${srcLibFiles}")

if (ENABLE_OPTIX)
  # Find all kernels source codes
  file(GLOB srcCudaFilesToPTX RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "kernels/*.cu")
endif()

# Add subdirectories with src files
message(STATUS "Adding module: template_src")
add_subdirectory(template_src)

message(STATUS "Adding module: .programs")
add_subdirectory(.programs)

# Exclude EXCLUDED_LIST from the list of src files
set(EXCLUDED_LIST "")
list(REMOVE_ITEM srcLibFiles ${EXCLUDED_LIST})

# Install headers from this directory
file(GLOB installable_headers "*.h") # Discover and install all header files
install(FILES ${installable_headers} 
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${project_name}
      ) # Install headers

# Delete installable_headers
unset(installable_headers)

add_library(${lib_name} SHARED
            ${srcLibFiles})

# If need to add other files after add_library, use:
#target_sources(${lib_name} SHARED ${srcLibFiles})

if ( ENABLE_OPTIX )
  message("Library .cu kernels files to compile as PTX: ${srcCudaFilesToPTX}")

  # TODO (PC) - Add CUDA to PTX code

  # Build CUDA to PTX code (conveniency tool by Ingo Wald)
  cuda_compile_and_embed(embedded_ptx_code ${srcCudaFilesToPTX}) 
  # NOTE: Macro above sets up the build of CUDA kernels -> PTX code from which OptiX module is built.
  # Build occurs before compilation and linking of the library (or executables).
  # Variable embedded_ptx_code must be extern "C" const char withing the cpp source code.
    
  # Add the embedded PTX code to the library
  target_sources(${lib_name} PUBLIC ${embedded_ptx_code})    
endif()

# Link targets
target_include_directories(${lib_name} PUBLIC 
                          ${OpenCV_INCLUDE_DIRS} 
                          )

# Link interface target for CUDA
target_link_libraries(${lib_name} PUBLIC cuda_compile_settings ${project_name}_include)

#target_include_directories(${lib_name} INTERFACE ${interface_INCLUDE_DIRS} ) 
# TIP: INTERFACE ensures that the consumer target does not need to include manually the include directories the library target depends on, because they are automagically propagated to the target, even if the target does not need thme. The same happens with PUBLIC, but this time the includes are used by the target library itself.

target_link_libraries(${lib_name} PUBLIC 
                                  ${OpenCV_LIBS} 
                                  Eigen3::Eigen 
                                  #gdt
                                  #glfw 
                                  ${OPENGL_LIBRARIES}
                                  ${optix_LIBRARY})

# Add target to installable list 
set(installable_targets ${installable_targets} ${lib_name} cuda_compile_settings ${project_name}_include)

message(NOTICE "Added target to installable list: ${installable_targets}")

# Install global_includes.h file
#install(FILES "${CMAKE_CURRENT_BINARY_DIR}/global_includes.h" 
#        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${project_name}) 

# Install config.h file
##install(FILES "${CMAKE_CURRENT_BINARY_DIR}/config.h" 
#        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${project_name}) 

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
    DESTINATION lib/cmake/${project_name}
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${project_name}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${project_name}Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${project_name}Config.cmake"
    INSTALL_DESTINATION lib/cmake/${project_name}
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${project_name}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${project_name}ConfigVersion.cmake"
    DESTINATION lib/cmake/${project_name}
)

# Add target export to allow usage of build by other consumer projects
export(EXPORT ${lib_name}
  FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/${project_name}Target.cmake"
)
# Add and build programs subdirectory
#message(STATUS "Building binary programs")
#add_subdirectory( .programs )

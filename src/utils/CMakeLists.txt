file(GLOB installable_headers "*.h" "*.cuh")
file(RELATIVE_PATH relative_path_from_root
     "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
string(REPLACE "src/" "" relative_path_without_src
       "${relative_path_from_root}")

set(SRC_PATTERN "*.cpp" "*.cu")
file(GLOB srcLibFiles_local ${SRC_PATTERN})
file(GLOB srcCudaFilesToPTX_local "*.ptx.cu")
list(FILTER srcLibFiles_local EXCLUDE REGEX "\\.ptx\\.cu$")

add_subdirectory(logging)
add_subdirectory(parsing)

set(srcLibFiles
    ${srcLibFiles} ${srcLibFiles_local}
    PARENT_SCOPE)
set(srcCudaFilesToPTX
    ${srcCudaFilesToPTX} ${srcCudaFilesToPTX_local}
    PARENT_SCOPE)

install(
  FILES ${installable_headers}
  DESTINATION
    ${CMAKE_INSTALL_INCLUDEDIR}/${project_name}/${relative_path_without_src})

# Optional utilities own separate targets; never append their sources to srcLibFiles.
if(autoforge_deploy_ENABLE_IMAGE_SUPPORT)
  add_subdirectory(images)
endif()
if(autoforge_deploy_ENABLE_INFERENCE_OUTPUT)
  add_subdirectory(inference_output)
endif()

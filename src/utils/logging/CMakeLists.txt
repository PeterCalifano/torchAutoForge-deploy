set(installable_headers "${CMAKE_CURRENT_SOURCE_DIR}/CLogger.h")
file(RELATIVE_PATH relative_path_from_root
     "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
string(REPLACE "src/" "" relative_path_without_src
       "${relative_path_from_root}")

set(srcLibFiles_local "${CMAKE_CURRENT_SOURCE_DIR}/CLogger.cpp")
set(srcCudaFilesToPTX_local "")

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

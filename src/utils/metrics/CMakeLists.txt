# Dependency-free metrics remain available to all core consumers.
set(srcLibFiles ${srcLibFiles} "${CMAKE_CURRENT_SOURCE_DIR}/metrics.cpp" PARENT_SCOPE)
install(FILES metrics.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${project_name}/utils/metrics)

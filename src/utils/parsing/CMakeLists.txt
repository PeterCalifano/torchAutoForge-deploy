# Keep parsing in the core library while installing its responsibility-specific path.
set(srcLibFiles ${srcLibFiles} "${CMAKE_CURRENT_SOURCE_DIR}/value_parsing.cpp" PARENT_SCOPE)
install(FILES value_parsing.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${project_name}/utils/parsing)

# Export this optional component independently from the core target set.
add_library(autoforge_deploy_images images.cpp)
add_library(autoforge_deploy::images ALIAS autoforge_deploy_images)
set_target_properties(autoforge_deploy_images PROPERTIES
  EXPORT_NAME images POSITION_INDEPENDENT_CODE ON)
target_compile_features(autoforge_deploy_images PUBLIC cxx_std_20)
target_include_directories(autoforge_deploy_images PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${project_name}>)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
target_link_libraries(autoforge_deploy_images PUBLIC ${OpenCV_LIBS})
install(TARGETS autoforge_deploy_images EXPORT autoforge_deploy_images
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES images.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${project_name}/utils/images)
install(EXPORT autoforge_deploy_images FILE autoforge_deploy_imagesTargets.cmake
  NAMESPACE autoforge_deploy:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${project_name})
export(EXPORT autoforge_deploy_images
  FILE "${PROJECT_BINARY_DIR}/autoforge_deploy_imagesTargets.cmake"
  NAMESPACE autoforge_deploy::)

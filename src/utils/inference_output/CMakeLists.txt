# Export this optional component independently from the core target set.
add_library(autoforge_deploy_inference_output inference_output.cpp report_reader.cpp)
add_library(autoforge_deploy::inference_output ALIAS autoforge_deploy_inference_output)
set_target_properties(autoforge_deploy_inference_output PROPERTIES
  EXPORT_NAME inference_output POSITION_INDEPENDENT_CODE ON)
target_compile_features(autoforge_deploy_inference_output PUBLIC cxx_std_20)
target_include_directories(autoforge_deploy_inference_output PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${project_name}>)
target_link_libraries(autoforge_deploy_inference_output PUBLIC autoforge_deploy::autoforge_deploy)
target_include_directories(autoforge_deploy_inference_output SYSTEM PRIVATE
  "${PROJECT_SOURCE_DIR}/lib/header_only")
install(FILES "${PROJECT_SOURCE_DIR}/lib/header_only/rapidjson/license.txt"
              "${PROJECT_SOURCE_DIR}/lib/header_only/rapidjson/PROVENANCE.md"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/${project_name}/licenses/rapidjson)
install(TARGETS autoforge_deploy_inference_output EXPORT autoforge_deploy_inference_output
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES inference_output.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${project_name}/utils/inference_output)
install(EXPORT autoforge_deploy_inference_output FILE autoforge_deploy_inference_outputTargets.cmake
  NAMESPACE autoforge_deploy:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${project_name})
export(EXPORT autoforge_deploy_inference_output
  FILE "${PROJECT_BINARY_DIR}/autoforge_deploy_inference_outputTargets.cmake"
  NAMESPACE autoforge_deploy::)

# Exercise optional product components without recursive configuration or dependency tests.
foreach(component IN ITEMS images inference_output)
  if(TARGET autoforge_deploy_${component})
    add_executable(test_utils_${component} test_${component}.cpp)
    target_link_libraries(test_utils_${component} PRIVATE
      autoforge_deploy::${component} Catch2::Catch2WithMain)
    include(Catch)
    catch_discover_tests(test_utils_${component})
  endif()
endforeach()

# General filesystem operations must also work when image support is disabled.
add_executable(test_utils_filesystem test_filesystem.cpp)
target_link_libraries(test_utils_filesystem PRIVATE
  autoforge_deploy::autoforge_deploy Catch2::Catch2WithMain)
include(Catch)
catch_discover_tests(test_utils_filesystem)

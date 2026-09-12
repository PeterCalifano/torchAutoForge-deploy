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

# Basic usage of catch2 for build config:
# These tests can use the Catch2-provided main
#add_executable(tests test.cpp)
#target_link_libraries(tests PRIVATE Catch2::Catch2WithMain) 

# These tests need their own main
#add_executable(custom-main-tests test.cpp test-main.cpp)
#target_link_libraries(custom-main-tests PRIVATE Catch2::Catch2)

# Add tests to build
set(TESTS_PATTERN "test*.cpp; test*.cu")
file(GLOB srcTestFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${TESTS_PATTERN})
message("Test files found: ${srcTestFiles}")

# Exclude EXCLUDED_LIST from the list of tests
set(EXCLUDED_LIST "")
list(REMOVE_ITEM srcTestFiles ${EXCLUDED_LIST})

set(TESTS_LIST "")

# Add to build and link
foreach(testFile ${srcTestFiles})

    get_filename_component(testName ${testFile} NAME_WE)
    add_executable(${testName} ${testFile})

    list(APPEND TESTS_LIST ${testName}) 

    target_link_libraries(${testName} PUBLIC lib_name)
    target_include_directories(${testName} PRIVATE ${lib_name_INCLUDE_DIRS})

endforeach()

foreach (testTarget ${TESTS_LIST})
    catch_discover_tests(${testTarget})
endforeach()

set(TESTS_LIST ${TESTS_LIST} PARENT_SCOPE) # Export list of tests to parent scope (applied ONLY here)

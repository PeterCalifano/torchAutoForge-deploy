set(lib_name ${project_name}) # Default name for the library is project_name

# Add examples to build
set(EXAMPLES_PATTERN "example_*.cpp; example_*.cu")
file(GLOB srcExampleFiles ${EXAMPLES_PATTERN})
message("Examples files found: ${srcExampleFiles}")

# Exclude EXCLUDED_LIST from the list of examples
set(EXCLUDED_LIST "")
list(REMOVE_ITEM srcExampleFiles ${EXCLUDED_LIST})

# Add to build and link
message(STATUS "Project linked lib ${lib_name} to examples: ${srcExampleFiles}")
message(STATUS "Include dirs of project root dir: ${${lib_name}_INCLUDE_DIRS}")

foreach(exampleFile ${srcExampleFiles})

    get_filename_component(exampleName ${exampleFile} NAME_WE)
    add_executable(${exampleName} ${exampleFile})

    target_link_libraries(${exampleName} PUBLIC ${lib_name} cuda_compile_settings)
    target_include_directories(${exampleName} PRIVATE ${${lib_name}_INCLUDE_DIRS})

endforeach()
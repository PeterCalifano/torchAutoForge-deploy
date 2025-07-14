# CMAKE script containing utility functions for cmake configuration

# Function for entry exclusion in a list based on pattern matching
function(filter_files_in_list input_var output_var exclude_list)
    set(filtered_files "")
    foreach(testFile ${${input_var}})
        get_filename_component(fileName ${testFile} NAME_WE)
        list(FIND ${exclude_list} ${fileName} index)
        if(index EQUAL -1)
            list(APPEND filtered_files ${testFile})
        endif()
    endforeach()
    set(${output_var} ${filtered_files} PARENT_SCOPE)
endfunction()

# Function to add examples files to the build
function(add_examples project_lib_name excluded_list target_compile_settings)

    set(EXAMPLES_PATTERN "example_*.cpp; example_*.cu")
    file(GLOB srcExampleFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${EXAMPLES_PATTERN})
    message("Example files found: ${srcExampleFiles}")

    # Exclude files in excluded_list (filter_files_in_list is assumed to be a custom macro)
    filter_files_in_list(srcExampleFiles srcExampleFiles ${excluded_list})

    message(STATUS "Project linked lib ${project_lib_name} to examples: ${srcExampleFiles}")
    message(STATUS "Include directories of project lib: ${${project_lib_name}_INCLUDE_DIRS}")

    foreach(exampleFile ${srcExampleFiles})
        get_filename_component(exampleName ${exampleFile} NAME_WE)
        add_executable(${exampleName} ${exampleFile})
        target_link_libraries(${exampleName} PRIVATE ${project_lib_name} ${target_compile_settings})
        target_include_directories(${exampleName} PRIVATE ${${project_lib_name}_INCLUDE_DIRS})
    endforeach()

endfunction()

# Function to add test files to the build
function(add_tests project_lib_name excluded_list TESTS_LIST target_compile_settings CATCH2_TEST_PROPERTIES catch2_target)

    set(TESTS_PATTERN "test*.cpp; test*.cu")
    file(GLOB srcTestFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${TESTS_PATTERN})
    filter_files_in_list(srcTestFiles srcTestFiles ${excluded_list})
    message(STATUS "Test files found: ${srcTestFiles}")

    # Add content of this directory 
    foreach(testFile ${srcTestFiles})

        get_filename_component(testName ${testFile} NAME_WE)
        add_executable(${testName} ${testFile})

        list(APPEND ${TESTS_LIST} ${testName}) 

        target_link_libraries(${testName} PRIVATE ${project_lib_name} ${target_compile_settings} ${catch2_target})
        catch_discover_tests(${testName} PROPERTIES ${CATCH2_TEST_PROPERTIES})

    endforeach()
endfunction()

# Function to detect the compute capability using nvidia-smi
function(detect_cuda_arch cuda_arch compute_cap)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        execute_process(
            COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE gpu_compute_cap
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        # Map the compute capability to the correct architecture
        string(REPLACE "." "" sm_version "${gpu_compute_cap}")
        set(${cuda_arch} "sm_${sm_version}" PARENT_SCOPE)
        set(${compute_cap} "${sm_version}" PARENT_SCOPE)

        message(STATUS "Detected CUDA compute capability: ${gpu_compute_cap}")
        message(STATUS "Using CUDA architecture: sm_${sm_version}")

    else()
        message(WARNING "CUDA architecture detection is not supported for this platform.")
    endif()
endfunction()

# CMAKE script containing utility functions for cmake configuration
include_guard(GLOBAL)
if(COMMAND add_examples AND COMMAND add_tests AND COMMAND filter_files_in_list)
    return()
endif()

# Function for entry exclusion in a list based on pattern matching
function(filter_files_in_list input_var output_var exclude_list)
    set(filtered_files "")
    set(exclude_entries ${${exclude_list}})
    foreach(testFile ${${input_var}})
        get_filename_component(fileName ${testFile} NAME)
        get_filename_component(fileNameNoExt ${testFile} NAME_WE)

        list(FIND exclude_entries "${fileName}" index_with_ext)
        list(FIND exclude_entries "${fileNameNoExt}" index_no_ext)
        if(index_with_ext EQUAL -1 AND index_no_ext EQUAL -1)
            list(APPEND filtered_files ${testFile})
        endif()
    endforeach()
    set(${output_var} ${filtered_files} PARENT_SCOPE)
endfunction()

# Function to fetch all source files (C++, CUDA, PTX) and set related variables for a project library target
function(collect_project_source_inventory)

    # Initialize arguments
    set(oneValueArgs
        ROOT_DIR
        COMPILED_SOURCES_OUT
        PTX_SOURCES_OUT
        HAS_COMPILED_SOURCES_OUT
        HAS_PTX_SOURCES_OUT)

    # CPSI: Collect Project Source Inventory
    cmake_parse_arguments(CPSI "" "${oneValueArgs}" "" ${ARGN})

    if(NOT CPSI_ROOT_DIR)
        message(FATAL_ERROR "collect_project_source_inventory requires ROOT_DIR.")
    endif()

    file(GLOB_RECURSE _compiled_sources CONFIGURE_DEPENDS
        "${CPSI_ROOT_DIR}/*.cpp"
        "${CPSI_ROOT_DIR}/*.cu")
    file(GLOB_RECURSE _ptx_sources CONFIGURE_DEPENDS
        "${CPSI_ROOT_DIR}/*.ptx.cu")

    if(_ptx_sources)
        list(REMOVE_ITEM _compiled_sources ${_ptx_sources})
    endif()

    set(_filtered_compiled_sources "")
    foreach(_source_file IN LISTS _compiled_sources)
        file(RELATIVE_PATH _source_rel_path "${CPSI_ROOT_DIR}" "${_source_file}")
        string(REPLACE "\\" "/" _source_rel_path "${_source_rel_path}")
        if(_source_rel_path MATCHES "^bin(/|$)")
            continue()
        endif()
        list(APPEND _filtered_compiled_sources "${_source_file}")
    endforeach()

    set(_filtered_ptx_sources "")
    foreach(_source_file IN LISTS _ptx_sources)
        file(RELATIVE_PATH _source_rel_path "${CPSI_ROOT_DIR}" "${_source_file}")
        string(REPLACE "\\" "/" _source_rel_path "${_source_rel_path}")
        if(_source_rel_path MATCHES "^bin(/|$)")
            continue()
        endif()
        list(APPEND _filtered_ptx_sources "${_source_file}")
    endforeach()

    list(REMOVE_DUPLICATES _filtered_compiled_sources)
    list(REMOVE_DUPLICATES _filtered_ptx_sources)

    if(CPSI_COMPILED_SOURCES_OUT)
        set(${CPSI_COMPILED_SOURCES_OUT} "${_filtered_compiled_sources}" PARENT_SCOPE)
    endif()
    if(CPSI_PTX_SOURCES_OUT)
        set(${CPSI_PTX_SOURCES_OUT} "${_filtered_ptx_sources}" PARENT_SCOPE)
    endif()
    if(CPSI_HAS_COMPILED_SOURCES_OUT)
        if(_filtered_compiled_sources)
            set(${CPSI_HAS_COMPILED_SOURCES_OUT} TRUE PARENT_SCOPE)
        else()
            set(${CPSI_HAS_COMPILED_SOURCES_OUT} FALSE PARENT_SCOPE)
        endif()
    endif()
    if(CPSI_HAS_PTX_SOURCES_OUT)
        if(_filtered_ptx_sources)
            set(${CPSI_HAS_PTX_SOURCES_OUT} TRUE PARENT_SCOPE)
        else()
            set(${CPSI_HAS_PTX_SOURCES_OUT} FALSE PARENT_SCOPE)
        endif()
    endif()
endfunction()

# Function to add examples files to the build
function(add_examples project_lib_name excluded_list target_compile_settings)

    set(EXAMPLES_PATTERN "example_*.cpp; example_*.cu")
    file(GLOB srcExampleFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${EXAMPLES_PATTERN})

    # Exclude files in excluded_list (filter_files_in_list is assumed to be a custom macro)
    filter_files_in_list(srcExampleFiles srcExampleFiles ${excluded_list})

    # Get current folder name
    get_filename_component(CURRENT_FOLDER_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    message(STATUS "Example files found in ${CURRENT_FOLDER_NAME}: ${srcExampleFiles}")

    #message(STATUS "Project linked lib ${project_lib_name} to examples: ${srcExampleFiles}")
    #message(STATUS "Include directories of project lib: ${${project_lib_name}_INCLUDE_DIRS}")

    foreach(exampleFile ${srcExampleFiles})
        get_filename_component(exampleName ${exampleFile} NAME_WE)
        add_executable(${exampleName} ${exampleFile})
        target_link_libraries(${exampleName} PRIVATE ${project_lib_name} ${target_compile_settings})
        target_include_directories(${exampleName} PRIVATE ${${project_lib_name}_INCLUDE_DIRS})
        if(SPDLOG_ENABLED)
            target_compile_definitions(${exampleName} PRIVATE SPDLOG_UTILS_ENABLED=1)
        else()
            target_compile_definitions(${exampleName} PRIVATE SPDLOG_UTILS_ENABLED=0)
        endif()
    endforeach()

endfunction()

# Function to add test files to the build
if (Catch2_FOUND)
    function(add_tests project_lib_name excluded_list TESTS_LIST target_compile_settings CATCH2_TEST_PROPERTIES catch2_target)

        set(TESTS_PATTERN "test*.cpp; test*.cu")
        file(GLOB srcTestFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${TESTS_PATTERN})
        filter_files_in_list(srcTestFiles srcTestFiles ${excluded_list})
        
        # Get current folder name
        get_filename_component(CURRENT_FOLDER_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
        message(STATUS "Test files found in ${CURRENT_FOLDER_NAME}: ${srcTestFiles}")

        # Add content of this directory 
        foreach(testFile ${srcTestFiles})

            get_filename_component(testName ${testFile} NAME_WE)
            add_executable(${testName} ${testFile})

            list(APPEND ${TESTS_LIST} ${testName}) 

            target_link_libraries(${testName} PRIVATE ${project_lib_name} ${target_compile_settings} ${catch2_target})
            if(SPDLOG_ENABLED)
                target_compile_definitions(${testName} PRIVATE SPDLOG_UTILS_ENABLED=1)
            else()
                target_compile_definitions(${testName} PRIVATE SPDLOG_UTILS_ENABLED=0)
            endif()
            catch_discover_tests(${testName} PROPERTIES ${CATCH2_TEST_PROPERTIES})

        endforeach()
    endfunction()
else()
    message(WARNING "Catch2 not found. Command to add tests will not be available!")
endif()

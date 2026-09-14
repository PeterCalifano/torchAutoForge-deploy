include_guard(GLOBAL)

function(_doxygen_bool output_var value)
    if(${value})
        set(${output_var} "YES" PARENT_SCOPE)
    else()
        set(${output_var} "NO" PARENT_SCOPE)
    endif()
endfunction()

function(_doxygen_make_path_list output_var)
    set(paths)
    foreach(path IN LISTS ARGN)
        if("${path}" STREQUAL "")
            continue()
        endif()
        if(IS_ABSOLUTE "${path}")
            list(APPEND paths "${path}")
        else()
            get_filename_component(
                absolute_path "${path}" ABSOLUTE
                BASE_DIR "${PROJECT_SOURCE_DIR}")
            list(APPEND paths "${absolute_path}")
        endif()
    endforeach()
    string(REPLACE ";" " " paths_string "${paths}")
    set(${output_var} "${paths_string}" PARENT_SCOPE)
endfunction()

function(handle_doxygen_docs)
    set(one_value_args
        TARGET_PREFIX MAIN_PAGE LAYOUT_FILE DOXYFILE_IN)
    set(multi_value_args
        INPUT_DIRS EXCLUDE_DIRS EXCLUDE_PATTERNS)
    cmake_parse_arguments(
        DOC "" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if(NOT DOXYGEN_FOUND)
        message(STATUS "Doxygen not found. Skipping documentation targets.")
        return()
    endif()

    if(NOT DOC_TARGET_PREFIX)
        set(DOC_TARGET_PREFIX "${PROJECT_NAME}")
    endif()
    if(NOT DOC_DOXYFILE_IN)
        set(DOC_DOXYFILE_IN
            "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in")
    endif()
    if(NOT DOC_MAIN_PAGE)
        set(DOC_MAIN_PAGE
            "${CMAKE_CURRENT_SOURCE_DIR}/main_page.md")
    endif()
    if(NOT DOC_LAYOUT_FILE)
        set(DOC_LAYOUT_FILE
            "${CMAKE_CURRENT_SOURCE_DIR}/DoxygenLayout.xml")
    endif()
    if(NOT DOC_INPUT_DIRS)
        set(DOC_INPUT_DIRS
            "${PROJECT_SOURCE_DIR}/src"
            "${PROJECT_SOURCE_DIR}/doc")
    endif()

    option(BUILD_DOC_HTML "Enable Doxygen HTML output" ON)
    option(BUILD_DOC_LATEX "Enable Doxygen LaTeX output" OFF)
    option(BUILD_DOC_XML
           "Enable Doxygen XML output for wrapper docstrings" OFF)
    option(DOC_WARN_AS_ERROR "Treat Doxygen warnings as errors" OFF)

    _doxygen_bool(BUILD_DOC_HTML_YN BUILD_DOC_HTML)
    _doxygen_bool(BUILD_DOC_LATEX_YN BUILD_DOC_LATEX)
    _doxygen_bool(BUILD_DOC_XML_YN BUILD_DOC_XML)
    _doxygen_bool(DOXYGEN_WARN_AS_ERROR DOC_WARN_AS_ERROR)

    if(DOXYGEN_DOT_FOUND)
        set(DOXYGEN_HAVE_DOT "YES")
        message(STATUS "Graphviz dot found: Doxygen graphs enabled")
    else()
        set(DOXYGEN_HAVE_DOT "NO")
    endif()

    _doxygen_make_path_list(DOXYGEN_INPUT_PATHS ${DOC_INPUT_DIRS})
    _doxygen_make_path_list(DOXYGEN_EXCLUDE_PATHS ${DOC_EXCLUDE_DIRS})
    string(REPLACE
        ";" " " DOXYGEN_EXCLUDE_PATTERNS "${DOC_EXCLUDE_PATTERNS}")

    get_filename_component(
        DOXYGEN_MAIN_PAGE "${DOC_MAIN_PAGE}" ABSOLUTE
        BASE_DIR "${PROJECT_SOURCE_DIR}")
    get_filename_component(
        DOXYGEN_LAYOUT_FILE "${DOC_LAYOUT_FILE}" ABSOLUTE
        BASE_DIR "${PROJECT_SOURCE_DIR}")

    set(doc_binary_dir "${CMAKE_CURRENT_BINARY_DIR}")
    set(doc_html_dir "${doc_binary_dir}/html")
    set(doc_latex_dir "${doc_binary_dir}/latex")
    set(doc_xml_dir "${doc_binary_dir}/xml")

    set(${PROJECT_NAME}_DOXYGEN_HTML_DIR
        "${doc_html_dir}" CACHE INTERNAL
        "HTML output directory for ${PROJECT_NAME}" FORCE)
    set(${PROJECT_NAME}_DOXYGEN_XML_DIR
        "${doc_xml_dir}" CACHE INTERNAL
        "XML output directory for ${PROJECT_NAME}" FORCE)
    set(${PROJECT_NAME}_DOXYGEN_DOXYFILE
        "${doc_binary_dir}/Doxyfile" CACHE INTERNAL
        "Configured Doxyfile for ${PROJECT_NAME}" FORCE)

    configure_file(
        "${DOC_DOXYFILE_IN}" "${doc_binary_dir}/Doxyfile" @ONLY)

    set(doc_target "${DOC_TARGET_PREFIX}_doc")
    set(doc_clean_target "${DOC_TARGET_PREFIX}_doc_clean")
    if(NOT TARGET ${doc_target})
        add_custom_target(
            ${doc_target}
            COMMAND
                "${DOXYGEN_EXECUTABLE}" "${doc_binary_dir}/Doxyfile"
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            COMMENT "Generating ${PROJECT_NAME} API documentation"
            VERBATIM)
    endif()
    if(NOT TARGET ${doc_clean_target})
        add_custom_target(
            ${doc_clean_target}
            COMMAND "${CMAKE_COMMAND}" -E remove_directory "${doc_html_dir}"
            COMMAND "${CMAKE_COMMAND}" -E remove_directory "${doc_latex_dir}"
            COMMAND "${CMAKE_COMMAND}" -E remove_directory "${doc_xml_dir}"
            COMMENT "Removing ${PROJECT_NAME} Doxygen documentation"
            VERBATIM)
    endif()

    if(BUILD_AS_MAIN_PROJECT AND NOT TARGET doc)
        add_custom_target(doc DEPENDS ${doc_target})
    endif()
    if(BUILD_AS_MAIN_PROJECT AND NOT TARGET doc_clean)
        add_custom_target(doc_clean DEPENDS ${doc_clean_target})
    endif()
endfunction()

# Doxygen documentation configuration

# Check if list with subfolders is defined, skip otherwise
if (DEFINED DOC_SUBDIRS AND NOT DOC_SUBDIRS STREQUAL "")
    message(STATUS "Configuring Doxygen documentation...")
    # configure doxygen
    option(BUILD_DOC_HTML              "Enable/Disable doxygen HTML output"    ON)
    option(BUILD_DOC_LATEX             "Enable/Disable doxygen LaTeX output"   OFF)

    # add a target to generate API documentation with Doxygen
    # Convert configuration to YES/NO variables
    if (BUILD_DOC_HTML)
        set(BUILD_DOC_HTML_YN "YES")
    else() 
        set(BUILD_DOC_HTML_YN "NO")
    endif()

    if (BUILD_DOC_LATEX)
        set(BUILD_DOC_LATEX_YN "YES")
    else() 
        set(BUILD_DOC_LATEX_YN "NO")
    endif()

    # From subfolders, build a list with whitespace separation of paths
    set(DOXYGEN_INPUT_PATHS "")
    foreach(dir ${DOC_SUBDIRS})
        set(DOXYGEN_INPUT_PATHS "${DOXYGEN_INPUT_PATHS} ${dir}")
    endforeach()

    # Generate Doxyfile (@ONLY is required to expand variable from cmake)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY) 

    # Add target to actually build documentation as configured
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM
    )

    # Clean target
    add_custom_target(doc_clean
        COMMAND
            cmake -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/latex
        COMMAND
            cmake -E remove_directory ${CMAKE_CURRENT_SOURCE_DIR}/html
        COMMENT "Removing Doxygen documentation"
    )

else()
    message(STATUS "Documentation is requested but DOC_SUBDIRS is not defined or empty. Skipping building documentation.")
endif()
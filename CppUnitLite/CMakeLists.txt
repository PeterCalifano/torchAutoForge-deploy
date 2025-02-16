# Build/install CppUnitLite
file(GLOB cppunitlite_headers "*.h")
file(GLOB cppunitlite_src "*.cpp")


if(NOT TARGET CppUnitLite)

    add_library(CppUnitLite STATIC ${cppunitlite_src} ${cppunitlite_headers})
    list(APPEND GTSAM_EXPORTED_TARGETS CppUnitLite)
    set(GTSAM_EXPORTED_TARGETS "${GTSAM_EXPORTED_TARGETS}" PARENT_SCOPE)

    option(GTSAM_INSTALL_CPPUNITLITE "Enable/Disable installation of CppUnitLite library" ON)
    if (GTSAM_INSTALL_CPPUNITLITE)

        install(FILES ${cppunitlite_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/CppUnitLite)
        install(TARGETS CppUnitLite EXPORT GTSAM-exports ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

    endif(GTSAM_INSTALL_CPPUNITLITE)
    
endif()
# CMake configuration for profiling-friendly builds.
#
# When ENABLE_PROFILING=ON:
#   - Adds -fno-omit-frame-pointer so perf/callgrind/gperftools produce useful call stacks
#   - Adds -fno-inline-functions to prevent inlining from obscuring call graphs
#   - Detects and links gperftools (libprofiler + libtcmalloc) whenever found.
#     Set GPERFTOOLS_LINK=OFF to suppress linking and use LD_PRELOAD only.
#   - Detects Valgrind client headers and exposes them when found (no option, zero overhead).
#     Defines VALGRIND_ANNOTATIONS_ENABLED when available; use with valgrind/callgrind.h etc.
#
# GPERFTOOLS_LINK controls compile-time linking behaviour:
#   ON  (default) - link whenever the library is found, all build types
#   OFF           - never link; use LD_PRELOAD via profiling/run_gperf_profiling.sh
#
# ACHTUNG! -fno-omit-frame-pointer applies to all build types including Release.
# Runtime overhead is ~1% but is required for accurate stack unwinding.

include_guard(GLOBAL)

option(ENABLE_PROFILING "Enable profiling-friendly build flags and gperftools/valgrind detection" OFF)
option(GPERFTOOLS_LINK "Link gperftools at compile-time when found (ON); or use LD_PRELOAD only (OFF)" ON)

function(handle_profiling)
    cmake_parse_arguments(PROF "" "TARGET" "" ${ARGN})

    if(NOT DEFINED PROF_TARGET)
        if(DEFINED LIB_NAMESPACE)
            set(PROF_TARGET "${LIB_NAMESPACE}_profiling_interface")
        else()
            set(PROF_TARGET "profiling_interface")
        endif()
    endif()

    if(TARGET ${PROF_TARGET})
        return()
    endif()

    add_library(${PROF_TARGET} INTERFACE)

    if(NOT ENABLE_PROFILING)
        message(STATUS "Profiling support: OFF (set ENABLE_PROFILING=ON to enable)")
        return()
    endif()

    message(STATUS "Profiling support: ON")

    # Core flags required for accurate profiler stack unwinding.
    target_compile_options(${PROF_TARGET} INTERFACE
        -fno-omit-frame-pointer
        -fno-inline-functions
    )

    # --- gperftools detection ---
    find_path(GPERFTOOLS_INCLUDE_DIR
        NAMES gperftools/profiler.h google/profiler.h
        PATHS /usr/local/include /usr/include
    )

    find_library(GPERFTOOLS_PROFILER_LIBRARY
        NAMES profiler
        PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
    )

    find_library(GPERFTOOLS_TCMALLOC_LIBRARY
        NAMES tcmalloc tcmalloc_minimal
        PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
    )

    if(GPERFTOOLS_PROFILER_LIBRARY)
        message(STATUS "  gperftools libprofiler : ${GPERFTOOLS_PROFILER_LIBRARY}")
        set(GPERFTOOLS_PROFILER_LIBRARY "${GPERFTOOLS_PROFILER_LIBRARY}" CACHE PATH "Path to libprofiler" FORCE)
    else()
        message(STATUS "  gperftools libprofiler : not found (install libgoogle-perftools-dev)")
    endif()

    if(GPERFTOOLS_TCMALLOC_LIBRARY)
        message(STATUS "  gperftools libtcmalloc : ${GPERFTOOLS_TCMALLOC_LIBRARY}")
        set(GPERFTOOLS_TCMALLOC_LIBRARY "${GPERFTOOLS_TCMALLOC_LIBRARY}" CACHE PATH "Path to libtcmalloc" FORCE)
    else()
        message(STATUS "  gperftools libtcmalloc : not found")
    endif()

    # --- Link gperftools when found and GPERFTOOLS_LINK=ON ---
    if(GPERFTOOLS_LINK)
        if(GPERFTOOLS_PROFILER_LIBRARY AND GPERFTOOLS_INCLUDE_DIR)
            message(STATUS "  gperftools link mode   : compile-time")
            target_include_directories(${PROF_TARGET} INTERFACE ${GPERFTOOLS_INCLUDE_DIR})
            target_link_libraries(${PROF_TARGET} INTERFACE ${GPERFTOOLS_PROFILER_LIBRARY})
            if(GPERFTOOLS_TCMALLOC_LIBRARY)
                target_link_libraries(${PROF_TARGET} INTERFACE ${GPERFTOOLS_TCMALLOC_LIBRARY})
            endif()
            target_compile_definitions(${PROF_TARGET} INTERFACE PROFILING_GPERFTOOLS_ENABLED)
        else()
            message(STATUS "  gperftools link mode   : library not found, falling back to LD_PRELOAD")
        endif()
    else()
        message(STATUS "  gperftools link mode   : LD_PRELOAD only (GPERFTOOLS_LINK=OFF)")
    endif()

    # Valgrind client headers (no option, zero runtime overhead when not running under Valgrind)
    # Exposes valgrind/valgrind.h, valgrind/callgrind.h, valgrind/memcheck.h for source
    # annotations (CALLGRIND_START/STOP_INSTRUMENTATION, VALGRIND_MALLOCLIKE_BLOCK, etc.).
    # Macros expand to no-ops when not running under Valgrind.
    find_path(VALGRIND_INCLUDE_DIR
        NAMES valgrind/valgrind.h
        PATHS /usr/local/include /usr/include
    )

    if(VALGRIND_INCLUDE_DIR)
        message(STATUS "  Valgrind headers       : ${VALGRIND_INCLUDE_DIR} (VALGRIND_ANNOTATIONS_ENABLED)")
        target_include_directories(${PROF_TARGET} INTERFACE ${VALGRIND_INCLUDE_DIR})
        target_compile_definitions(${PROF_TARGET} INTERFACE VALGRIND_ANNOTATIONS_ENABLED)
    else()
        message(STATUS "  Valgrind headers       : not found (install valgrind)")
    endif()

    set(PROFILING_TARGET ${PROF_TARGET} PARENT_SCOPE)
endfunction()

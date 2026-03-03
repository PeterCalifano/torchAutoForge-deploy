# CMake configuration for profiling-friendly builds.
#
# When ENABLE_PROFILING=ON:
#   - Adds -fno-omit-frame-pointer so perf/callgrind/gperftools produce useful call stacks
#   - Adds -fno-inline-functions to prevent inlining from obscuring call graphs
#   - Detects gperftools (libprofiler + libtcmalloc) and links them by default for
#     non-Release builds; for Release builds linking is OFF by default (LD_PRELOAD only).
#
# GPERFTOOLS_LINK controls compile-time linking behaviour:
#   auto (default) — link for Debug/RelWithDebInfo/MinSizeRel/NOPTIM; skip for Release
#   ON             — always link (also for Release; needed for ProfilerStart/Stop API)
#   OFF            — never link (always use LD_PRELOAD via profiling scripts)
#
# ACHTUNG! -fno-omit-frame-pointer applies to all build types including Release.
# Runtime overhead is ~1% but is required for accurate stack unwinding.

include_guard(GLOBAL)

option(ENABLE_PROFILING "Enable profiling-friendly build flags and gperftools detection" OFF)
set(GPERFTOOLS_LINK "auto" CACHE STRING
    "gperftools compile-time link mode: auto (skip for Release), ON (always), OFF (never)")
set_property(CACHE GPERFTOOLS_LINK PROPERTY STRINGS auto ON OFF)

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
    # Find include dir
    find_path(GPERFTOOLS_INCLUDE_DIR
        NAMES gperftools/profiler.h google/profiler.h
        PATHS /usr/local/include /usr/include
    )

    # Find libprofiler (CPU + heap profiler)
    find_library(GPERFTOOLS_PROFILER_LIBRARY
        NAMES profiler
        PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
    )

    # Find libtcmalloc
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

    # --- Resolve effective link behaviour ---
    # auto: link for all build types except Release (where LD_PRELOAD is preferred)
    # ON:   always link (user override, also enables profiling API in Release)
    # OFF:  never link (purely LD_PRELOAD workflow)
    set(_do_link FALSE)
    if(GPERFTOOLS_LINK STREQUAL "ON")
        set(_do_link TRUE)
    elseif(GPERFTOOLS_LINK STREQUAL "auto" OR GPERFTOOLS_LINK STREQUAL "")
        string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_upper)
        if(NOT _build_upper STREQUAL "RELEASE")
            set(_do_link TRUE)
        endif()
    endif()
    # GPERFTOOLS_LINK=OFF leaves _do_link FALSE

    if(_do_link)
        if(GPERFTOOLS_PROFILER_LIBRARY AND GPERFTOOLS_INCLUDE_DIR)
            message(STATUS "  gperftools link mode   : compile-time (GPERFTOOLS_LINK=${GPERFTOOLS_LINK}, build=${CMAKE_BUILD_TYPE})")
            target_include_directories(${PROF_TARGET} INTERFACE ${GPERFTOOLS_INCLUDE_DIR})
            target_link_libraries(${PROF_TARGET} INTERFACE ${GPERFTOOLS_PROFILER_LIBRARY})
            if(GPERFTOOLS_TCMALLOC_LIBRARY)
                target_link_libraries(${PROF_TARGET} INTERFACE ${GPERFTOOLS_TCMALLOC_LIBRARY})
            endif()
            target_compile_definitions(${PROF_TARGET} INTERFACE PROFILING_GPERFTOOLS_ENABLED)
        else()
            message(STATUS "  gperftools link mode   : compile-time requested but library not found — falling back to LD_PRELOAD")
        endif()
    else()
        message(STATUS "  gperftools link mode   : LD_PRELOAD only (use profiling/run_gperf_profiling.sh)")
    endif()

    set(PROFILING_TARGET ${PROF_TARGET} PARENT_SCOPE)
endfunction()

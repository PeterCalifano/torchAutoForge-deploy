# Modified and improved by Pietro Califano from the NVIDIA OptiX SDK
# implementation, December 2025. The attribution below is retained for
# licensing provenance; this module has no OptiX build or runtime dependency.
# ======================================================================== #
# Copyright 2018 Ingo Wald                                                 #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

include_guard(GLOBAL)

set(PTAFDEPLOY_CUDA_PTX_TOOLS_DIR "${CMAKE_CURRENT_LIST_DIR}")

if(NOT CUDAToolkit_FOUND)
  message(FATAL_ERROR "CUDA Toolkit is required for PTX compilation.")
endif()

find_program(
  BIN2C
  NAMES bin2c
  HINTS "${CUDAToolkit_BIN_DIR}"
  DOC "Path to the CUDA bin2c executable")
if(NOT BIN2C)
  message(FATAL_ERROR
    "CUDA bin2c was not found. Set BIN2C to the executable shipped with "
    "the CUDA Toolkit.")
endif()

if(NOT COMMAND cuda_compile_and_embed)
  macro(cuda_compile_and_embed output_var cuda_arch include_dirs)
    set(cuda_files ${ARGN})
    if(NOT cuda_files)
      message(FATAL_ERROR
        "cuda_compile_and_embed requires at least one CUDA file.")
    endif()

    set(cuda_arch_values ${cuda_arch})
    if(NOT cuda_arch_values)
      message(FATAL_ERROR
        "PTX embedding requires one resolved CUDA architecture. "
        "Set CUDA_ARCHITECTURES or CMAKE_CUDA_ARCHITECTURES explicitly.")
    endif()

    list(LENGTH cuda_arch_values cuda_arch_count)
    if(NOT cuda_arch_count EQUAL 1)
      message(FATAL_ERROR
        "PTX embedding requires one CUDA architecture, but received "
        "'${cuda_arch}'.")
    endif()

    list(GET cuda_arch_values 0 cuda_arch_value)
    string(STRIP "${cuda_arch_value}" cuda_arch_value)
    if(NOT cuda_arch_value MATCHES "^[0-9]+$")
      message(FATAL_ERROR
        "PTX embedding expected a numeric CUDA architecture, but received "
        "'${cuda_arch_value}'.")
    endif()

    set(${output_var} "")
    set(${output_var}_SYMBOLS "")

    set(include_dirs_list)
    if(NOT "${include_dirs}" STREQUAL "")
      set(include_dirs_list ${include_dirs})
    endif()

    set(clean_include_dirs)
    foreach(include_dir IN LISTS include_dirs_list)
      if(include_dir MATCHES "^\\$<BUILD_INTERFACE:([^>]+)>$")
        set(include_dir "${CMAKE_MATCH_1}")
      elseif(include_dir MATCHES "^\\$<INSTALL_INTERFACE:([^>]+)>$")
        continue()
      elseif(include_dir MATCHES "^\\$<.+>$")
        continue()
      endif()
      if(NOT "${include_dir}" STREQUAL "")
        list(APPEND clean_include_dirs "${include_dir}")
      endif()
    endforeach()
    list(REMOVE_DUPLICATES clean_include_dirs)

    set(expanded_include_dirs)
    foreach(include_dir IN LISTS clean_include_dirs)
      list(APPEND expanded_include_dirs "-I${include_dir}")
    endforeach()

    set(cuda_cxx_standard_flag)
    if(DEFINED CMAKE_CUDA_STANDARD)
      set(cuda_cxx_standard_flag "-std=c++${CMAKE_CUDA_STANDARD}")
    elseif(DEFINED CMAKE_CXX_STANDARD)
      set(cuda_cxx_standard_flag "-std=c++${CMAKE_CXX_STANDARD}")
    endif()

    set(ptx_nvcc_flags ${CUDA_SHARED_NVCC_FLAGS})
    if(CUDA_PTX_USE_FAST_MATH)
      list(APPEND ptx_nvcc_flags --use_fast_math)
    endif()
    if(DEFINED CUDA_PTX_NVCC_FLAGS
       AND NOT "${CUDA_PTX_NVCC_FLAGS}" STREQUAL "")
      separate_arguments(
        ptx_user_nvcc_flags UNIX_COMMAND "${CUDA_PTX_NVCC_FLAGS}")
      list(APPEND ptx_nvcc_flags ${ptx_user_nvcc_flags})
    endif()
    list(REMOVE_DUPLICATES ptx_nvcc_flags)

    list(LENGTH cuda_files cuda_files_count)
    foreach(cuda_file IN LISTS cuda_files)
      get_filename_component(cuda_file_name "${cuda_file}" NAME_WE)
      set(ptx_file "${CMAKE_CURRENT_BINARY_DIR}/${cuda_file_name}.ptx")

      if(cuda_files_count GREATER 1)
        set(c_var_name "${output_var}_${cuda_file_name}")
        string(MAKE_C_IDENTIFIER "${c_var_name}" c_var_name)
      else()
        set(c_var_name "${output_var}")
      endif()

      add_custom_command(
        OUTPUT "${ptx_file}"
        COMMAND
          "${CMAKE_CUDA_COMPILER}"
          --ptx
          --generate-line-info
          --keep
          "${cuda_cxx_standard_flag}"
          --relocatable-device-code=true
          ${ptx_nvcc_flags}
          "-arch=sm_${cuda_arch_value}"
          "${cuda_file}"
          -o "${ptx_file}"
          ${expanded_include_dirs}
        DEPENDS "${cuda_file}"
        VERBATIM
        COMMENT "Compiling ${cuda_file_name} to PTX")

      set(embedded_file
          "${CMAKE_CURRENT_BINARY_DIR}/${cuda_file_name}_embedded.c")
      add_custom_command(
        OUTPUT "${embedded_file}"
        COMMAND
          "${CMAKE_COMMAND}"
          "-DPTAFDEPLOY_BIN2C=${BIN2C}"
          "-DPTAFDEPLOY_PTX_INPUT=${ptx_file}"
          "-DPTAFDEPLOY_EMBEDDED_OUTPUT=${embedded_file}"
          "-DPTAFDEPLOY_EMBEDDED_SYMBOL=${c_var_name}"
          -P "${PTAFDEPLOY_CUDA_PTX_TOOLS_DIR}/RunBin2C.cmake"
        DEPENDS "${ptx_file}"
        VERBATIM
        COMMENT "Embedding ${cuda_file_name}.ptx as ${c_var_name}")

      set_source_files_properties(
        "${embedded_file}" PROPERTIES GENERATED TRUE LANGUAGE C)
      list(APPEND ${output_var} "${embedded_file}")
      list(APPEND ${output_var}_SYMBOLS "${c_var_name}")
    endforeach()
  endmacro()
endif()

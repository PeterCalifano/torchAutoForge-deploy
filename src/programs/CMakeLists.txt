# Discover source-owned native CLI entry points.
set(EXAMPLES_PATTERN "*.cpp" "*.cu")
file(GLOB srcProgramsFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${EXAMPLES_PATTERN})

# Report the discovered sources during configuration for build diagnostics.
get_filename_component(CURRENT_FOLDER_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
message(STATUS "Programs files found in ${CURRENT_FOLDER_NAME}: ${srcProgramsFiles}")

# Retain the shared exclusion hook used by template-derived source directories.
set(EXCLUDED_LIST "")
list(REMOVE_ITEM srcProgramsFiles ${EXCLUDED_LIST})
if(NOT autoforge_deploy_ENABLE_INFERENCE_OUTPUT)
  list(REMOVE_ITEM srcProgramsFiles evaluate_inference.cpp prepare_image_sequence.cpp)
elseif(NOT autoforge_deploy_ENABLE_IMAGE_SUPPORT)
  list(REMOVE_ITEM srcProgramsFiles prepare_image_sequence.cpp)
endif()

set(programs_targets "")

# Build each CLI with the core runtime and source-private parser interface.
foreach(exampleFile ${srcProgramsFiles})
  get_filename_component(program_name ${exampleFile} NAME_WE)
  add_executable(${program_name} ${exampleFile})

  target_link_libraries(${program_name} PRIVATE
    ${project_name}
    ${CUDA_COMPILE_TARGET}
    ${TCLAP_COMPILE_TARGET})

  if(program_name STREQUAL "evaluate_inference" OR program_name STREQUAL "prepare_image_sequence")
    target_link_libraries(${program_name} PRIVATE autoforge_deploy::inference_output)
  endif()
  if(program_name STREQUAL "prepare_image_sequence")
    target_link_libraries(${program_name} PRIVATE autoforge_deploy::images)
  endif()

  target_compile_definitions(${program_name} PRIVATE
    PTAFDEPLOY_CLI_VERSION="${FULL_VERSION}")

  set_target_properties(${program_name} PROPERTIES OUTPUT_NAME "${program_name}")
  list(APPEND programs_targets ${program_name})
endforeach()

# Install all discovered command-line tools together.
install(TARGETS ${programs_targets} RUNTIME DESTINATION bin)

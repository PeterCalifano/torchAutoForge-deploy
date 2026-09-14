foreach(required_variable
        PTAFDEPLOY_BIN2C
        PTAFDEPLOY_PTX_INPUT
        PTAFDEPLOY_EMBEDDED_OUTPUT
        PTAFDEPLOY_EMBEDDED_SYMBOL)
  if(NOT DEFINED ${required_variable}
     OR "${${required_variable}}" STREQUAL "")
    message(FATAL_ERROR "${required_variable} is required.")
  endif()
endforeach()

execute_process(
  COMMAND
    "${PTAFDEPLOY_BIN2C}"
    -c
    --padd 0
    --type char
    --name "${PTAFDEPLOY_EMBEDDED_SYMBOL}"
    "${PTAFDEPLOY_PTX_INPUT}"
  RESULT_VARIABLE bin2c_result
  OUTPUT_FILE "${PTAFDEPLOY_EMBEDDED_OUTPUT}"
  ERROR_VARIABLE bin2c_error)

if(NOT bin2c_result EQUAL 0)
  file(REMOVE "${PTAFDEPLOY_EMBEDDED_OUTPUT}")
  message(FATAL_ERROR
    "bin2c failed with exit code ${bin2c_result}: ${bin2c_error}")
endif()

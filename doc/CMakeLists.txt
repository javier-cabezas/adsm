########### install files ###############

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

execute_process(WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND doxygen Doxyfile
    RESULT_VARIABLE ret_var
    OUTPUT_VARIABLE output_var)

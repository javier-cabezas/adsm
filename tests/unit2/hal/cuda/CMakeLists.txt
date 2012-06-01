include(FindCUDA)

# Remove c++0x for now, since nvcc seems not very happy with it
string(REGEX REPLACE "[-]std[=]c[+][+]0x" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

cuda_compile_ptx(common_lib common.cu)

add_custom_target(backend_code
    DEPENDS ${common_lib}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/code
    COMMAND ${CMAKE_COMMAND} -E copy ${common_lib}
    ${CMAKE_BINARY_DIR}/code/common.lib)

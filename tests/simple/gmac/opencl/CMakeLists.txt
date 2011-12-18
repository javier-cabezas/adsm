set(common_ROOT ${CMAKE_SOURCE_DIR}/tests/common)

include_directories(${PROJECT_SOURCE_DIR}/src/include ${PROJECT_BINARY_DIR}/src/include ${common_ROOT} ${OPENCL_INCLUDE})

set(common_SRC
    ${common_ROOT}/debug.h
    ${common_ROOT}/utils.h ${common_ROOT}/utils-impl.h ${common_ROOT}/utils.cpp 
    ${common_ROOT}/barrier.h
    ${common_ROOT}/semaphore.h
    ${common_ROOT}/cycle.h)
set(common_os_SRC
        ${common_ROOT}/${OS_DIR}/utils.cpp
        ${common_ROOT}/${OS_DIR}/semaphore.h
        ${common_ROOT}/${OS_DIR}/semaphore.c
        ${common_ROOT}/${OS_DIR}/barrier.h
        ${common_ROOT}/${OS_DIR}/barrier.c)
# Set source group for common files
string(REPLACE "${CMAKE_SOURCE_DIR}" "" common_GROUP ${common_ROOT})
string(REGEX REPLACE "^/" "" common_GROUP ${common_GROUP})
string(REPLACE "/" "\\\\" common_GROUP ${common_GROUP})
source_group(${common_GROUP} FILES ${common_SRC})
source_group("${common_GROUP}\\${OS_DIR}" FILES ${common_os_SRC})
set(common_SRC ${common_SRC} ${common_os_SRC})

# Create a source group for current directory
string(REPLACE "${CMAKE_SOURCE_DIR}" "" current_GROUP ${CMAKE_CURRENT_SOURCE_DIR})
string(REGEX REPLACE "^/" "" current_GROUP ${current_GROUP})
string(REPLACE "/" "\\\\" current_GROUP ${current_GROUP})
source_group(${current_GROUP} FILES
    oclVecAdd.cpp
    NBody/NBody.h
    NBody/NBody.cpp
    eclCompressCommon.cl
    eclMatrixMulKernel.cl
    c/eclBarr.cpp
    c/eclFile.cpp
    c/eclFileVecAdd.cpp
    c/eclGetAccInfo.cpp
    c/eclInit.cpp
    c/eclMatrixMul.cpp
    c/eclMemcpy.cpp
    c/eclMemset.cpp
    c/eclPingPong.cpp
    c/eclSharedVecAdd.cpp
    c/eclStencil.cpp
    c/eclStencilCommon.h
    c/eclThreading.cpp
    c/eclThreadVecAdd.cpp
    c/eclThreadStencil.cpp
    c/eclVecAdd.cpp
	c/eclThreadVecAdd_eclMemcpy.cpp
    c/NBody_GMAC/NBody.h
    c/NBody_GMAC/NBody.cpp
    cpp/eclColumnMajor.cpp
    cpp/eclCompress.cpp
    cpp/eclCompressSend.cpp
    cpp/eclPartialVecAdd_cplusplus.cpp
    cpp/eclStaticVecAdd_cplusplus.cpp
    cpp/eclSharedPtrVecAdd_cplusplus.cpp
    cpp/eclMultiArrayMatrixMul_cplusplus.cpp
    cpp/eclRandomAccess.cpp
    cpp/eclVecAdd_cplusplus.cpp
    cpp/eclFileVecAdd_cplusplus.cpp
)

# Plain OpenCL tests

add_executable(oclVecAdd ${common_SRC} oclVecAdd.cpp)
target_link_libraries(oclVecAdd ${opencl_LIB} ${thread_LIB})

# C API

add_executable(eclBarr ${common_SRC} c/eclBarr.cpp)
target_link_libraries(eclBarr gmac-hpe)

add_executable(eclFile ${common_SRC} c/eclFile.cpp)
target_link_libraries(eclFile gmac-hpe)

add_executable(eclFileVecAdd ${common_SRC} c/eclFileVecAdd.cpp)
target_link_libraries(eclFileVecAdd gmac-hpe)

add_executable(eclGetAccInfo ${common_SRC} c/eclGetAccInfo.cpp)
target_link_libraries(eclGetAccInfo gmac-hpe)

add_executable(eclInit ${common_SRC} c/eclInit.cpp)
target_link_libraries(eclInit gmac-hpe)

add_executable(eclMatrixMul ${common_SRC} c/eclMatrixMul.cpp eclMatrixMulKernel.cl)
target_link_libraries(eclMatrixMul gmac-hpe)

add_executable(eclMemcpy ${common_SRC} c/eclMemcpy.cpp)
target_link_libraries(eclMemcpy gmac-hpe)

add_executable(eclMemset ${common_SRC} c/eclMemset.cpp)
target_link_libraries(eclMemset gmac-hpe)

# add_executable(eclPingPong ${common_SRC} c/eclPingPong.cpp)
# target_link_libraries(eclPingPong gmac-hpe)

add_executable(eclSharedVecAdd ${common_SRC} c/eclSharedVecAdd.cpp)
target_link_libraries(eclSharedVecAdd gmac-hpe)

add_executable(eclStencil ${common_SRC} c/eclStencil.cpp c/eclStencilCommon.h)
target_link_libraries(eclStencil gmac-hpe)

add_executable(eclThreading ${common_SRC} c/eclThreading.cpp)
target_link_libraries(eclThreading gmac-hpe)

add_executable(eclThreadStencil ${common_SRC} c/eclThreadStencil.cpp c/eclStencilCommon.h)
target_link_libraries(eclThreadStencil gmac-hpe)

add_executable(eclThreadVecAdd ${common_SRC} c/eclThreadVecAdd.cpp)
target_link_libraries(eclThreadVecAdd gmac-hpe)

add_executable(eclVecAdd ${common_SRC} c/eclVecAdd.cpp)
target_link_libraries(eclVecAdd gmac-hpe)

add_executable(eclThreadVecAdd_eclMemcpy ${common_SRC} c/eclThreadVecAdd_eclMemcpy.cpp)
target_link_libraries(eclThreadVecAdd_eclMemcpy gmac-hpe)


# C++ API

add_executable(eclColumnMajor_cpp ${common_SRC} cpp/eclColumnMajor.cpp)
target_link_libraries(eclColumnMajor_cpp gmac-hpe)

add_executable(eclCompress_cpp ${common_SRC} cpp/eclCompress.cpp eclCompressCommon.cl)
target_link_libraries(eclCompress_cpp gmac-hpe)

# add_executable(eclCompressSend_cpp ${common_SRC} cpp/eclCompressSend.cpp eclCompressCommon.cl)
# target_link_libraries(eclCompressSend_cpp gmac-hpe)

add_executable(eclVecAdd_cpp ${common_SRC} cpp/eclVecAdd.cpp)
target_link_libraries(eclVecAdd_cpp gmac-hpe)

add_executable(eclPartialVecAdd_cpp ${common_SRC} cpp/eclPartialVecAdd.cpp)
target_link_libraries(eclPartialVecAdd_cpp gmac-hpe)

add_executable(eclStaticVecAdd_cpp ${common_SRC} cpp/eclStaticVecAdd.cpp)
target_link_libraries(eclStaticVecAdd_cpp gmac-hpe)

add_executable(eclFileVecAdd_cpp ${common_SRC} cpp/eclFileVecAdd_cplusplus.cpp)
target_link_libraries(eclFileVecAdd_cpp gmac-hpe)

if(USE_TR1)
add_executable(eclSharedPtrVecAdd_cpp ${common_SRC} cpp/eclSharedPtrVecAdd.cpp)
target_link_libraries(eclSharedPtrVecAdd_cpp gmac-hpe)
endif(USE_TR1)

add_executable(eclRandomAccess_cpp ${common_SRC} cpp/eclRandomAccess.cpp)
target_link_libraries(eclRandomAccess_cpp gmac-hpe)

if(USE_BOOST)
add_executable(eclMultiArrayMatrixMul_cpp ${common_SRC} cpp/eclMultiArrayMatrixMul.cpp eclMatrixMulKernel.cl)
target_link_libraries(eclMultiArrayMatrixMul_cpp gmac-hpe)
endif(USE_BOOST)

# OpenGL tests

option(USE_OPENGL "Enable building OpenGL tests" OFF)
if(USE_OPENGL)
    set(GL_LIBRARY "/usr/lib" CACHE PATH "GL Library Files")
    set(GLU_LIBRARY "/usr/lib" CACHE PATH "GLU Library Files")
    set(GLUT_LIBRARY "/usr/lib" CACHE PATH "GLUT Library Files")
    set(GLEW_LIBRARY "/usr/lib" CACHE PATH "GLEW Library Files")
    find_library(gl_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}GL${CMAKE_SHARED_LIBRARY_SUFFIX})
    find_library(glu_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}GLU${CMAKE_SHARED_LIBRARY_SUFFIX})
    find_library(glut_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}glut${CMAKE_SHARED_LIBRARY_SUFFIX}
        HINTS ${GLUT_LIBRARY} ${OPENCL_LIBRARY} ENV "OPENCL_LIBRARY_PATH")
    find_library(glew_LIB ${CMAKE_SHARED_LIBRARY_PREFIX}GLEW${CMAKE_SHARED_LIBRARY_SUFFIX}
        HINTS ${GLEW_LIBRARY} ${OPENCL_LIBRARY} ENV "OPENCL_LIBRARY_PATH")

    # Regular OpenCL application
    add_executable(oclNBody ${common_SRC} NBody/NBody.h NBody/NBody.cpp)
    target_link_libraries(oclNBody ${opencl_LIB} ${thread_LIB} ${opengl_LIBS})

    # C API
    set(opengl_LIBS ${gl_LIB} ${glu_LIB} ${glut_LIB} ${glew_LIB})
    message(STATUS "OpenGL Libraries: ${opengl_LIBS}")
    add_executable(eclNBody ${common_SRC} NBody_GMAC/NBody.h NBody_GMAC/NBody.cpp)
    target_link_libraries(eclNBody gmac-hpe ${opengl_LIBS})

    file(COPY NBody/NBody_Kernels.cl DESTINATION ${PROJECT_BINARY_DIR})
endif(USE_OPENGL)

file(COPY vars.spec tests.spec DESTINATION ${PROJECT_BINARY_DIR})

add_custom_target(InputSets ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../../input
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/inputset)

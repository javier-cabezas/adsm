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
    eclInit.cpp
    eclColumnMajor.cpp
    eclCompress.cpp
    eclCompressCommon.cl
    eclCompressSend.cl
    eclMatrixMul.cpp
    eclMatrixMulKernel.cl
    eclMemcpy.cpp
    eclMemset.cpp
    eclFile.cpp
    eclFileVecAdd.cpp
    eclPingPong.cpp
    eclBarr.cpp
    eclPartialVecAdd_cplusplus.cpp
    eclStaticVecAdd_cplusplus.cpp
    eclRandomAccess.cpp
    eclSharedVecAdd.cpp
    eclStencil.cpp
    eclStencilCommon.h
    eclThreading.cpp
    eclThreadVecAdd.cpp
    eclThreadStencil.cpp
    oclVecAdd.cpp
    eclVecAdd.cpp
    eclVecAdd_cplusplus.cpp
    NBody/NBody.h
    NBody/NBody.cpp
    NBody_GMAC/NBody.h
    NBody_GMAC/NBody.cpp
)

add_executable(eclInit ${common_SRC} eclInit.cpp)
target_link_libraries(eclInit gmac-hpe)

add_executable(eclColumnMajor ${common_SRC} eclColumnMajor.cpp)
target_link_libraries(eclColumnMajor gmac-hpe)

add_executable(eclCompress ${common_SRC} eclCompress.cpp eclCompressCommon.cl)
target_link_libraries(eclCompress gmac-hpe)

add_executable(eclCompressSend ${common_SRC} eclCompressSend.cpp eclCompressCommon.cl)
target_link_libraries(eclCompressSend gmac-hpe)

add_executable(oclVecAdd ${common_SRC} oclVecAdd.cpp)
target_link_libraries(oclVecAdd ${opencl_LIB} ${thread_LIB})

add_executable(eclVecAdd ${common_SRC} eclVecAdd.cpp)
target_link_libraries(eclVecAdd gmac-hpe)

add_executable(eclVecAdd_cplusplus ${common_SRC} eclVecAdd_cplusplus.cpp)
target_link_libraries(eclVecAdd_cplusplus gmac-hpe)

add_executable(eclThreadVecAdd ${common_SRC} eclThreadVecAdd.cpp)
target_link_libraries(eclThreadVecAdd gmac-hpe)

add_executable(eclPartialVecAdd_cplusplus ${common_SRC} eclPartialVecAdd_cplusplus.cpp)
target_link_libraries(eclPartialVecAdd_cplusplus gmac-hpe)

add_executable(eclStaticVecAdd_cplusplus ${common_SRC} eclStaticVecAdd_cplusplus.cpp)
target_link_libraries(eclStaticVecAdd_cplusplus gmac-hpe)

add_executable(eclMemcpy ${common_SRC} eclMemcpy.cpp)
target_link_libraries(eclMemcpy gmac-hpe)

add_executable(eclMemset ${common_SRC} eclMemset.cpp)
target_link_libraries(eclMemset gmac-hpe)

add_executable(eclFile ${common_SRC} eclFile.cpp)
target_link_libraries(eclFile gmac-hpe)

add_executable(eclFileVecAdd ${common_SRC} eclFileVecAdd.cpp)
target_link_libraries(eclFileVecAdd gmac-hpe)

add_executable(eclPingPong ${common_SRC} eclPingPong.cpp)
target_link_libraries(eclPingPong gmac-hpe)

add_executable(eclRandomAccess ${common_SRC} eclRandomAccess.cpp)
target_link_libraries(eclRandomAccess gmac-hpe)

add_executable(eclSharedVecAdd ${common_SRC} eclSharedVecAdd.cpp)
target_link_libraries(eclSharedVecAdd gmac-hpe)

add_executable(eclStencil ${common_SRC} eclStencil.cpp eclStencilCommon.h)
target_link_libraries(eclStencil gmac-hpe)

add_executable(eclThreadStencil ${common_SRC} eclThreadStencil.cpp eclStencilCommon.h)
target_link_libraries(eclThreadStencil gmac-hpe)

add_executable(eclThreading ${common_SRC} eclThreading.cpp)
target_link_libraries(eclThreading gmac-hpe)

add_executable(eclBarr ${common_SRC} eclBarr.cpp)
target_link_libraries(eclBarr gmac-hpe)

add_executable(eclMatrixMul ${common_SRC} eclMatrixMul.cpp eclMatrixMulKernel.cl)
target_link_libraries(eclMatrixMul gmac-hpe)

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

    set(opengl_LIBS ${gl_LIB} ${glu_LIB} ${glut_LIB} ${glew_LIB})
    message(STATUS "OpenGL Libraries: ${opengl_LIBS}")
    add_executable(eclNBody ${common_SRC} NBody_GMAC/NBody.h NBody_GMAC/NBody.cpp)
    target_link_libraries(eclNBody gmac-hpe ${opengl_LIBS})

    add_executable(oclNBody ${common_SRC} NBody/NBody.h NBody/NBody.cpp)
    target_link_libraries(oclNBody ${opencl_LIB} ${thread_LIB} ${opengl_LIBS})

    file(COPY NBody/NBody_Kernels.cl DESTINATION ${PROJECT_BINARY_DIR})
endif(USE_OPENGL)

file(COPY vars.spec tests.spec DESTINATION ${PROJECT_BINARY_DIR})

add_custom_target(InputSets ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../../input
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/inputset)

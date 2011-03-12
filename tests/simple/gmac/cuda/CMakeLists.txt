set(common_ROOT ${CMAKE_SOURCE_DIR}/tests/common)
include_directories(${PROJECT_SOURCE_DIR}/src/include ${PROJECT_BINARY_DIR}/src/include ${common_ROOT})

include(FindCUDA)

# Create a source group for current directory
string(REPLACE "${CMAKE_SOURCE_DIR}" "" current_GROUP ${CMAKE_CURRENT_SOURCE_DIR})
string(REGEX REPLACE "^/" "" current_GROUP ${current_GROUP})
string(REPLACE "/" "\\\\" current_GROUP ${current_GROUP})
source_group(${current_GROUP} FILES
    gmacAlloc.cu
    gmacVecAdd.cu
    gmacThreadVecAdd.cu
    cudaThreadVecAdd.cu
    gmacSharedVecAdd.cu
    gmacFileVecAdd.cu
    gmacMatrixMul.cu
    gmacSharedMatrixMul.cu
    gmacMemcpy.cu
    gmacMemset.cu
    gmacFile.cu
    gmacMigration.cu
    gmacMeminfo.cu
    gmacStencil.cu
    gmacThreadStencil.cu
    gmacStencilCommon.cu
    cudaThreadStencil.cu
    cudaStencilCommon.cu
    gmacConstant.cu
    gmacTexture.cu
    gmacCompressCommon.cu
    gmacCompress.cu
    gmacPingPong.cu
)

message(STATUS "Size of pointer: ${CMAKE_SIZEOF_VOID_P}")
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(CUDA_64_BIT_DEVICE_CODE ON)
else()
    set(CUDA_64_BIT_DEVICE_CODE OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCC)
    set(CUDA_NVCC_FLAGS "--ptxas-options=-v" "--compiler-options;-fno-strict-aliasing")
endif(CMAKE_COMPILER_IS_GNUCC)

# gmacAlloc
cuda_compile(alloc gmacAlloc.cu)
add_executable(gmacAlloc ${alloc}  gmacAlloc.cu)
target_link_libraries(gmacAlloc test-common test-common gmac)

# gmacVecAdd
cuda_compile(vecAdd gmacVecAdd.cu)
add_executable(gmacVecAdd ${vecAdd}  gmacVecAdd.cu)
target_link_libraries(gmacVecAdd test-common test-common gmac)

# gmacThreadVecAdd
cuda_compile(threadVecAdd gmacThreadVecAdd.cu)
add_executable(gmacThreadVecAdd ${threadVecAdd})
target_link_libraries(gmacThreadVecAdd test-common gmac)

# cudaThreadVecAdd
cuda_compile(threadVecAdd cudaThreadVecAdd.cu)
add_executable(cudaThreadVecAdd ${threadVecAdd})
target_link_libraries(cudaThreadVecAdd test-common ${CUDA_CUDART_LIBRARY})

# gmacSharedVecAdd
cuda_compile(sharedVecAdd gmacSharedVecAdd.cu)
add_executable(gmacSharedVecAdd ${sharedVecAdd}  gmacSharedVecAdd.cu)
target_link_libraries(gmacSharedVecAdd test-common gmac)

# gmacFileVecAdd
cuda_compile(fileVecAdd gmacFileVecAdd.cu)
add_executable(gmacFileVecAdd ${fileVecAdd}  gmacFileVecAdd.cu)
target_link_libraries(gmacFileVecAdd test-common gmac)

# gmacMatrixMul
cuda_compile(matrixMul gmacMatrixMul.cu)
add_executable(gmacMatrixMul ${matrixMul}  gmacMatrixMul.cu)
target_link_libraries(gmacMatrixMul test-common gmac)

# gmacSharedMatrixMul
cuda_compile(sharedMatrixMul gmacSharedMatrixMul.cu)
add_executable(gmacSharedMatrixMul ${sharedMatrixMul}  gmacSharedMatrixMul.cu)
target_link_libraries(gmacSharedMatrixMul test-common gmac)

# gmacStencil
cuda_compile(stencil gmacStencil.cu)
add_executable(gmacStencil ${stencil}  gmacStencil.cu gmacStencilCommon.cu)
target_link_libraries(gmacStencil test-common gmac)

# gmacThreadStencil
cuda_compile(threadStencil gmacThreadStencil.cu)
add_executable(gmacThreadStencil ${threadStencil}  gmacThreadStencil.cu gmacStencilCommon.cu)
target_link_libraries(gmacThreadStencil test-common gmac)

# cudaThreadStencil
cuda_compile(threadStencil cudaThreadStencil.cu)
add_executable(cudaThreadStencil ${threadStencil})
target_link_libraries(cudaThreadStencil test-common ${CUDA_CUDART_LIBRARY})

# gmacMemcpy
cuda_compile(memcpy gmacMemcpy.cu)
add_executable(gmacMemcpy ${memcpy}  gmacMemcpy.cu)
target_link_libraries(gmacMemcpy test-common gmac)

# gmacMemset
cuda_compile(memset gmacMemset.cu)
add_executable(gmacMemset ${memset}  gmacMemset.cu)
target_link_libraries(gmacMemset test-common gmac)

# gmacFile
cuda_compile(gfile gmacFile.cu)
add_executable(gmacFile ${gfile}  gmacFile.cu)
target_link_libraries(gmacFile test-common gmac)

# gmacMigration
cuda_compile(migration gmacMigration.cu)
add_executable(gmacMigration ${migration}  gmacMigration.cu)
target_link_libraries(gmacMigration test-common gmac)

# gmacMeminfo
cuda_compile(meminfo gmacMeminfo.cu)
add_executable(gmacMeminfo ${meminfo}  gmacMeminfo.cu)
target_link_libraries(gmacMeminfo test-common gmac)

# gmacTexture
cuda_compile(texture gmacTexture.cu)
add_executable(gmacTexture ${texture}  gmacTexture.cu)
target_link_libraries(gmacTexture test-common gmac)

# gmacConstant
cuda_compile(constant gmacConstant.cu)
add_executable(gmacConstant ${constant}  gmacConstant.cu)
target_link_libraries(gmacConstant test-common gmac)

if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
# gmacPingPong
cuda_compile(pingpong gmacPingPong.cu)
add_executable(gmacPingPong ${pingpong}  gmacPingPong.cu)
target_link_libraries(gmacPingPong test-common gmac)

# gmacCompress
cuda_compile(compress gmacCompressCommon.cu gmacCompress.cu)
add_executable(gmacCompress ${compress}  gmacCompressCommon.cu gmacCompress.cu)
target_link_libraries(gmacCompress test-common gmac)

# gmacCompress
cuda_compile(compressSend gmacCompressCommon.cu gmacCompressSend.cu)
add_executable(gmacCompressSend ${compressSend}  gmacCompressCommon.cu gmacCompressSend.cu)
target_link_libraries(gmacCompressSend test-common gmac)
endif(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")


#install(TARGETS gmacVecAdd DESTINATION ${CMAKE_INSTALL_PREFIX}/share/test-common gmac)
#install(TARGETS gmacThreadVecAdd DESTINATION ${CMAKE_INSTALL_PREFIX}/share/test-common gmac)
#install(TARGETS gmacSharedVecAdd DESTINATION ${CMAKE_INSTALL_PREFIX}/share/test-common gmac)
#install(TARGETS gmacFileVecAdd DESTINATION ${CMAKE_INSTALL_PREFIX}/share/test-common gmac)
#install(TARGETS gmacMatrixMul DESTINATION ${CMAKE_INSTALL_PREFIX}/share/test-common gmac)
#install(TARGETS gmacSharedMatrixMul DESTINATION ${CMAKE_INSTALL_PREFIX}/share/test-common gmac)
#install(TARGETS gmacMemcpy DESTINATION ${CMAKE_INSTALL_PREFIX}/share/test-common gmac)
#install(TARGETS gmacMemset DESTINATION ${CMAKE_INSTALL_PREFIX}/share/test-common gmac)
#install(TARGETS gmacFile DESTINATION ${CMAKE_INSTALL_PREFIX}/share/test-common gmac)
#install(TARGETS gmacMigration DESTINATION ${CMAKE_INSTALL_PREFIX}/share/test-common gmac)
#install(TARGETS gmacMeminfo DESTINATION ${CMAKE_INSTALL_PREFIX}/share/test-common gmac)
#install(TARGETS gmacTexture DESTINATION ${CMAKE_INSTALL_PREFIX}/share/test-common gmac)
#install(TARGETS gmacConstant DESTINATION ${CMAKE_INSTALL_PREFIX}/share/test-common gmac)

#if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
#install(TARGETS gmacCompress DESTINATION ${CMAKE_INSTALL_PREFIX}/share/test-common gmac)
#install(TARGETS gmacCompressSend DESTINATION ${CMAKE_INSTALL_PREFIX}/share/test-common gmac)
#endif(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")

file(COPY vars.spec tests.spec DESTINATION ${PROJECT_BINARY_DIR})

# gmacMap
# source_group(${current_GROUP} FILES gmacMap.cu)
# cuda_compile(map gmacMap.cu)
# add_executable(gmacMap ${map}  gmacMap.cu)
# target_link_libraries(gmacMap test-common gmac)

add_custom_target(InputSets ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../../input
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/inputset)


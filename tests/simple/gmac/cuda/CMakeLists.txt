set(common_ROOT ${CMAKE_SOURCE_DIR}/tests/common)

include_directories(${PROJECT_SOURCE_DIR}/src/include ${PROJECT_BINARY_DIR}/src/include ${common_ROOT})

include(FindCUDA)

set(common_SRC
     ${common_ROOT}/debug.h
     ${common_ROOT}/utils.h ${common_ROOT}/utils-impl.h ${common_ROOT}/utils.cpp 
     ${common_ROOT}/barrier.h
     ${common_ROOT}/cycle.h)
set(common_os_SRC
        ${common_ROOT}/${OS_DIR}/utils.cpp
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
    gmacAlloc.cu
    gmacVecAdd.cu
    gmacThreadVecAdd.cu
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
    gmacStencilCommon.cu
    gmacConstant.cu
    gmacTexture.cu
    gmacThreadStencil.cu
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
add_executable(gmacAlloc ${alloc} ${common_SRC} gmacAlloc.cu)
target_link_libraries(gmacAlloc gmac)

# gmacVecAdd
cuda_compile(vecAdd gmacVecAdd.cu)
add_executable(gmacVecAdd ${vecAdd} ${common_SRC} gmacVecAdd.cu)
target_link_libraries(gmacVecAdd gmac)

# gmacThreadVecAdd
cuda_compile(threadVecAdd gmacThreadVecAdd.cu)
add_executable(gmacThreadVecAdd ${threadVecAdd} ${common_SRC} gmacThreadVecAdd.cu)
target_link_libraries(gmacThreadVecAdd gmac)

# gmacSharedVecAdd
cuda_compile(sharedVecAdd gmacSharedVecAdd.cu)
add_executable(gmacSharedVecAdd ${sharedVecAdd} ${common_SRC} gmacSharedVecAdd.cu)
target_link_libraries(gmacSharedVecAdd gmac)

# gmacFileVecAdd
cuda_compile(fileVecAdd gmacFileVecAdd.cu)
add_executable(gmacFileVecAdd ${fileVecAdd} ${common_SRC} gmacFileVecAdd.cu)
target_link_libraries(gmacFileVecAdd gmac)

# gmacMatrixMul
cuda_compile(matrixMul gmacMatrixMul.cu)
add_executable(gmacMatrixMul ${matrixMul} ${common_SRC} gmacMatrixMul.cu)
target_link_libraries(gmacMatrixMul gmac)

# gmacSharedMatrixMul
cuda_compile(sharedMatrixMul gmacSharedMatrixMul.cu)
add_executable(gmacSharedMatrixMul ${sharedMatrixMul} ${common_SRC} gmacSharedMatrixMul.cu)
target_link_libraries(gmacSharedMatrixMul gmac)


# gmacStencil
cuda_compile(stencil gmacStencil.cu)
add_executable(gmacStencil ${stencil} ${common_SRC} gmacStencil.cu gmacStencilCommon.cu)
target_link_libraries(gmacStencil gmac)

# gmacThreadStencil
cuda_compile(threadStencil gmacThreadStencil.cu)
add_executable(gmacThreadStencil ${threadStencil} ${common_SRC} gmacThreadStencil.cu gmacStencilCommon.cu)
target_link_libraries(gmacThreadStencil gmac)

# gmacMemcpy
cuda_compile(memcpy gmacMemcpy.cu)
add_executable(gmacMemcpy ${memcpy} ${common_SRC} gmacMemcpy.cu)
target_link_libraries(gmacMemcpy gmac)

# gmacMemset
cuda_compile(memset gmacMemset.cu)
add_executable(gmacMemset ${memset} ${common_SRC} gmacMemset.cu)
target_link_libraries(gmacMemset gmac)

# gmacFile
cuda_compile(gfile gmacFile.cu)
add_executable(gmacFile ${gfile} ${common_SRC} gmacFile.cu)
target_link_libraries(gmacFile gmac)

# gmacMigration
cuda_compile(migration gmacMigration.cu)
add_executable(gmacMigration ${migration} ${common_SRC} gmacMigration.cu)
target_link_libraries(gmacMigration gmac)

# gmacMeminfo
cuda_compile(meminfo gmacMeminfo.cu)
add_executable(gmacMeminfo ${meminfo} ${common_SRC} gmacMeminfo.cu)
target_link_libraries(gmacMeminfo gmac)

# gmacTexture
cuda_compile(texture gmacTexture.cu)
add_executable(gmacTexture ${texture} ${common_SRC} gmacTexture.cu)
target_link_libraries(gmacTexture gmac)

# gmacConstant
cuda_compile(constant gmacConstant.cu)
add_executable(gmacConstant ${constant} ${common_SRC} gmacConstant.cu)
target_link_libraries(gmacConstant gmac)

install(TARGETS gmacVecAdd DESTINATION ${CMAKE_INSTALL_PREFIX}/share/gmac)
install(TARGETS gmacThreadVecAdd DESTINATION ${CMAKE_INSTALL_PREFIX}/share/gmac)
install(TARGETS gmacSharedVecAdd DESTINATION ${CMAKE_INSTALL_PREFIX}/share/gmac)
install(TARGETS gmacFileVecAdd DESTINATION ${CMAKE_INSTALL_PREFIX}/share/gmac)
install(TARGETS gmacMatrixMul DESTINATION ${CMAKE_INSTALL_PREFIX}/share/gmac)
install(TARGETS gmacSharedMatrixMul DESTINATION ${CMAKE_INSTALL_PREFIX}/share/gmac)
install(TARGETS gmacMemcpy DESTINATION ${CMAKE_INSTALL_PREFIX}/share/gmac)
install(TARGETS gmacMemset DESTINATION ${CMAKE_INSTALL_PREFIX}/share/gmac)
install(TARGETS gmacFile DESTINATION ${CMAKE_INSTALL_PREFIX}/share/gmac)
install(TARGETS gmacMigration DESTINATION ${CMAKE_INSTALL_PREFIX}/share/gmac)
install(TARGETS gmacMeminfo DESTINATION ${CMAKE_INSTALL_PREFIX}/share/gmac)

file(COPY vars.spec tests.spec DESTINATION ${PROJECT_BINARY_DIR})

# gmacMap
# source_group(${current_GROUP} FILES gmacMap.cu)
# cuda_compile(map gmacMap.cu)
# add_executable(gmacMap ${map} ${common_SRC} gmacMap.cu)
# target_link_libraries(gmacMap gmac)

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
add_custom_target(InputSets ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../input
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/inputset)
endif(CMAKE_SYSTEM_NAME MATCHES "Windows")


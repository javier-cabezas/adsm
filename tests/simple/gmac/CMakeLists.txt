set(common_ROOT ${CMAKE_SOURCE_DIR}/tests/common)

include_directories(${PROJECT_SOURCE_DIR}/src/include ${PROJECT_SOURCE_DIR}/src ${common_ROOT})

include(FindCUDA)

set(common_SRC
     ${common_ROOT}/debug.h
     ${common_ROOT}/utils.cpp ${common_ROOT}/utils.ipp ${common_ROOT}/utils.h 
     ${common_ROOT}/cycle.h)
set(common_os_SRC
        ${common_ROOT}/${OS_DIR}/utils.cpp)
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
    gmacVecAdd.cu
    gmacThreadVecAdd.cu
    gmacSharedVecAdd.cu
    gmacFileVecAdd.cu
    gmacMatrixMul.cu
    gmacMemcpy.cu
    gmacFile.cu
)

message(STATUS "Size of pointer: ${CMAKE_SIZEOF_VOID_P}")
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(CUDA_64_BIT_DEVICE_CODE ON)
else()
    set(CUDA_64_BIT_DEVICE_CODE OFF)
endif()

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
configure_file(../input/vectorA ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/inputset/vectorA COPYONLY)
configure_file(../input/vectorB ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/inputset/vectorB COPYONLY)
configure_file(../input/vectorC ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/inputset/vectorC COPYONLY)

# gmacFileVecAdd
cuda_compile(fileVecAdd gmacFileVecAdd.cu)
add_executable(gmacFileVecAdd ${fileVecAdd} ${common_SRC} gmacFileVecAdd.cu)
target_link_libraries(gmacFileVecAdd gmac)

# gmacMatrixMul
cuda_compile(matrixMul gmacMatrixMul.cu)
add_executable(gmacMatrixMul ${matrixMul} ${common_SRC} gmacMatrixMul.cu)
target_link_libraries(gmacMatrixMul gmac)

# gmacMemcpy
cuda_compile(memcpy gmacMemcpy.cu)
add_executable(gmacMemcpy ${memcpy} ${common_SRC} gmacMemcpy.cu)
target_link_libraries(gmacMemcpy gmac)

# gmacFile
cuda_compile(gfile gmacFile.cu)
add_executable(gmacFile ${gfile} ${common_SRC})
target_link_libraries(gmacFile gmac)

# gmacMap
source_group(${current_GROUP} FILES gmacMap.cu)
cuda_compile(map gmacMap.cu)
add_executable(gmacMap ${map} ${common_SRC})
target_link_libraries(gmacMap gmac)



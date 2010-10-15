set(common_ROOT ../../common)

include_directories(${PROJECT_SOURCE_DIR}/src/include ${common_ROOT})

include(FindCUDA)

set(common_SRC ${common_ROOT}/debug.h ${common_ROOT}/utils.cpp ${common_ROOT}/utils.ipp ${common_ROOT}/utils.h ${common_ROOT}/cycle.h)

message(STATUS "Size of pointer: ${CMAKE_SIZEOF_VOID_P}")
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(CUDA_64_BIT_DEVICE_CODE ON)
else()
    set(CUDA_64_BIT_DEVICE_CODE OFF)
endif()

# gmacVeacAdd
cuda_compile(vecAdd gmacVecAdd.cu)
add_executable(gmacVecAdd ${vecAdd} ${common_SRC})
target_link_libraries(gmacVecAdd gmac)

# gmacThreadVecAdd
cuda_compile(threadVecAdd gmacThreadVecAdd.cu)
add_executable(gmacThreadVecAdd ${threadVecAdd} ${common_SRC})
target_link_libraries(gmacThreadVecAdd gmac)

# gmacSharedVecAdd
cuda_compile(sharedVecAdd gmacSharedVecAdd.cu)
add_executable(gmacSharedVecAdd ${sharedVecAdd} ${common_SRC})
target_link_libraries(gmacSharedVecAdd gmac)
configure_file(../input/vectorA ${CMAKE_CURRENT_BINARY_DIR}/input/vectorA COPYONLY)
configure_file(../input/vectorB ${CMAKE_CURRENT_BINARY_DIR}/input/vectorB COPYONLY)
configure_file(../input/vectorC ${CMAKE_CURRENT_BINARY_DIR}/input/vectorC COPYONLY)

# gmacFileVecAdd
cuda_compile(fileVecAdd gmacFileVecAdd.cu)
add_executable(gmacFileVecAdd ${fileVecAdd} ${common_SRC})
target_link_libraries(gmacFileVecAdd gmac)

# gmacMatrixMul
cuda_compile(matrixMul gmacMatrixMul.cu)
add_executable(gmacMatrixMul ${matrixMul} ${common_SRC})
target_link_libraries(gmacMatrixMul gmac)


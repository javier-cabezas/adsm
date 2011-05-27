set(common_ROOT ${CMAKE_SOURCE_DIR}/tests/common)

include_directories(${PROJECT_SOURCE_DIR}/src/include ${PROJECT_BINARY_DIR}/src/include ${common_ROOT})

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
    liteVecAdd.cpp
    liteVecAdd_cplusplus.cpp
    liteStencil.cpp
    liteStencilCommon.h
    liteMatrixMul.cpp
    liteMatrixMulKernel.cl
)

add_executable(liteVecAdd ${common_SRC} liteVecAdd.cpp)
target_link_libraries(liteVecAdd gmac-lite)

add_executable(liteVecAdd_cplusplus ${common_SRC} liteVecAdd_cplusplus.cpp)
target_link_libraries(liteVecAdd_cplusplus gmac-lite)

add_executable(liteStencil ${common_SRC} liteStencil.cpp liteStencilCommon.h)
target_link_libraries(liteStencil gmac-lite)

add_executable(liteMatrixMul ${common_SRC} liteMatrixMul.cpp liteMatrixMulKernel.cl)
target_link_libraries(liteMatrixMul gmac-lite)

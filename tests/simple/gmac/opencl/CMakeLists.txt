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
    gmacInit.cpp
    gmacVecAdd.cpp
    gmacVecAdd_cplusplus.cpp
    gmacThreadVecAdd.cpp
    gmacMemcpy.cpp
    gmacMemset.cpp
    gmacFile.cpp
    gmacFileVecAdd.cpp
    gmacSharedVecAdd.cpp
    gmacStencil.cpp
    gmacStencilCommon.h
    gmacMatrixMul.cpp
    gmacMatrixMulKernel.cl
)

add_executable(gmacInit ${common_SRC} gmacInit.cpp)
target_link_libraries(gmacInit ${GMAC_TARGET_NAME})

add_executable(gmacVecAdd ${common_SRC} gmacVecAdd.cpp)
target_link_libraries(gmacVecAdd ${GMAC_TARGET_NAME})

add_executable(gmacVecAdd_cplusplus ${common_SRC} gmacVecAdd_cplusplus.cpp)
target_link_libraries(gmacVecAdd_cplusplus ${GMAC_TARGET_NAME})

add_executable(gmacThreadVecAdd ${common_SRC} gmacThreadVecAdd.cpp)
target_link_libraries(gmacThreadVecAdd ${GMAC_TARGET_NAME})

add_executable(gmacMemcpy ${common_SRC} gmacMemcpy.cpp)
target_link_libraries(gmacMemcpy ${GMAC_TARGET_NAME})

add_executable(gmacMemset ${common_SRC} gmacMemset.cpp)
target_link_libraries(gmacMemset ${GMAC_TARGET_NAME})

add_executable(gmacFile ${common_SRC} gmacFile.cpp)
target_link_libraries(gmacFile ${GMAC_TARGET_NAME})

add_executable(gmacFileVecAdd ${common_SRC} gmacFileVecAdd.cpp)
target_link_libraries(gmacFileVecAdd ${GMAC_TARGET_NAME})

add_executable(gmacSharedVecAdd ${common_SRC} gmacSharedVecAdd.cpp)
target_link_libraries(gmacSharedVecAdd ${GMAC_TARGET_NAME})

add_executable(gmacStencil ${common_SRC} gmacStencil.cpp)
target_link_libraries(gmacStencil ${GMAC_TARGET_NAME})

add_executable(gmacMatrixMul ${common_SRC} gmacMatrixMul.cpp gmacMatrixMulKernel.cl)
target_link_libraries(gmacMatrixMul ${GMAC_TARGET_NAME})

file(COPY vars.spec tests.spec DESTINATION ${PROJECT_BINARY_DIR})

add_custom_target(InputSets ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../../input
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/inputset)


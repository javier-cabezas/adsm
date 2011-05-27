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
    oclInit.cpp
    oclVecAdd.cpp
    oclVecAdd_cplusplus.cpp
    oclThreadVecAdd.cpp
    oclMemcpy.cpp
    oclMemset.cpp
    oclFile.cpp
    oclFileVecAdd.cpp
    oclSharedVecAdd.cpp
    oclStencil.cpp
    oclStencilCommon.h
    oclMatrixMul.cpp
    oclMatrixMulKernel.cl
)

add_executable(oclInit ${common_SRC} oclInit.cpp)
target_link_libraries(oclInit gmac-cl)

add_executable(oclVecAdd ${common_SRC} oclVecAdd.cpp)
target_link_libraries(oclVecAdd gmac-cl)

add_executable(oclVecAdd_cplusplus ${common_SRC} oclVecAdd_cplusplus.cpp)
target_link_libraries(oclVecAdd_cplusplus gmac-cl)

add_executable(oclThreadVecAdd ${common_SRC} oclThreadVecAdd.cpp)
target_link_libraries(oclThreadVecAdd gmac-cl)

add_executable(oclMemcpy ${common_SRC} oclMemcpy.cpp)
target_link_libraries(oclMemcpy gmac-cl)

add_executable(oclMemset ${common_SRC} oclMemset.cpp)
target_link_libraries(oclMemset gmac-cl)

add_executable(oclFile ${common_SRC} oclFile.cpp)
target_link_libraries(oclFile gmac-cl)

add_executable(oclFileVecAdd ${common_SRC} oclFileVecAdd.cpp)
target_link_libraries(oclFileVecAdd gmac-cl)

add_executable(oclSharedVecAdd ${common_SRC} oclSharedVecAdd.cpp)
target_link_libraries(oclSharedVecAdd gmac-cl)

add_executable(oclStencil ${common_SRC} oclStencil.cpp)
target_link_libraries(oclStencil gmac-cl)

add_executable(oclMatrixMul ${common_SRC} oclMatrixMul.cpp oclMatrixMulKernel.cl)
target_link_libraries(oclMatrixMul gmac-cl)

file(COPY vars.spec tests.spec DESTINATION ${PROJECT_BINARY_DIR})

add_custom_target(InputSets ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../../input
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/inputset)


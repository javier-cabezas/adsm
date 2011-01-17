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
    gmacThreadVecAdd.cpp
    gmacMemcpy.cpp
    gmacMemset.cpp
    gmacFile.cpp
    gmacFileVecAdd.cpp
)

add_executable(gmacInit ${common_SRC} gmacInit.cpp)
target_link_libraries(gmacInit gmac)

add_executable(gmacVecAdd ${common_SRC} gmacVecAdd.cpp)
target_link_libraries(gmacVecAdd gmac)

add_executable(gmacThreadVecAdd ${common_SRC} gmacThreadVecAdd.cpp)
target_link_libraries(gmacThreadVecAdd gmac)

add_executable(gmacMemcpy ${common_SRC} gmacMemcpy.cpp)
target_link_libraries(gmacMemcpy gmac)

add_executable(gmacMemset ${common_SRC} gmacMemset.cpp)
target_link_libraries(gmacMemset gmac)

add_executable(gmacFile ${common_SRC} gmacFile.cpp)
target_link_libraries(gmacFile gmac)

add_executable(gmacFileVecAdd ${common_SRC} gmacFileVecAdd.cpp)
target_link_libraries(gmacFileVecAdd gmac)


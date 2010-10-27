include_directories(
    ${gmac_test_INCLUDE}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}/src)

# Some really common sources
set(parameter_SRC
    ${CMAKE_SOURCE_DIR}/src/util/Parameter.h
    ${CMAKE_SOURCE_DIR}/src/util/Parameter-def.h
    ${CMAKE_SOURCE_DIR}/src/util/Parameter.ipp
    ${CMAKE_SOURCE_DIR}/src/util/Parameter.cpp)

set(logger_SRC
    ${parameter_SRC}
    ${CMAKE_SOURCE_DIR}/src/util/Logger.h
    ${CMAKE_SOURCE_DIR}/src/util/Logger.cpp)

set(lock_SRC
    ${logger_SRC}
    ${CMAKE_SOURCE_DIR}/src/util/Lock.h
    ${CMAKE_SOURCE_DIR}/src/util/Lock.cpp
    ${CMAKE_SOURCE_DIR}/src/util/${OS_DIR}/Lock.h
    ${CMAKE_SOURCE_DIR}/src/util/${OS_DIR}/Lock.ipp
    ${CMAKE_SOURCE_DIR}/src/util/${OS_DIR}/Lock.cpp)


add_subdirectory(memory)

add_executable(UnitTests UnitTests.cpp ${unit_SRC} ${lock_SRC})
target_link_libraries(UnitTests ${gmac_test_LIB})

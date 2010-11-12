include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${gmac_test_INCLUDE}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}/src)

add_subdirectory(memory)
add_subdirectory(core)
add_subdirectory(api)

set(all_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/UnitTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/init.h
    ${unit_SRC})
add_gmac_groups(${all_SRC})
add_executable(UnitTests ${all_SRC})
target_link_libraries(UnitTests gmac-static ${gmac_test_LIB})

add_gmac_groups(${gmac_SRC})
add_library(gmac-static STATIC ${gmac_SRC})
target_link_libraries(gmac-static ${gmac_LIBS})

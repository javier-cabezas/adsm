include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${gmac_test_INCLUDE}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}/src)

add_subdirectory(dsm)

set(all_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/UnitTests.cpp
    ${unit_SRC})
add_gmac_groups(${all_SRC})
add_executable(UnitTests ${all_SRC})
target_link_libraries(UnitTests gmac-common gmac-core gmac-dsm ${gmac_test_LIB} ${gmac_LIBS})
set_target_properties(UnitTests PROPERTIES
    COMPILE_DEFINITIONS_DEBUG USE_DBC)

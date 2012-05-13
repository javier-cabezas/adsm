set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUNIT_TEST")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${gmac_test_INCLUDE}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}/src)

set(UNIT_TESTS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(dsm)
add_subdirectory(hal)

add_custom_target(UnitTests
                  DEPENDS UnitTests_dsm UnitTests_hal)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUNIT_TEST_HAL")

add_subdirectory(${API_DIR})

set(hal_SRC
    exec.h
    exec.cpp
    init.h
    init.cpp
)

set(all_SRC
    ${UNIT_TESTS_SOURCE_DIR}/UnitTests.cpp
    ${hal_SRC})

add_gmac_groups(${all_SRC})
add_executable(UnitTests_hal ${all_SRC})
target_link_libraries(UnitTests_hal gmac-hal gmac-common gmac-core gmac-dsm ${gmac_test_LIB} ${gmac_LIBS})
set_target_properties(UnitTests_hal PROPERTIES
    COMPILE_DEFINITIONS_DEBUG USE_DBC)

add_dependencies(UnitTests_hal backend_code)

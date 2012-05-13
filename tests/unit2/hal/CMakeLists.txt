set(hal_SRC
    init.h
    init.cpp
)

set(all_SRC
    ${UNIT_TESTS_SOURCE_DIR}/UnitTests.cpp
    ${hal_SRC})

add_gmac_groups(${all_SRC})
add_executable(UnitTests_hal ${all_SRC})
target_link_libraries(UnitTests_hal gmac-hal gmac-common gmac-core ${gmac_test_LIB} ${gmac_LIBS})
set_target_properties(UnitTests_hal PROPERTIES
    COMPILE_DEFINITIONS_DEBUG USE_DBC)

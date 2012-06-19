set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUNIT_TEST_DSM")

set(dsm_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/mapping.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mapping.cpp
    ${CMAKE_SOURCE_DIR}/src/dsm/mapping.h
    ${CMAKE_SOURCE_DIR}/src/dsm/mapping-impl.h
    ${CMAKE_SOURCE_DIR}/src/dsm/mapping.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/manager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/manager.cpp
    ${CMAKE_SOURCE_DIR}/src/dsm/manager.h
    ${CMAKE_SOURCE_DIR}/src/dsm/manager-impl.h
    ${CMAKE_SOURCE_DIR}/src/dsm/manager.cpp

    ${CMAKE_SOURCE_DIR}/src/dsm/coherence/block.h
    ${CMAKE_SOURCE_DIR}/src/dsm/coherence/block-impl.h
    ${CMAKE_SOURCE_DIR}/src/dsm/coherence/block.cpp
)  

set(all_SRC
    ${UNIT_TESTS_SOURCE_DIR}/UnitTests.cpp
    ${dsm_SRC})

add_gmac_groups(${all_SRC})
add_executable(UnitTests_dsm ${all_SRC})
target_link_libraries(UnitTests_dsm gmac-common gmac-core ${gmac_test_LIB} ${gmac_LIBS})
set_target_properties(UnitTests_dsm PROPERTIES
    COMPILE_DEFINITIONS_DEBUG USE_DBC)

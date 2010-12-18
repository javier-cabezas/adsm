set(memory_test_SRC
    Block.h Block.cpp
    Manager.h Manager.cpp
    Object.h Object.cpp
)

add_gmac_sources(gmac-dbc ${memory_test_SRC})

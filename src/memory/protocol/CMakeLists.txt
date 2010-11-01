add_subdirectory(dbc)

set(memory_protocol_SRC
    Lazy.h Lazy.cpp)

add_gmac_sources(gmac ${memory_protocol_SRC})
add_gmac_sources(gmac-dbc)

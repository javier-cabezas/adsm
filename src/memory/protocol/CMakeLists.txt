add_subdirectory(test)

set(memory_protocol_SRC
    Lazy.h Lazy.cpp)
add_gmac_sources(gmac ${memory_protocol_SRC})

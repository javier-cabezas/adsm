set(core_SRC
    thread.h
    thread-impl.h
    thread.cpp
)

set(core_DBC
)

add_gmac_sources(gmac-core ${core_SRC} ${core_DBC})

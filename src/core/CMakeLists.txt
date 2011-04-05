add_subdirectory(dbc)

add_subdirectory(allocator)
add_subdirectory(hpe)

set(core_SRC
    Process.h Mode.h
    IOBuffer.h IOBuffer-impl.h)

add_gmac_sources(gmac ${core_SRC})
add_gmac_sources(gmac-dbc)

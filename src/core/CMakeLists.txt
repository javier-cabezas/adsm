add_subdirectory(hpe)

set(core_SRC
    Process.h
    Process.cpp
    Mode.h
    IOBuffer.h
    IOBuffer-impl.h
    allocator/Buddy.h
    allocator/Buddy.cpp
)

set(core_DBC
    dbc/IOBuffer.h
    dbc/IOBuffer.cpp
)

add_gmac_sources(gmac-core ${core_SRC} ${core_DBC})
add_gmac_sources(gmac-core-hpe)

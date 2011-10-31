add_subdirectory(hpe)

set(core_SRC
    io_buffer.h
    io_buffer-impl.h
    vdevice.h
    vdevice-impl.h
    process.h
    process.cpp
    thread.h
    thread-impl.h
    thread.cpp
)

set(core_DBC
    dbc/AllocationMap.h
    dbc/AllocationMap.cpp
    dbc/io_buffer.h
    dbc/io_buffer.cpp
)

add_gmac_sources(gmac-core ${core_SRC} ${core_DBC})
add_gmac_sources(gmac-core-hpe)

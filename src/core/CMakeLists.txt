add_subdirectory(dbc)

add_subdirectory(allocator)
add_subdirectory(hpe)

set(core_SRC
    Accelerator.h Accelerator-impl.h Accelerator.cpp
    IOBuffer.h IOBuffer-impl.h)

add_gmac_sources(gmac ${core_SRC})
add_gmac_sources(gmac-dbc)

add_subdirectory(dbc)
add_subdirectory(memory)

add_subdirectory(hpe)

set(api_SRC
    IOBuffer.h IOBuffer-impl.h IOBuffer.cpp
    Tracer.h Tracer-impl.h)

add_gmac_sources(gmac ${api_SRC})
add_gmac_sources(gmac-lib ${api_lib_SRC})
add_gmac_sources(gmac-dbc)

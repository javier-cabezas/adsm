add_subdirectory(dbc)

add_subdirectory(hpe)
add_subdirectory(gmac)
add_subdirectory(memory)

set(api_lib_SRC
    opencl.cpp)

set(api_SRC
    IOBuffer.h IOBuffer-impl.h
    Tracer.h Tracer-impl.h
   )

add_gmac_sources(gmac ${api_SRC})
add_gmac_sources(gmac-lib ${api_lib_SRC})
add_gmac_sources(gmac-dbc)

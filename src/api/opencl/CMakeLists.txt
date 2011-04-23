add_subdirectory(hpe)
add_subdirectory(lite)

set(arch_SRC
    IOBuffer.h
    IOBuffer-impl.h
    Tracer.h
    Tracer-impl.h
)

add_gmac_sources(gmac-arch ${arch_SRC})

add_gmac_sources(gmac-arch-hpe)
add_gmac_sources(gmac-arch-hpe-api)

if(MAKE_LITE)
    add_gmac_sources(gmac-arch-lite)
    add_gmac_sources(gmac-arch-lite-api)
endif(MAKE_LITE)

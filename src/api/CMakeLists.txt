add_subdirectory(${API_DIR})

add_gmac_sources(gmac-arch)
add_gmac_sources(gmac-arch-hpe)
add_gmac_sources(gmac-arch-hpe-api)

if(MAKE_LITE)
    add_gmac_sources(gmac-arch-lite)
    add_gmac_sources(gmac-arch-lite-api)
endif(MAKE_LITE)

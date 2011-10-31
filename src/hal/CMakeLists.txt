add_subdirectory(${API_DIR})

set(hal_SRC
    coherence_domain.h
    coherence_domain-impl.h
    device.h
    device-impl.h
    types.h
    types-detail.h
    types-impl.h
)

add_gmac_sources(gmac-hal ${hal_SRC})
add_gmac_sources(gmac-hal-api)
add_gmac_sources(gmac-hal-api-hpe)

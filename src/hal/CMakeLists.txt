add_subdirectory(${API_DIR})

set(hal_SRC
    coherence_domain.h
    coherence_domain-impl.h
    context.h
    context-impl.h
    device.h
    device-impl.h
    event.h
    event-impl.h
    kernel.h
    kernel-impl.h
    ptr.h
    stream.h
    stream-impl.h
    types.h
    types-detail.h
    types-impl.h
)

add_gmac_sources(gmac-hal ${hal_SRC})
add_gmac_sources(gmac-hal-api)
add_gmac_sources(gmac-hal-api-hpe)

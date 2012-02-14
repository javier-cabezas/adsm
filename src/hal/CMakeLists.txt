add_subdirectory(cpu)
add_subdirectory(${API_DIR})

set(hal_SRC
    coherence_domain.h
    coherence_domain-impl.h
    aspace.h
    aspace.cpp
    aspace-impl.h
    copy.cpp
    device.h
    device.cpp
    device-impl.h
    event.h
    event-impl.h
    kernel.h
    kernel-impl.h
    ptr.h
    ptr-impl.h
    stream.h
    stream-impl.h
    types.h
    types-detail.h
)

add_gmac_sources(gmac-hal ${hal_SRC})
add_gmac_sources(gmac-hal-api)

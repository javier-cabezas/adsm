add_subdirectory(hpe)

set(hal_cuda_SRC
    coherence_domain.h
    coherence_domain.cpp
    device.h
    device.cpp
    context.h
    context-impl.h
    context.cpp
    event.h
    event-impl.h
    event.cpp
    kernel.h
    kernel-impl.h
    module.h
    module-impl.h
    module.cpp
    ptr.h
    stream.h
    stream-impl.h
    types.h
    types.cpp
)

add_gmac_sources(gmac-hal-api ${hal_cuda_SRC})
add_gmac_sources(gmac-hal-api-hpe)

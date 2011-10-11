set(hal_cuda_SRC
    coherence_domain.h
    coherence_domain.cpp
    device.h
    device.cpp
    types.h
    types-impl.h
)

add_gmac_sources(gmac-hal-api ${hal_cuda_SRC})

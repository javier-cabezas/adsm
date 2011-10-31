add_subdirectory(hpe)

set(hal_cuda_SRC
    coherence_domain.h
    coherence_domain.cpp
    device.h
    device.cpp
    module.h
    module-impl.h
    module.cpp
    types.h
    types-impl.h
    types.cpp
)

add_gmac_sources(gmac-hal-api ${hal_cuda_SRC})
add_gmac_sources(gmac-hal-api-hpe)

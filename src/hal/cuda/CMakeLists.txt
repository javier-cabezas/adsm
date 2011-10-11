set(hal_cuda_SRC
    types.h
    types-impl.h
    CoherenceDomain.h
    CoherenceDomain.cpp
    Device.h
    Device.cpp
)

add_gmac_sources(gmac-hal-api ${hal_cuda_SRC})

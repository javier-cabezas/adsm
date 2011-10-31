set(hal_opencl_SRC
    coherence_domain.h
    coherence_domain.cpp
    device.h
    device.cpp
    types.h
    types-impl.h
    util/opencl_util.h
    util/opencl_util-impl.h
    util/opencl_util.cpp
)

add_gmac_sources(gmac-hal-api ${hal_opencl_SRC})

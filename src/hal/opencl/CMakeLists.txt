set(hal_opencl_SRC
    coherence_domain.h
    coherence_domain.cpp
    device.h
    device.cpp
    types.h
    types-impl.h
    types.cpp
    helper/opencl_helper.h
    helper/opencl_helper-impl.h
    helper/opencl_helper.cpp
)

add_gmac_sources(gmac-hal-api ${hal_opencl_SRC})

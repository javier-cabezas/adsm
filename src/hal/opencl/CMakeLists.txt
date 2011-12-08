configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hpe/gmac.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/hpe/gmac.cpp)

set(hal_opencl_SRC
    coherence_domain.h
    coherence_domain.cpp
    context.h
    context-impl.h
    context.cpp
    device.h
    device.cpp
    event.h
    event-impl.h
    kernel.h
    kernel-impl.h
    kernel.cpp
    module.h
    module-impl.h
    module.cpp
    ptr.h
    stream.h
    stream-impl.h
    types.h
    types-impl.h
    types.cpp
    helper/opencl_helper.h
    helper/opencl_helper-impl.h
    helper/opencl_helper.cpp
    kernels/memset.cl
    kernels/kernels.h
)

set(hal_opencl_hpe_SRC
    hpe/gmac.cpp
)

add_gmac_sources(gmac-hal-api ${hal_opencl_SRC})
add_gmac_sources(gmac-hal-api-hpe ${hal_opencl_hpe_SRC})

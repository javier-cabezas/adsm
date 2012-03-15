set(hal_cuda_SRC
    phys/aspace.h
    phys/aspace.cpp
    phys/processing_unit.h
    phys/processing_unit.cpp
    phys/platform.h
    virt/aspace.h
    virt/aspace-impl.h
    virt/aspace.cpp
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

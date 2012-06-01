set(hal_cuda_SRC
    phys/aspace.h
    phys/aspace.cpp
    phys/processing_unit.h
    phys/processing_unit.cpp
    phys/platform.h
    virt/aspace.h
    virt/aspace-impl.h
    virt/aspace.cpp
    virt/context.h
    virt/context-impl.h
    virt/scheduler.h
    code/module.h
    code/module-impl.h
    code/module.cpp
    kernel.h
    kernel-impl.h
    operation.h
    operation-impl.h
    operation.cpp
    stream.h
    stream-impl.h
    types.h
    types.cpp
)

add_gmac_sources(gmac-hal-api ${hal_cuda_SRC})
add_gmac_sources(gmac-hal-api-hpe)

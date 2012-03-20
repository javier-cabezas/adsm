add_subdirectory(cpu)
add_subdirectory(${API_DIR})

set(hal_SRC
    detail/virt/aspace.h
    detail/virt/aspace.cpp
    detail/virt/aspace-impl.h
    detail/virt/object.h
    detail/virt/object-impl.h
    detail/virt/object.cpp
    detail/virt/processing_unit.h
    detail/virt/processing_unit.cpp
    detail/virt/processing_unit-impl.h
    detail/event.h
    detail/event-impl.h
    detail/kernel.h
    detail/kernel-impl.h
    detail/phys/aspace.h
    detail/phys/aspace-impl.h
    detail/phys/processing_unit.h
    detail/phys/processing_unit.cpp
    detail/phys/processing_unit-impl.h
    detail/phys/memory.h
    detail/phys/memory-impl.h
    detail/phys/platform.h
    detail/phys/platform.cpp
    detail/phys/platform-impl.h
    detail/ptr.h
    detail/ptr-impl.h
    detail/stream.h
    detail/stream-impl.h
    detail/types.h
    copy.cpp
    types.h
)

add_gmac_sources(gmac-hal ${hal_SRC})
add_gmac_sources(gmac-hal-api)

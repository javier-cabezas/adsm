set(hal_cpu_SRC
    phys/aspace.h
    phys/aspace.cpp
    phys/processing_unit.h
    phys/processing_unit.cpp
    virt/aspace.h
    virt/aspace.cpp
    virt/map_file.h
    virt/map_file.cpp
)

add_gmac_sources(gmac-hal-cpu ${hal_cpu_SRC})

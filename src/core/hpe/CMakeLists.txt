add_subdirectory(dbc)

set(core_SRC
    Context.h Context.cpp
    Descriptor.h Descriptor-impl.h
    Kernel.h Kernel-impl.h
    Map.h Map.cpp)

add_gmac_sources(gmac ${core_SRC})
add_gmac_sources(gmac-dbc)

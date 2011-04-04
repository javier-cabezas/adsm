add_subdirectory(dbc)

set(core_SRC
    Context.h Context.cpp
    Descriptor.h Descriptor-impl.h)

add_gmac_sources(gmac ${core_SRC})
add_gmac_sources(gmac-dbc)

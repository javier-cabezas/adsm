add_subdirectory(dbc)

set(core_hpe_SRC
    Context.h Context.cpp
    Descriptor.h Descriptor-impl.h
    Kernel.h Kernel-impl.h
    Map.h Map.cpp
    Mode.h Mode-impl.h Mode.cpp
    Process.h Process-impl.h Process.cpp
    Queue.h Queue.cpp)

add_gmac_sources(gmac ${core_hpe_SRC})
add_gmac_sources(gmac-dbc)

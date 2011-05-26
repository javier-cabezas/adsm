set(arch_hpe_api_SRC
    gmac.cpp
    compat.cpp
)

set(arch_hpe_SRC
    opencl.cpp
    Accelerator.h
    Accelerator-impl.h
    Accelerator.cpp
    Context.h
    Context-impl.h
    Context.cpp
    ContextFactory.h
    ContextFactory.cpp
    Error.cpp
    Kernel.h
    Kernel-impl.h
    Kernel.cpp
    Mode.h
    Mode-impl.h
    Mode.cpp
    ModeFactory.h
    ModeFactory.cpp
)

set(arch_hpe_DBC
    dbc/Accelerator.h   
    dbc/Accelerator.cpp
    dbc/Mode.h
    dbc/Mode.cpp
)

add_gmac_sources(gmac-arch-hpe ${arch_hpe_SRC} ${arch_hpe_DBC})
add_gmac_sources(gmac-arch-hpe-api ${arch_hpe_api_SRC})

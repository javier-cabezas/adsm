add_subdirectory(dbc)

set(api_lib_SRC
    opencl.cpp)

set(api_SRC
    Accelerator.h Accelerator-impl.h Accelerator.cpp
    Context.h Context-impl.h Context.cpp
    Error.cpp
    Kernel.h Kernel-impl.h Kernel.cpp
    Mode.h Mode-impl.h Mode.cpp
   )

add_gmac_sources(gmac ${api_SRC})
add_gmac_sources(gmac-lib ${api_lib_SRC})
add_gmac_sources(gmac-dbc)

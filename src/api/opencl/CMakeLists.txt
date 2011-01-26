add_subdirectory(dbc)

add_subdirectory(gmac)
add_subdirectory(memory)

set(api_lib_SRC
    opencl.cpp)

set(api_SRC
    IOBuffer.h IOBuffer-impl.h
    Context.h Context-impl.h Context.cpp
    Kernel.h Kernel-impl.h Kernel.cpp
    Mode.h Mode-impl.h Mode.cpp
    Accelerator.h Accelerator-impl.h Accelerator.cpp Error.cpp
   )

add_gmac_sources(gmac ${api_SRC})
add_gmac_sources(gmac-lib ${api_lib_SRC})
add_gmac_sources(gmac-dbc)

add_subdirectory(dbc)

add_subdirectory(memory)

set(api_lib_SRC
    cuda.cpp driver.cpp compat.cpp)
set(api_SRC
    Accelerator.h Accelerator-impl.h Accelerator.cpp
    Error.cpp
    IOBuffer.h IOBuffer-impl.h IOBuffer.cpp
    Kernel.h Kernel-impl.h Kernel.cpp
    Mode.h Mode-impl.h Mode.cpp
    Context.h Context-impl.h Context.cpp
    Module.h Module-impl.h Module.cpp
    Tracer.h Tracer-impl.h)

add_gmac_sources(gmac ${api_SRC})
add_gmac_sources(gmac-lib ${api_lib_SRC})
add_gmac_sources(gmac-dbc)

add_subdirectory(memory)

set(api_lib_SRC
    cuda.cpp driver.cpp compat.cpp)
set(api_SRC
    Accelerator.h Accelerator.ipp Accelerator.cpp
    Error.cpp
    Kernel.h Kernel.cpp
    Mode.h Mode.ipp Mode.cpp
    Context.h Context.ipp Context.cpp
    Module.h Module.ipp Module.cpp)

add_gmac_sources(gmac ${api_SRC})
add_gmac_sources(gmac-lib ${api_lib_SRC})

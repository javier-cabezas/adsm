add_subdirectory(memory)
add_subdirectory(allocator)

set(api_SRC
    cuda.cpp driver.cpp compat.cpp
    Accelerator.h Accelerator.ipp Accelerator.cpp
    Error.cpp
    Kernel.h Kernel.cpp
    Mode.h Mode.ipp Mode.cpp
    Context.h Context.ipp Context.cpp
    Module.h Module.ipp Module.cpp)

add_gmac_sources(${api_SRC})

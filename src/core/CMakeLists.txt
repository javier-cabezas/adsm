add_subdirectory(allocator)

set(core_SRC
    Process.h Process.ipp Process.cpp
    Accelerator.h Accelerator.ipp Accelerator.cpp
    Descriptor.h
    Kernel.h Kernel.cpp
    IOBuffer.h
    Mode.h Mode.ipp Mode.cpp
    Context.h Context.cpp
    Queue.h Queue.ipp Queue.cpp)

add_gmac_sources(${core_SRC})

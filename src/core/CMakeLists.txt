add_subdirectory(allocator)

set(core_SRC
    Process.h Process-impl.h Process.cpp
    Accelerator.h Accelerator.ipp Accelerator.cpp
    Descriptor.h Descriptor.ipp
    Kernel.h Kernel.ipp Kernel.cpp
    IOBuffer.h IOBuffer.ipp
    Mode.h Mode-impl.h Mode.cpp
    Context.h Context.cpp
    Queue.h Queue.ipp Queue.cpp)

add_gmac_sources(gmac ${core_SRC})

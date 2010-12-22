add_subdirectory(allocator)

set(core_SRC
    Process.h Process-impl.h Process.cpp
    Accelerator.h Accelerator-impl.h Accelerator.cpp
    Descriptor.h Descriptor-impl.h
    Kernel.h Kernel-impl.h Kernel.cpp
    IOBuffer.h IOBuffer-impl.h
    Mode.h Mode-impl.h Mode.cpp
    Context.h Context.cpp
    Queue.h Queue.cpp)

add_gmac_sources(gmac ${core_SRC})

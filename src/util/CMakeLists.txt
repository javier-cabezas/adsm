add_subdirectory(${OS_DIR})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Parameter-def.h.in ${CMAKE_CURRENT_BINARY_DIR}/Parameter-def.h)

set(util_SRC
    atomics.h
    FileLock.h
    delayed_exec.h
    descriptor.h
    descriptor-impl.h
    file.h
    file.cpp
    gmac_base.h
    gmac_base.cpp
    lock.h
    lock-impl.h
    lock.cpp
    locked_counter.h
    locked_iterator.h
    locked_iterator-impl.h
    locked_object.h
    NonCopyable.h
    Parameter.h
    Parameter-impl.h
    Parameter.cpp
    Parameter-def.h
    private.h
    Reference.h
    Reference-impl.h
    ReusableObject.h
    semaphore.h
    Shared.h
    singleton.h
    singleton-impl.h
    thread.h
    unique.h
    unique-impl.h
    smart_ptr.h
    allocator/Buddy.h
    allocator/Buddy.cpp
    stl/locked_map.h
)

set(util_DBC
    allocator/dbc/Buddy.h
    allocator/dbc/Buddy.cpp
)

add_gmac_sources(gmac-util ${util_SRC} ${util_DBC})

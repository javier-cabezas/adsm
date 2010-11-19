add_subdirectory(dbc)

add_subdirectory(protocol)
add_subdirectory(allocator)
add_subdirectory(${OS_DIR})

set(memory_SRC
    memory.cpp Memory.h
    Bitmap.h Bitmap.ipp Bitmap.cpp
    Block.h Block-impl.h
    StateBlock.h StateBlock-impl.h
    SharedBlock.h SharedBlock-impl.h
    DistributedBlock.h DistributedBlock-impl.h
    Object.h Object-impl.h Object.cpp
    SharedObject.h SharedObject-impl.h
    DistributedObject.h DistributedObject-impl.h
    Map.h Map.cpp
    Protocol.h Protocol.cpp
    Manager.h Manager-impl.h Manager.cpp
    Handler.h Handler.cpp
    Allocator.h Allocator.cpp)

add_gmac_sources(gmac ${memory_SRC})
add_gmac_sources(gmac-lib ${memory_lib_SRC})
add_gmac_sources(gmac-dbc)

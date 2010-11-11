add_subdirectory(dbc)

add_subdirectory(protocol)
add_subdirectory(allocator)
add_subdirectory(${OS_DIR})

set(memory_SRC
    memory.cpp Memory.h
    AcceleratorBlock.h AcceleratorBlock.ipp
    Bitmap.h Bitmap.ipp Bitmap.cpp
    Block.h Block.ipp
    SystemBlock.h SystemBlock.ipp
    Object.h Object.ipp Object.cpp
    OrphanObject.h OrphanObject.ipp
    ObjectSet.h
    StateObject.h StateObject.ipp
    SharedObject.h SharedObject.ipp
    DistributedObject.h DistributedObject.cpp
    ReplicatedObject.h ReplicatedObject.ipp
    CentralizedObject.h CentralizedObject.ipp
    Map.h Map.cpp
    Protocol.h Protocol.cpp
    Manager.h Manager.cpp Manager.ipp
    Handler.h Handler.cpp
    Allocator.h Allocator.cpp)

add_gmac_sources(gmac ${memory_SRC})
add_gmac_sources(gmac-lib ${memory_lib_SRC})
add_gmac_sources(gmac-dbc)

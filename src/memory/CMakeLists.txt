add_subdirectory(test)

add_subdirectory(protocol)
add_subdirectory(allocator)
add_subdirectory(${OS_DIR})

set(memory_SRC
    memory.cpp
    Bitmap.h Bitmap.ipp Bitmap.cpp
    Block.h Block.ipp
    Object.h Object.ipp Object.cpp
    ObjectSet.h
    StateObject.h StateObject.ipp
    SharedObject.h SharedObject.ipp
    DistributedObject.h
    ReplicatedObject.h ReplicatedObject.ipp
    CentralizedObject.h
    Map.h Map.cpp
    Protocol.h Protocol.cpp
    Manager.h Manager.cpp Manager.ipp
    Handler.h Handler.cpp
    Allocator.h)

add_gmac_sources(${memory_SRC})

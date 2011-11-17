add_subdirectory(vm)
add_subdirectory(${OS_DIR})

set(memory_SRC
    Allocator.h
    Allocator.cpp
    Block.h
    Block-impl.h
    Block.cpp
    BlockGroup.h
    BlockGroup-impl.h
    GenericBlock.h
    GenericBlock-impl.h
    Handler.h
    Handler.cpp
    HostMappedObject.h
    HostMappedObject-impl.h
    HostMappedObject.cpp
    manager.h
    manager-impl.h
    manager.cpp
    Memory.h
    object.h
    object-impl.h
    object.cpp
    map_object.h
    map_object-impl.h
    map_object.cpp
    Protocol.h
    Protocol.cpp
    memory.cpp
    allocator/Cache.h
    allocator/Cache-impl.h
    allocator/Cache.cpp
    allocator/Slab.h
    allocator/Slab-impl.h
    allocator/Slab.cpp
    protocol/BlockList.h
    protocol/BlockList-impl.h
    protocol/Gather.h
    protocol/Gather-impl.h
    protocol/Gather.cpp
    protocol/LazyBase.h
    protocol/Lazy.h
    protocol/Lazy-impl.h
    protocol/Lazy.cpp
    protocol/common/BlockList.h
    protocol/common/BlockList-impl.h
    protocol/common/BlockState.h
    protocol/common/BlockState-impl.h
    protocol/common/BlockState.cpp
    protocol/lazy/BlockState.h
    protocol/lazy/BlockState-impl.h
    protocol/lazy/LazyTypes.h
)

set(memory_DBC
    dbc/Block.h
    dbc/Block.cpp
    dbc/Manager.h
    dbc/Manager.cpp
    protocol/dbc/Lazy.h
    protocol/dbc/Lazy-impl.h
    protocol/dbc/LazyBase.h
    protocol/dbc/Lazy.cpp
)

# dbc/object.h
# dbc/object.cpp

add_gmac_sources(gmac-memory ${memory_SRC} ${memory_DBC})

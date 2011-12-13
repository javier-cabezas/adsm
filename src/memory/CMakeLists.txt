add_subdirectory(vm)
add_subdirectory(${OS_DIR})

set(memory_SRC
    allocator.h
    block.h
    block-impl.h
    block.cpp
    block_state.h
    block_state-impl.h
    handler.h
    handler.cpp
    object_mapped.h
    object_mapped-impl.h
    object_mapped.cpp
    manager.h
    manager-impl.h
    manager.cpp
    memory.h
    object.h
    object-impl.h
    object.cpp
    object_state.h
    object_state-impl.h
    map_object.h
    map_object-impl.h
    map_object.cpp
    protocol.h
    protocol.cpp
    memory.cpp
    allocator/cache.h
    allocator/cache-impl.h
    allocator/cache.cpp
    allocator/slab.h
    allocator/slab-impl.h
    allocator/slab.cpp
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
    dbc/block.h
    dbc/block.cpp
    dbc/manager.h
    dbc/manager.cpp
    protocol/dbc/Lazy.h
    protocol/dbc/Lazy-impl.h
    protocol/dbc/LazyBase.h
    protocol/dbc/Lazy.cpp
)

# dbc/object.h
# dbc/object.cpp

add_gmac_sources(gmac-memory ${memory_SRC} ${memory_DBC})

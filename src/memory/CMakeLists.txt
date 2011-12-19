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
    protocol/Gather.h
    protocol/Gather-impl.h
    protocol/Gather.cpp
    protocol/lazy_base.h
    protocol/lazy.h
    protocol/lazy-impl.h
    protocol/lazy.cpp
    protocol/common/list_block.h
    protocol/common/list_block-impl.h
    protocol/common/block_state.h
    protocol/common/block_state-impl.h
    protocol/common/block_state.cpp
    protocol/lazy/block_state.h
    protocol/lazy/block_state-impl.h
    protocol/lazy/lazy_types.h
)

set(memory_DBC
    dbc/block.h
    dbc/block.cpp
    dbc/manager.h
    dbc/manager.cpp
    protocol/dbc/lazy.h
    protocol/dbc/lazy-impl.h
    protocol/dbc/lazy_base.h
    protocol/dbc/lazy.cpp
)

# dbc/object.h
# dbc/object.cpp

add_gmac_sources(gmac-memory ${memory_SRC} ${memory_DBC})

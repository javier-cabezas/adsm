set(memory_allocator_SRC
    Cache.h Cache.ipp Cache.cpp
    Slab.h Slab.ipp Slab.cpp)
add_gmac_sources(${memory_allocator_SRC})

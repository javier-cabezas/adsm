set(memory_allocator_SRC
    Cache.h Cache.ipp Cache.cpp
    Slab.h Slab-impl.h Slab.cpp)
add_gmac_sources(gmac ${memory_allocator_SRC})

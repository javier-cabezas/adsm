#ifndef GMAC_MEMORY_ALLOCATOR_SLAB_IPP_
#define GMAC_MEMORY_ALLOCATOR_SLAB_IPP_

namespace __impl { namespace memory { namespace allocator {

inline Slab::Slab(manager &manager) : manager_(manager) {}

inline Slab::~Slab()
{
    aspace_map::iterator i;
    aspaces_.lockWrite();
    for(i = aspaces_.begin(); i != aspaces_.end(); i++) {
        CacheMap &map = i->second;
        CacheMap::iterator j;
        for(j = map.begin(); j != map.end(); j++) {
            delete j->second;
        }
        map.clear();
    }
    aspaces_.clear();
    aspaces_.unlock();
}

}}}
#endif

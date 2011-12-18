#ifndef GMAC_MEMORY_ALLOCATOR_SLAB_IPP_
#define GMAC_MEMORY_ALLOCATOR_SLAB_IPP_

namespace __impl { namespace memory { namespace allocator {

inline slab::slab(manager &manager) : manager_(manager) {}

inline slab::~slab()
{
    map_aspace::iterator i;
    aspaces_.lock_write();
    for(i = aspaces_.begin(); i != aspaces_.end(); i++) {
        map_cache &map = i->second;
        map_cache::iterator j;
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

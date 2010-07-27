#ifndef __MEMORY_ALLOCATOR_SLAB_IPP_
#define __MEMORY_ALLOCATOR_SLAB_IPP_

namespace gmac { namespace memory { namespace allocator {

inline
Slab::Slab(Manager *manager) :
    Allocator(manager)
{}

inline
Slab::~Slab() {}

}}}
#endif

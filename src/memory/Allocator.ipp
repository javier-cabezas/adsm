#ifndef __MEMORY_ALLOCATOR_IPP_
#define __MEMORY_ALLOCATOR_IPP_

namespace gmac { namespace memory {

inline 
Allocator::Allocator(Manager *manager) :
    logger(util::Logger("Malloc")),
    manager(manager)
{}

}}

#endif

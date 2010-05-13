#include "Allocator.h"

#include "allocator/Slab.h"

namespace gmac { namespace memory {

Allocator *Allocator::getAllocator(const char *allocatorName, Manager *manager)
{
    if(allocatorName == NULL) return new allocator::Slab(manager)
    TRACE("Using %s Allocator", allocatorName);
    if(strcasecmp(allocatorName, "None") == 0)
        return NULL;
    else return new allocator::Slab(manager);

}

}}

#include "Allocator.h"

#include "allocator/Slab.h"

namespace gmac { namespace memory {

Allocator *getAllocator(Manager *manager, const char *allocatorName)
{
    if(allocatorName == NULL) return new allocator::Slab(manager)
    TRACE("Using %s Allocator", allocatorName);
    if(strcasecmp(allocatorName, "None") == 0)
        return NULL;
    else return new allocator::Slab(manager);

}

}}

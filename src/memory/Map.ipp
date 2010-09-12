#ifndef __MEMORY_MAP_IPP_
#define __MEMORY_MAP_IPP_

namespace gmac { namespace memory {

inline Object *
Map::localFind(const void *addr)
{
    return mapFind(*this, addr);
}

}}

#endif

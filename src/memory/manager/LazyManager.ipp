#ifndef __MEMORY_LAZYMANAGER_IPP_
#define __MEMORY_LAZYMANAGER_IPP_

inline ProtRegion *
LazyManager::get(const void *addr)
{
    ProtRegion *reg = NULL;
    if(current()) current()->find<ProtRegion>(addr);
    return reg;
}

#endif

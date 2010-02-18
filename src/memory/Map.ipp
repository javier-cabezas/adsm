#ifndef __MEMORY_MAP_IPP_
#define __MEMORY_MAP_IPP_

#include "memory/Region.h"

namespace gmac { namespace memory {

inline void
Map::realloc()
{
    __pageTable.realloc();
}

inline void
Map::insert(Region *i)
{
    lockWrite();
    RegionMap::insert(value_type(i->end(), i));
    unlock();

    global.lockWrite();
    __global->insert(value_type(i->end(), i));
    global.unlock();
}

inline PageTable &
Map::pageTable()
{
    return __pageTable;
}

inline const PageTable &
Map::pageTable() const
{
    return __pageTable;
}

template<typename T>
inline T *
Map::find(const void *addr)
{
    Region *ret = NULL;
    lockRead();
    ret = localFind(addr);
    if(ret == NULL) {
        global.lockRead();
        ret = globalFind(addr);
        global.unlock();
    }
    unlock();

    return dynamic_cast<T *>(ret);
}

}}

#endif

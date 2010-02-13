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
Map::lock()
{
    local.read();
}

inline void
Map::unlock()
{
    local.unlock();
}

inline void
Map::insert(Region *i)
{
    local.write();
    RegionMap::insert(value_type(i->end(), i));
    local.unlock();

    global.write();
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
    local.read();
    ret = localFind(addr);
    if(ret == NULL) {
        global.read();
        ret = globalFind(addr);
        global.unlock();
    }
    local.unlock();
    return dynamic_cast<T *>(ret);
}

}}

#endif

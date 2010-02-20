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
Map::insert(Region *r)
{
    lockWrite();
    RegionMap::insert(value_type(r->end(), r));
    unlock();

    __global.lockWrite();
    __global.insert(value_type(r->end(), r));
    __global.unlock();
}

inline void
Map::addShared(Region * r)
{
    __shared.lockWrite();
    __shared.insert(value_type(r->end(), r));
    __shared.unlock();
}

inline void
Map::removeShared(Region * r)
{
    Map::iterator i;
    __shared.lockWrite();
    for (i = Map::__shared.begin(); i != Map::__shared.end(); i++) {
        if (r == i->second) {
            __shared.erase(i);
            delete r;
            break;
        }
    }
    __shared.unlock();
}

inline bool
Map::isShared(const void *addr)
{
    bool ret;
    __shared.lockRead();
    ret = __shared.find(addr) != __shared.end();
    __shared.unlock();

    return ret;
}

inline RegionMap &
Map::shared()
{
    return __shared;
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
        ret = globalFind(addr);
    }
    unlock();

    return dynamic_cast<T *>(ret);
}

}}

#endif

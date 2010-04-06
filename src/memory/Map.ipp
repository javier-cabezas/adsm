#ifndef __MEMORY_MAP_IPP_
#define __MEMORY_MAP_IPP_

#include <memory/Region.h>

namespace gmac { namespace memory {

inline void
Map::realloc()
{
    __pageTable.realloc();
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
    if(ret == NULL) {
        ret = sharedFind(addr);
    }
    unlock();

    return dynamic_cast<T *>(ret);
}

}}

#endif

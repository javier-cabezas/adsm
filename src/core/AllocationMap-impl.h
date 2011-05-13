#ifndef GMAC_CORE_ALLOCATIONMAP_IMPL_H_
#define GMAC_CORE_ALLOCATIONMAP_IMPL_H_

#include "config/common.h"
#include "util/Logger.h"

namespace __impl { namespace core { 

inline
AllocationMap::AllocationMap() :
    gmac::util::RWLock("AllocationMap")
{
}

inline
void AllocationMap::insert(hostptr_t key, const accptr_t &val, size_t size)
{
    lockWrite();
    ASSERTION(MapAlloc::find(key) == end());
    MapAlloc::insert(MapAlloc::value_type(key, PairAlloc((val.get()), size)));
    unlock();
}

inline
void AllocationMap::erase(hostptr_t key, size_t size)
{
    lockWrite();
    MapAlloc::erase(key);
    unlock();
}

inline
const accptr_t &AllocationMap::find(hostptr_t key, size_t &size)
{
    lockRead();
    MapAlloc::const_iterator it = MapAlloc::find(key);
    if(it != MapAlloc::end()) {
        size = it->second.second;
        const accptr_t &ref = it->second.first;
        unlock();
        return ref;
    }
    unlock();
    return nullaccptr;
}


}}

#endif

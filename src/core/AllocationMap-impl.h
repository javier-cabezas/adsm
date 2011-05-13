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

inline std::pair<const accptr_t &, bool>
AllocationMap::find(hostptr_t key, size_t &size)
{
    lockRead();
    MapAlloc::const_iterator it = MapAlloc::find(key);
    if(it != MapAlloc::end()) {
        size = it->second.second;
        std::pair<const accptr_t &, bool> ret =
            std::make_pair(it->second.first, true);
        unlock();
        return ret;
    }
    unlock();
    return std::make_pair(nullaccptr, false);
}


}}

#endif

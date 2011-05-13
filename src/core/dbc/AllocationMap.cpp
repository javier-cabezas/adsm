#ifdef USE_DBC
#include "AllocationMap.h"

namespace __dbc { namespace core {

AllocationMap::~AllocationMap()
{
    EXPECTS(empty() == true);
}

void AllocationMap::insert(hostptr_t key, const accptr_t &val, size_t size)
{
    REQUIRES(key != NULL);
    REQUIRES(val != nullaccptr);
    REQUIRES(size > 0);
    lockRead();
    REQUIRES(__impl::core::MapAlloc::find(key) == end());
    unlock();
    __impl::core::AllocationMap::insert(key, val, size);
}

void AllocationMap::erase(hostptr_t key, size_t size)
{
    REQUIRES(key != NULL);
    REQUIRES(size > 0);
    lockRead();
    REQUIRES(__impl::core::MapAlloc::find(key) != end());
    unlock();
    __impl::core::AllocationMap::erase(key, size);
}

const accptr_t &AllocationMap::find(hostptr_t key, size_t &size)
{
    REQUIRES(key != NULL);
    return __impl::core::AllocationMap::find(key, size);
}

}}

#endif

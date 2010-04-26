#ifndef __KERNEL_CONTEXT_IPP_
#define __KERNEL_CONTEXT_IPP_

#include "Kernel.h"

namespace gmac {

inline
memory::Map &
Context::mm()
{
    return *_mm;
}

inline
const memory::Map &
Context::mm() const
{
    return *_mm;
}

inline void
Context::enable()
{
    _mm->realloc();
}

inline
bool
Context::hasCurrent()
{
    return key.get() != NULL;
}

inline
void
Context::kernel(gmacKernel_t k, Kernel * kernel)
{
    ASSERT(kernel != NULL);
    TRACE("CTX: %p Registering kernel %s: %p", this, kernel->name(), k);
    KernelMap::iterator i;
    i = _kernels.find(k);
    ASSERT(i == _kernels.end());
    _kernels[k] = kernel;
}

inline
Kernel *
Context::kernel(gmacKernel_t k)
{
    KernelMap::iterator i;
    i = _kernels.find(k);
    if (i != _kernels.end()) {
        return i->second;
    }
    return NULL;
}

inline gmacError_t
Context::error() const
{
    return _error;
}


inline unsigned
Context::id() const
{
    return _id;
}

inline unsigned
Context::accId() const
{
    return _acc->_id;
}

inline memory::RegionSet
Context::releaseRegions()
{
    memory::RegionSet regions;
    regions.swap(_releasedRegions);
    _releasedAll = false;

    return regions;
}

inline void *
Context::bufferPageLocked() const
{
    return _bufferPageLocked;
}

inline size_t
Context::bufferPageLockedSize() const
{
    return _bufferPageLockedSize;
}



}

#endif

#ifndef __KERNEL_CONTEXT_IPP_
#define __KERNEL_CONTEXT_IPP_

#include "Kernel.h"

namespace gmac {

inline
Context::Status
Context::status() const
{
    return _status;
}

inline
memory::Map &
Context::mm()
{
    return _mm;
}

inline
const memory::Map &
Context::mm() const
{
    return _mm;
}

inline void
Context::enable()
{
    _mm.realloc();
}

inline
bool
Context::hasCurrent()
{
    return PRIVATE_GET(key) != NULL;
}

inline
void
Context::kernel(gmacKernel_t k, Kernel * kernel)
{
    assert(kernel != NULL);
    TRACE("CTX: %p Registering kernel %s: %p", this, kernel->name(), k);
    KernelMap::iterator i;
    i = _kernels.find(k);
    assert(i == _kernels.end());
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

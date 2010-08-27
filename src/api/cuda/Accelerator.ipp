#ifndef __API_CUDADRV_ACCELERATOR_IPP_
#define __API_CUDADRV_ACCELERATOR_IPP_

#include "Accelerator.h"

namespace gmac { namespace gpu {

inline CUdeviceptr
Context::gpuAddr(void *addr) const
{
    unsigned long a = (unsigned long)addr;
    return (CUdeviceptr)(a & 0xffffffff);
}

inline CUdeviceptr
Context::gpuAddr(const void *addr) const
{
    unsigned long a = (unsigned long)addr;
    return (CUdeviceptr)(a & 0xffffffff);
}

inline CUdevice
Accelerator::device() const
{
    return _device;
}

inline size_t
Accelerator::nContexts() const
{
    return queue.size();
}


inline int
Accelerator::major() const
{
    return _major;
}

inline int
Accelerator::minor() const
{
    return _minor;
}

}}

#endif

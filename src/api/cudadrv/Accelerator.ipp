#ifndef __API_CUDADRV_ACCELERATOR_IPP_
#define __API_CUDADRV_ACCELERATOR_IPP_

#include "Accelerator.h"

namespace gmac { namespace gpu {

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

inline bool
Accelerator::async() const
{
    return _async;
}

}}

#endif

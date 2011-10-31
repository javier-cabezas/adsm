#ifndef GMAC_API_CUDA_HPE_ACCELERATOR_IMPL_H_
#define GMAC_API_CUDA_HPE_ACCELERATOR_IMPL_H_

#include <cuda.h>

#include "util/Logger.h"
#include "trace/Tracer.h"

#include "api/cuda/IOBuffer.h"
#include "api/cuda/hpe/Mode.h"

namespace __impl { namespace cuda { namespace hpe {

inline CUdevice
Accelerator::device() const
{
    return device_;
}

inline
int Accelerator::major() const
{
    return dev_->get_major();
}

inline
int Accelerator::minor() const
{
    return dev_->get_minor();
}

#if 0
inline
void Accelerator::setContext(aspace_t &aspace) const
{
    CUresult ret;
    ret = cuCtxSetCurrent(aspace());
    CFATAL(ret == CUDA_SUCCESS, "Error pushing CUcontext: %d", ret);
}
#endif

#ifdef USE_VM
inline
cuda::hpe::Mode *
Accelerator::getLastMode()
{
    return lastMode_;
}

inline
void
Accelerator::setLastMode(cuda::hpe::Mode &mode)
{
    lastMode_ = &mode;
}
#endif

}}}

#endif

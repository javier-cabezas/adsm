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
    return major_;
}

inline
int Accelerator::minor() const
{
    return minor_;
}


inline
CUresult Accelerator::queryCUstream(CUstream stream)
{
    pushContext();
    CUresult ret = cuStreamQuery(stream);
    popContext();
    return ret;
}

inline
gmacError_t Accelerator::syncStream(CUstream stream)
{
    trace::EnterCurrentFunction();
    pushContext();
    CUresult ret = cuStreamSynchronize(stream);
    popContext();
    trace::ExitCurrentFunction();
    return error(ret);
}

inline
CUresult Accelerator::queryCUevent(CUevent event)
{
    pushContext();
    CUresult ret = cuEventQuery(event);
    popContext();
    return ret;
}

inline
gmacError_t Accelerator::syncCUevent(CUevent event)
{
    trace::EnterCurrentFunction();
    pushContext();
    CUresult ret = cuEventSynchronize(event);
    popContext();
    trace::ExitCurrentFunction();
    return error(ret);
}

inline
gmacError_t Accelerator::timeCUevents(uint64_t &t, CUevent start, CUevent end)
{
    float delta = 0.0;
    pushContext();
    CUresult ret = cuEventElapsedTime(&delta, start, end);
    popContext();
    t = uint64_t(1000.0 * delta);
    return error(ret);
}

inline
void Accelerator::pushContext() const
{
    CUresult ret;
#ifdef USE_MULTI_CONTEXT
#if 0
    std::list<CUcontext> *contexts = reinterpret_cast<std::list<CUcontext> *>(Ctx_.get());
    ASSERTION(contexts->size() > 0);
#endif
    CUcontext *ctx = reinterpret_cast<CUcontext *>(Ctx_.get());
    ASSERTION(ctx != NULL);
    mutex_.lock();
    ret = cuCtxPushCurrent(*ctx);
#else
    mutex_.lock();
    ret = cuCtxPushCurrent(ctx_);
#endif
    CFATAL(ret == CUDA_SUCCESS, "Error pushing CUcontext: %d", ret);
}

inline
void Accelerator::popContext() const
{
    CUresult ret;
    CUcontext tmp;
    ret = cuCtxPopCurrent(&tmp);
    mutex_.unlock();
    CFATAL(ret == CUDA_SUCCESS, "Error poping CUcontext: %d", ret);
}

#ifndef USE_MULTI_CONTEXT
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
#endif



#ifdef USE_MULTI_CONTEXT
inline void
Accelerator::setCUcontext(CUcontext *ctx)
{
    Ctx_.set(ctx);
}

#else
inline const CUcontext
Accelerator::getCUcontext() const
{
    return ctx_;
}

#endif

}}}

#endif

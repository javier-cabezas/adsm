#ifndef CUDA_API_CUDA_ACCELERATOR_IMPL_H_
#define CUDA_API_CUDA_ACCELERATOR_IMPL_H_

#include <cuda.h>

#include "util/Logger.h"
#include "trace/Tracer.h"

#include "IOBuffer.h"

namespace __impl { namespace cuda {


inline
CUdeviceptr Accelerator::gpuAddr(void *addr)
{
#if CUDA_VERSION <= 3010
    unsigned long a = (unsigned long)addr;
    return (CUdeviceptr)(a & 0xffffffff);
#else
    return (CUdeviceptr)addr;
#endif
}

inline
CUdeviceptr Accelerator::gpuAddr(const void *addr)
{
#if CUDA_VERSION <= 3010
    unsigned long a = (unsigned long)addr;
    return (CUdeviceptr)(a & 0xffffffff);
#else
    return (CUdeviceptr)addr;
#endif
}

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
gmacError_t Accelerator::copyToAccelerator(void *dev, const void *host, size_t size)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL,"Copy to accelerator: %p -> %p ("FMT_SIZE")", host, dev, size);
    pushContext();
#if CUDA_VERSION >= 3020
    CUresult ret = cuMemcpyHtoD(gpuAddr(dev), host, size);
#else
	CUresult ret = cuMemcpyHtoD(gpuAddr(dev), host, unsigned(size));
#endif
    popContext();
    trace::ExitCurrentFunction();
    return error(ret);
}

inline
gmacError_t Accelerator::copyToAcceleratorAsync(void *dev, IOBuffer &buffer, unsigned bufferOff, size_t count, Mode &mode, CUstream stream)
{
    trace::EnterCurrentFunction();
    uint8_t *host = buffer.addr() + bufferOff;
    TRACE(LOCAL,"Async copy to accelerator: %p -> %p ("FMT_SIZE")", host, dev, count);
    pushContext();

    buffer.toAccelerator(mode, stream);
#if CUDA_VERSION >= 3020
    CUresult ret = cuMemcpyHtoDAsync(gpuAddr(dev), host, count, stream);
#else
    CUresult ret = cuMemcpyHtoDAsync(gpuAddr(dev), host, unsigned(count), stream);
#endif
    buffer.started();
    popContext();
    trace::ExitCurrentFunction();
    return error(ret);
}

inline
gmacError_t Accelerator::copyToHost(void *host, const void *dev, size_t size)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL,"Copy to host: %p -> %p ("FMT_SIZE")", dev, host, size);
    pushContext();
#if CUDA_VERSION >= 3020
    CUresult ret = cuMemcpyDtoH(host, gpuAddr(dev), size);
#else
	CUresult ret = cuMemcpyDtoH(host, gpuAddr(dev), unsigned(size));
#endif
    popContext();
    trace::ExitCurrentFunction();
    return error(ret);
}

inline
gmacError_t Accelerator::copyToHostAsync(IOBuffer &buffer, unsigned bufferOff, const void *dev, size_t count, Mode &mode, CUstream stream)
{
    trace::EnterCurrentFunction();
    uint8_t *host = buffer.addr() + bufferOff;
    TRACE(LOCAL,"Async copy to host: %p -> %p ("FMT_SIZE")", dev, host, count);
    pushContext();
    buffer.toHost(mode, stream);
#if CUDA_VERSION >= 3020
    CUresult ret = cuMemcpyDtoHAsync(host, gpuAddr(dev), count, stream);
#else
	CUresult ret = cuMemcpyDtoHAsync(host, gpuAddr(dev), unsigned(count), stream);
#endif
    buffer.started();
    popContext();
    trace::ExitCurrentFunction();
    return error(ret);
}

inline
gmacError_t Accelerator::copyAccelerator(void *dst, const void *src, size_t size)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL,"Copy accelerator-accelerator: %p -> %p ("FMT_SIZE")", src, dst, size);
    pushContext();
#if CUDA_VERSION >= 3020
    CUresult ret = cuMemcpyDtoD(gpuAddr(dst), gpuAddr(src), size);
#else
	CUresult ret = cuMemcpyDtoD(gpuAddr(dst), gpuAddr(src), unsigned(size));
#endif
    popContext();
    trace::ExitCurrentFunction();
    return error(ret);
}

inline
gmacError_t Accelerator::execute(KernelLaunch &launch)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL,"Executing KernelLaunch");
    pushContext();
    gmacError_t ret = launch.execute();
    popContext();
    trace::ExitCurrentFunction();
    return ret;
}

inline
CUstream Accelerator::createCUstream()
{
    trace::EnterCurrentFunction();
    CUstream stream;
    pushContext();
    CUresult ret = cuStreamCreate(&stream, 0);
    popContext();
    CFATAL(ret == CUDA_SUCCESS, "Unable to create CUDA stream");
    trace::ExitCurrentFunction();
    return stream;
}

inline
void Accelerator::destroyCUstream(CUstream stream)
{
    trace::EnterCurrentFunction();
    pushContext();
    CUresult ret = cuStreamDestroy(stream);
    popContext();
    CFATAL(ret == CUDA_SUCCESS, "Unable to destroy CUDA stream");
    trace::ExitCurrentFunction();
}

inline
CUresult Accelerator::queryCUstream(CUstream stream)
{
    trace::EnterCurrentFunction();
    pushContext();
    CUresult ret = cuStreamQuery(stream);
    popContext();
    trace::ExitCurrentFunction();
    return ret;
}

inline
gmacError_t Accelerator::syncCUstream(CUstream stream)
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
    trace::EnterCurrentFunction();
    pushContext();
    CUresult ret = cuEventQuery(event);
    popContext();
    trace::ExitCurrentFunction();
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
void Accelerator::pushContext() const
{
    CUresult ret;
#ifdef USE_MULTI_CONTEXT
    ret = cuCtxPushCurrent(*Accelerator::_Ctx.get());
#else
    _mutex.lock();
    ret = cuCtxPushCurrent(_ctx);
#endif
    CFATAL(ret == CUDA_SUCCESS, "Error pushing CUcontext");
}

inline
void Accelerator::popContext() const
{
    CUresult ret;
    CUcontext tmp;
    ret = cuCtxPopCurrent(&tmp);
#ifndef USE_MULTI_CONTEXT
    _mutex.unlock();
#endif
    CFATAL(ret == CUDA_SUCCESS, "Error pushing CUcontext");
}

#ifndef USE_MULTI_CONTEXT
#ifdef USE_VM
inline
cuda::Mode *
Accelerator::getLastMode()
{
    return lastMode_;
}

inline
void
Accelerator::setLastMode(cuda::Mode &mode)
{
    lastMode_ = &mode;
}
#endif
#endif



#ifdef USE_MULTI_CONTEXT
inline void
Accelerator::setCUcontext(CUcontext * ctx)
{
    _Ctx.set(ctx);
}
#endif

}}

#endif

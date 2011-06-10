#ifndef GMAC_API_CUDA_HPE_ACCELERATOR_IMPL_H_
#define GMAC_API_CUDA_HPE_ACCELERATOR_IMPL_H_

#include <cuda.h>

#include "util/Logger.h"
#include "trace/Tracer.h"

#include "api/cuda/IOBuffer.h"

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
gmacError_t Accelerator::copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size, core::hpe::Mode &mode)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL,"Copy to accelerator: %p -> %p ("FMT_SIZE")", host, (void *) acc, size);
    trace::SetThreadState(trace::Wait);
    pushContext();
    CUresult ret = CUDA_SUCCESS;
#if USE_TRACE
    ret = cuEventRecord(start, 0);
    ASSERTION(ret == CUDA_SUCCESS);
    trace::SetThreadState(trace::Wait);
#endif
#if CUDA_VERSION >= 3020
    ret = cuMemcpyHtoD(acc, host, size);
#else
    ret = cuMemcpyHtoD(CUdeviceptr(acc), host, unsigned(size));
#endif
#if USE_TRACE
    ret = cuEventRecord(end, 0);
    ret = cuEventSynchronize(end);
    trace::SetThreadState(trace::Running);
    DataCommToAccelerator(mode, start_, end_, size);
#endif
    popContext();
    trace::SetThreadState(trace::Running);
    // TODO: add communication
    trace::ExitCurrentFunction();
    return error(ret);
}

inline
gmacError_t Accelerator::copyToAcceleratorAsync(accptr_t acc, core::IOBuffer &_buffer, size_t bufferOff, size_t count, core::hpe::Mode &mode, CUstream stream)
{
    IOBuffer &buffer = dynamic_cast<IOBuffer &>(_buffer);
    trace::EnterCurrentFunction();
    uint8_t *host = buffer.addr() + bufferOff;
    TRACE(LOCAL,"Async copy to accelerator: %p -> %p ("FMT_SIZE")", host, (void *) acc, count);
    pushContext();

    buffer.toAccelerator(dynamic_cast<cuda::Mode &>(mode), stream);
#if CUDA_VERSION >= 3020
    CUresult ret = cuMemcpyHtoDAsync(acc, host, count, stream);
#else
    CUresult ret = cuMemcpyHtoDAsync(CUdeviceptr(acc), host, unsigned(count), stream);
#endif
    buffer.started();
    popContext();
    trace::ExitCurrentFunction();
    return error(ret);
}

inline
gmacError_t Accelerator::copyToHost(hostptr_t host, const accptr_t acc, size_t size, core::hpe::Mode &mode)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL,"Copy to host: %p -> %p ("FMT_SIZE")", (void *) acc, host, size);
    trace::SetThreadState(trace::Wait);
    pushContext();
#if USE_TRACE
    ret = cuEventRecord(start, 0);
    ASSERTION(ret == CUDA_SUCCESS);
    trace::SetThreadState(trace::Wait);
#endif

#if CUDA_VERSION >= 3020
    CUresult ret = cuMemcpyDtoH(host, acc, size);
#else
        CUresult ret = cuMemcpyDtoH(host, acc, unsigned(size));
#endif
#if USE_TRACE
    ret = cuEventRecord(end, 0);
    ret = cuEventSynchronize(end);
    trace::SetThreadState(trace::Running);
    DataCommToHost(mode, start_, end_, size);
#endif

    popContext();
    trace::SetThreadState(trace::Running);
    trace::ExitCurrentFunction();
    return error(ret);
}

inline
gmacError_t Accelerator::copyToHostAsync(core::IOBuffer &_buffer, size_t bufferOff, const accptr_t acc, size_t count, core::hpe::Mode &mode, CUstream stream)
{
    IOBuffer &buffer = dynamic_cast<IOBuffer &>(_buffer);
    trace::EnterCurrentFunction();
    uint8_t *host = buffer.addr() + bufferOff;
    TRACE(LOCAL,"Async copy to host: %p -> %p ("FMT_SIZE")", (void *) acc, host, count);
    pushContext();
    buffer.toHost(dynamic_cast<cuda::Mode &>(mode), stream);
#if CUDA_VERSION >= 3020
    CUresult ret = cuMemcpyDtoHAsync(host, acc, count, stream);
#else
    CUresult ret = cuMemcpyDtoHAsync(host, acc, unsigned(count), stream);
#endif
    buffer.started();
    popContext();
    trace::ExitCurrentFunction();
    return error(ret);
}

inline
gmacError_t Accelerator::copyAccelerator(accptr_t dst, const accptr_t src, size_t size, stream_t /*stream*/)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL,"Copy accelerator-accelerator: %p -> %p ("FMT_SIZE")", (void *) src, (void *) dst, size);
    pushContext();
#if CUDA_VERSION >= 3020
    CUresult ret = cuMemcpyDtoD(dst, src, size);
#else
    CUresult ret = cuMemcpyDtoD(dst, src, unsigned(size));
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
gmacError_t Accelerator::registerMem(hostptr_t ptr, size_t size)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL,"Executing KernelLaunch");
    CUresult ret = cuMemHostRegister(ptr, size, CU_MEMHOSTREGISTER_PORTABLE);
    CFATAL(ret == CUDA_SUCCESS);
    trace::ExitCurrentFunction();
    return error(ret);
}

inline
gmacError_t Accelerator::unregisterMem(hostptr_t ptr)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL,"Executing KernelLaunch");
    pushContext();
    CUresult ret = cuMemHostUnregister(ptr);
    CFATAL(ret == CUDA_SUCCESS);
    popContext();
    trace::ExitCurrentFunction();
    return error(ret);
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
    //trace::EnterCurrentFunction();
    pushContext();
    CUresult ret = cuStreamQuery(stream);
    popContext();
    //trace::ExitCurrentFunction();
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
    //trace::EnterCurrentFunction();
    pushContext();
    CUresult ret = cuEventQuery(event);
    popContext();
    //trace::ExitCurrentFunction();
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

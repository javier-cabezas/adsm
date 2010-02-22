#ifndef __API_CUDADRV_CONTEXT_IPP_
#define __API_CUDADRV_CONTEXT_IPP_

#include "Kernel.h"

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

inline void
Context::zero(void **addr) const
{
    memory::addr_t *ptr = (memory::addr_t *)addr;
    *ptr = 0;
}

inline Context *
Context::current()
{
    return static_cast<Context *>(gmac::Context::current());
}

#ifdef USE_MULTI_CONTEXT
inline void
Context::lock()
{
    mutex.lock();
    CUresult ret = cuCtxPushCurrent(_ctx);
    ASSERT(ret == CUDA_SUCCESS);
}

inline void
Context::unlock()
{
    CUcontext tmp;
    CUresult ret = cuCtxPopCurrent(&tmp);
    ASSERT(ret == CUDA_SUCCESS);
    mutex.unlock();
}
#else
inline void
Context::lock()
{
    _gpu.lock();
}

inline void
Context::unlock()
{
    _gpu.unlock();
}

#endif

inline gmacError_t
Context::copyToDeviceAsync(void *dev, const void *host, size_t size)
{
    lock();
    enterFunction(accHostDeviceCopy);
    CUresult ret = cuMemcpyHtoDAsync(gpuAddr(dev), host, size, streamToDevice);
    exitFunction();
    unlock();
    return error(ret);
}

inline gmacError_t
Context::copyToHostAsync(void *host, const void *dev, size_t size) 
{
    lock();
    enterFunction(accDeviceHostCopy);
    CUresult ret = cuMemcpyDtoHAsync(host, gpuAddr(dev), size, streamToHost);
    exitFunction();
    unlock();
    return error(ret);
}


inline gmacError_t
Context::sync()
{
    CUresult ret = CUDA_SUCCESS;
    lock();
    while ((ret = cuStreamQuery(streamLaunch)) == CUDA_ERROR_NOT_READY) {
        unlock();
        usleep(Context::USleepLaunch);
        lock();
    }
    if (ret == CUDA_SUCCESS) {
        TRACE("Sync: success");
    } else {
        TRACE("Sync: error: %d", ret);
    }

    unlock();

    return error(ret);
}

inline gmacError_t
Context::syncToHost()
{
    CUresult ret;
    lock();
    if (_gpu.async()) {
        ret = cuStreamSynchronize(streamToHost);
    } else {
        ret = cuCtxSynchronize();
    }
    unlock();
    return error(ret);
}

inline gmacError_t
Context::syncToDevice()
{
    CUresult ret;
    lock();
    if (_gpu.async()) {
        ret = cuStreamSynchronize(streamToDevice);
    } else {
        ret = cuCtxSynchronize();
    }
    unlock();
    return error(ret);
}

inline gmacError_t
Context::syncDevice()
{
    CUresult ret;
    lock();
    if (_gpu.async()) {
        ret = cuStreamSynchronize(streamDevice);
    } else {
        ret = cuCtxSynchronize();
    }
    unlock();
    return error(ret);
}

inline void
Context::call(dim3 Dg, dim3 Db, size_t shared, int tokens)
{
    _call = KernelConfig(Dg, Db, shared, tokens);
}

inline void
Context::argument(const void *arg, size_t size, off_t offset)
{
    _call.pushArgument(arg, size, offset);
}

inline bool
Context::async() const
{
    return _gpu.async();
}

}}

#endif

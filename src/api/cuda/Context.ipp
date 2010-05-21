#ifndef __API_CUDADRV_CONTEXT_IPP_
#define __API_CUDADRV_CONTEXT_IPP_

#include "Kernel.h"

#include <config/paraver.h>

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

inline CUstream
Context::stream()
{
    return current()->streamLaunch;
}


#ifdef USE_MULTI_CONTEXT
inline void
Context::pushLock()
{
    mutex.lock();
    CUresult ret = cuCtxPushCurrent(_ctx);
    cfatal(ret == CUDA_SUCCESS, "Error pushing context %d", ret);
}

inline void
Context::popUnlock()
{
    CUcontext tmp;
    CUresult ret = cuCtxPopCurrent(&tmp);
    cfatal(ret == CUDA_SUCCESS, "Error popping context %d", ret);
    mutex.unlock();
}
#else
inline void
Context::pushLock()
{
    _gpu->pushLock();
}

inline void
Context::popUnlock()
{
    _gpu->popUnlock();
}

#endif

inline gmacError_t
Context::copyToDeviceAsync(void *dev, const void *host, size_t size)
{
    pushLock();
    enterFunction(FuncAccHostDevice);
    CUresult ret = cuMemcpyHtoDAsync(gpuAddr(dev), host, size, streamToDevice);
    exitFunction();
    popUnlock();
    return error(ret);
}

inline gmacError_t
Context::copyToHostAsync(void *host, const void *dev, size_t size) 
{
    pushLock();
    enterFunction(FuncAccDeviceHost);
    CUresult ret = cuMemcpyDtoHAsync(host, gpuAddr(dev), size, streamToHost);
    exitFunction();
    popUnlock();
    return error(ret);
}


inline gmacError_t
Context::sync()
{
    CUresult ret = CUDA_SUCCESS;
    if (_pendingKernel) {
        pushLock();
        while ((ret = cuStreamQuery(streamLaunch)) == CUDA_ERROR_NOT_READY) {
            popUnlock();
            pushLock();
        }
        popEventState(paraver::Accelerator, 0x10000000 + _id);

        if (ret == CUDA_SUCCESS) {
            trace("Sync: success");
        } else {
            trace("Sync: error: %d", ret);
        }

        _pendingKernel = false;
        popUnlock();
    }

    return error(ret);
}

inline gmacError_t
Context::syncToHost()
{
    CUresult ret;
    pushLock();
    ret = cuStreamSynchronize(streamToHost);
    popUnlock();
    return error(ret);
}

inline gmacError_t
Context::syncToDevice()
{
    CUresult ret;
    pushLock();
    ret = cuStreamSynchronize(streamToDevice);
    if (_pendingToDevice) {
        popEventState(paraver::Accelerator, 0x10000000 + _id);
        _pendingToDevice = false;
    }

    popUnlock();
    return error(ret);
}

inline gmacError_t
Context::syncDevice()
{
    CUresult ret;
    pushLock();
    ret = cuStreamSynchronize(streamDevice);
    popUnlock();
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


}}

#endif

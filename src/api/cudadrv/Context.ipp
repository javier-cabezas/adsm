#ifndef __API_CUDADRV_CONTEXT_IPP_
#define __API_CUDADRV_CONTEXT_IPP_

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

inline void
Context::lock()
{
    mutex->lock();
    assert(cuCtxPushCurrent(ctx) == CUDA_SUCCESS);
}

inline void
Context::unlock()
{
    CUcontext tmp;
    assert(cuCtxPopCurrent(&tmp) == CUDA_SUCCESS);
    mutex->unlock();
}

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

#if 0
inline gmacError_t
Context::copyDeviceAsync(void *src, const void *dest, size_t size)
{
    lock();
    enterFunction(accDeviceCopy);
    CUresult ret = cuMemcpyDtoDAsync(gpuAddr(dest), gpuAddr(src), size, streamDevice);
    exitFunction();
    unlock();
    return error(ret);
}
#endif

inline gmacError_t
Context::sync()
{
    CUresult ret;
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
    if (gpu.async()) {
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
    if (gpu.async()) {
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
    if (gpu.async()) {
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
    Call c(Dg, Db, shared, tokens);
    _calls.push_back(c);
}

inline void
Context::argument(const void *arg, size_t size, off_t offset)
{
	memcpy(&_stack[offset], arg, size);
	_sp = (_sp > (offset + size)) ? _sp : offset + size;
}

inline bool
Context::async() const
{
    return gpu.async();
}

#endif

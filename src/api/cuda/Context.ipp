#ifndef __API_CUDA_GPUCONTEXT_IPP_
#define __API_CUDA_GPUCONTEXT_IPP_

inline void
Context::check()
{
    assert(current() == this);
}

inline void
Context::lock()
{}

inline void
Context::unlock()
{}

// Standard Accelerator Interface
inline gmacError_t
Context::malloc(void **addr, size_t size)
{
    check();
    cudaError_t ret = cudaMalloc(addr, size);
    return error(ret);
}


inline gmacError_t
Context::free(void *addr)
{
    check();
    cudaError_t ret = cudaFree(addr);
    return error(ret);
}

inline gmacError_t
Context::hostAlloc(void **host, void **dev, size_t size)
{
    check();
    if (dev != NULL) {
        *dev = NULL;
        cudaError_t ret = cudaHostAlloc(host, size, cudaHostAllocMapped | cudaHostAllocPortable);
        if(ret == cudaSuccess)
            assert(cudaHostGetDevicePointer(dev, *host, 0) == cudaSuccess);
    } else {
        cudaError_t ret = cudaHostAlloc(host, size, cudaHostAllocPortable);
    }
    return error(ret);
}

inline gmacError_t
Context::hostMemAlign(void **host, void **dev, size_t size)
{
    FATAL("Not implemented");
}

inline gmacError_t
Context::hostMap(void *host, void **dev, size_t size)
{
    FATAL("Not implemented");
}

inline gmacError_t
Context::hostFree(void *addr)
{
    check();
    cudaError_t ret = cudaFreeHost(addr);
    return error(ret);
}

inline gmacError_t
Context::copyToDevice(void *dev, const void *host, size_t size)
{
    check();
    enterFunction(accHostDeviceCopy);
    cudaError_t ret = cudaMemcpy(dev, host, size, cudaMemcpyHostToDevice);
    exitFunction();
    return error(ret);
}

inline gmacError_t
Context::copyToHost(void *host, const void *dev, size_t size)
{
    check();
    enterFunction(accDeviceHostCopy);
    cudaError_t ret = cudaMemcpy(host, dev, size, cudaMemcpyDeviceToHost);
    exitFunction();
    return error(ret);
}

inline gmacError_t
Context::copyDevice(void *dst, const void *src, size_t size)
{
    check();
    enterFunction(accDeviceDeviceCopy);
    cudaError_t ret = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    exitFunction();
    return error(ret);
}

inline gmacError_t
Context::copyToDeviceAsync(void *dev, const void *host, size_t size)
{
    check();
    enterFunction(accHostDeviceCopy);
    cudaError_t ret = cudaMemcpyAsync(dev, host, size,
            cudaMemcpyHostToDevice, 0);
    exitFunction();
    return error(ret);
}

inline gmacError_t
Context::copyToHostAsync(void *host, const void *dev, size_t size)
{
    check();
    enterFunction(accDeviceHostCopy);
    cudaError_t ret = cudaMemcpyAsync(host, dev, size,
            cudaMemcpyDeviceToHost, 0);
    exitFunction();
    return error(ret);
}

inline gmacError_t
Context::memset(void *dev, int c, size_t size)
{
    check();
    cudaError_t ret = cudaMemset(dev, c, size);
    return error(ret);
}

inline gmacError_t
Context::launch(const char *kernel)
{
    check();
    cudaError_t ret = __cudaLaunch(kernel);
    return error(ret);
}
	
inline gmacError_t
Context::sync()
{
    check();
    cudaError_t ret = cudaThreadSynchronize();
    return error(ret);
}

inline void
Context::flush()
{
#ifdef USE_VM
    devicePageTable.ptr = mm().pageTable().flush();
    devicePageTable.shift = mm().pageTable().getTableShift();
    devicePageTable.size = mm().pageTable().getTableSize();
    devicePageTable.page = mm().pageTable().getPageSize();

    assert(cudaMemcpyToSymbol(pageTableSymbol, &devicePageTable,
                sizeof(devicePageTable), 0, cudaMemcpyHostToDevice) == cudaSuccess);
#endif
}

inline void
Context::invalidate()
{
#ifdef USE_VM
    mm().pageTable().invalidate();
#endif
}

#endif

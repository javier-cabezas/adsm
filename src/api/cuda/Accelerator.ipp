#ifndef __API_CUDA_ACCELERATOR_IPP_
#define __API_CUDA_ACCELERATOR_IPP_

namespace gmac { namespace cuda {

inline
void Accelerator::switchIn()
{
#ifndef USE_MULTI_CONTEXT
    _mutex.lock();
    CUresult ret = cuCtxPushCurrent(_ctx);
    cfatal(ret == CUDA_SUCCESS, "Unable to switch to CUDA mode [%d]", ret);
#endif
}

inline
void Accelerator::switchOut()
{
#ifndef USE_MULTI_CONTEXT
    CUcontext tmp;
    CUresult ret = cuCtxPopCurrent(&tmp);
    _mutex.unlock();
    cfatal(ret == CUDA_SUCCESS, "Unable to switch back from CUDA mode [%d]", ret);
#endif
}

inline
CUdeviceptr Accelerator::gpuAddr(void *addr)
{
    unsigned long a = (unsigned long)addr;
    return (CUdeviceptr)(a & 0xffffffff);
}

inline
CUdeviceptr Accelerator::gpuAddr(const void *addr)
{
    unsigned long a = (unsigned long)addr;
    return (CUdeviceptr)(a & 0xffffffff);
}

inline CUdevice
Accelerator::device() const
{
    return _device;
}

inline
int Accelerator::major() const
{
    return _major;
}

inline 
int Accelerator::minor() const
{
    return _minor;
}

inline
gmacError_t Accelerator::copyToDevice(void *dev, const void *host, size_t size)
{
    CUresult ret = cuMemcpyHtoD(gpuAddr(dev), host, size);
    return error(ret);
}

inline
gmacError_t Accelerator::copyToDeviceAsync(void *dev, const void *host, size_t size, Stream stream)
{
    CUresult ret = cuMemcpyHtoDAsync(gpuAddr(dev), host, size, stream);
    return error(ret);
}

inline
gmacError_t Accelerator::copyToHost(void *host, const void *dev, size_t size)
{
    CUresult ret =cuMemcpyDtoH(host, gpuAddr(dev), size);
    return error(ret);
}

inline
gmacError_t Accelerator::copyToHostAsync(void *host, const void *dev, size_t size, Stream stream)
{
    CUresult ret = cuMemcpyDtoHAsync(host, gpuAddr(dev), size, stream);
    return error(ret);
}

inline
gmacError_t Accelerator::copyDevice(void *dst, const void *src, size_t size)
{
    CUresult ret = cuMemcpyDtoD(gpuAddr(dst), gpuAddr(src), size);
    return error(ret);
}

inline
gmacError_t Accelerator::syncStream(Stream stream)
{
    CUresult ret = cuStreamSynchronize(stream);
    return error(ret);
}

}}

#endif

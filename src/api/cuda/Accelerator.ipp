#ifndef __API_CUDADRV_ACCELERATOR_IPP_
#define __API_CUDADRV_ACCELERATOR_IPP_

#include "Accelerator.h"

namespace gmac { namespace gpu {

inline
void Accelerator::switchIn()
{
#ifndef USE_MULTI_CONTEXT
    __mutex.lock();
    CUresult ret = cuCtxPushCurrent(__ctx);
    cfatal(ret != CUDA_SUCCESS, "Unable to switch to CUDA mode");
#endif
}

inline
void Accelerator::switchOut()
{
#ifndef USE_MULTI_CONTEXT
    CUcontext tmp;
    CUresult ret = cuCtxPopCurrent(&tmp);
    __mutex.unlock();
    cfatal(ret != CUDA_SUCCESS, "Unable to switch back from CUDA mode");
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
    switchIn();
    CUresult ret = cuMemcpyHtoD(gpuAddr(dev), host, size);
    switchOut();
    return error(ret);
}

inline
gmacError_t Accelerator::copyToDeviceAsync(void *dev, const void *host, size_t size, Stream stream)
{
    switchIn();
    CUresult ret = cuMemcpyHtoDAsync(gpuAddr(dev), host, size, stream);
    switchOut();
    return error(ret);
}

inline
gmacError_t Accelerator::copyToHost(void *host, const void *dev, size_t size)
{
    switchIn();
    CUresult ret =cuMemcpyDtoH(host, gpuAddr(dev), size);
    switchOut();
    return error(ret);
}

inline
gmacError_t Accelerator::copyToHostAsync(void *host, const void *dev, size_t size, Stream stream)
{
    switchIn();
    CUresult ret = cuMemcpyDtoHAsync(host, gpuAddr(dev), size, stream);
    switchOut();
    return error(ret);
}

inline
gmacError_t Accelerator::copyDevice(void *dst, const void *src, size_t size)
{
    switchIn();
    CUresult ret = cuMemcpyDtoD(gpuAddr(dst), gpuAddr(src), size);
    switchOut();
    return error(ret);
}

gmacError_t Accelerator::syncStream(Stream stream)
{
    switchIn();
    CUresult ret = cuStreamSynchronize(stream);
    switchOut();
    return error(ret);
}

}}

#endif

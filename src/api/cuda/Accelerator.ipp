#ifndef CUDA_API_CUDA_ACCELERATOR_IPP_
#define CUDA_API_CUDA_ACCELERATOR_IPP_

#include <cuda.h>

namespace gmac { namespace cuda {

#if 0
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
#endif

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
    return _major;
}

inline 
int Accelerator::minor() const
{
    return _minor;
}

inline
gmacError_t Accelerator::copyToAccelerator(void *dev, const void *host, size_t size)
{
    gmac::trace::Function::start("Accelerator","copyToAccelerator");
    trace("Copy to accelerator: %p -> %p (%zd)", host, dev, size);
    pushContext();
    CUresult ret = cuMemcpyHtoD(gpuAddr(dev), host, size);
    popContext();
    gmac::trace::Function::end("Accelerator");
    return error(ret);
}

inline
gmacError_t Accelerator::copyToAcceleratorAsync(void *dev, const void *host, size_t size, CUstream stream)
{
    gmac::trace::Function::start("Accelerator","copyToAcceleratorAsync");
    trace("Async copy to accelerator: %p -> %p (%zd)", host, dev, size);
    pushContext();
    CUresult ret = cuMemcpyHtoDAsync(gpuAddr(dev), host, size, stream);
    popContext();
    gmac::trace::Function::end("Accelerator");
    return error(ret);
}

inline
gmacError_t Accelerator::copyToHost(void *host, const void *dev, size_t size)
{
    gmac::trace::Function::start("Accelerator","copyToHost");
    trace("Copy to host: %p -> %p (%zd)", dev, host, size);
    pushContext();
    CUresult ret =cuMemcpyDtoH(host, gpuAddr(dev), size);
    popContext();
    gmac::trace::Function::end("Accelerator");
    return error(ret);
}

inline
gmacError_t Accelerator::copyToHostAsync(void *host, const void *dev, size_t size, CUstream stream)
{
    gmac::trace::Function::start("Accelerator","copyToHostAsync");
    trace("Async copy to host: %p -> %p (%zd)", dev, host, size);
    pushContext();
    CUresult ret = cuMemcpyDtoHAsync(host, gpuAddr(dev), size, stream);
    popContext();
    gmac::trace::Function::end("Accelerator");
    return error(ret);
}

inline
gmacError_t Accelerator::copyAccelerator(void *dst, const void *src, size_t size)
{
    gmac::trace::Function::start("Accelerator","copyAccelerator");
    trace("Copy accelerator-accelerator: %p -> %p (%zd)", src, dst, size);
    pushContext();
    CUresult ret = cuMemcpyDtoD(gpuAddr(dst), gpuAddr(src), size);
    popContext();
    gmac::trace::Function::end("Accelerator");
    return error(ret);
}

inline
gmacError_t Accelerator::execute(KernelLaunch &launch)
{
    gmac::trace::Function::start("Accelerator","execute");
    trace("Executing KernelLaunch");
    pushContext();
    gmacError_t ret = launch.execute();
    popContext();
    gmac::trace::Function::end("Accelerator");
    return ret;
}

inline
CUstream Accelerator::createCUstream()
{
    gmac::trace::Function::start("Accelerator","createCUstream");
    CUstream stream;
    pushContext();
    CUresult ret = cuStreamCreate(&stream, 0);
    popContext();
    CFatal(ret == CUDA_SUCCESS, "Unable to create CUDA stream");
    gmac::trace::Function::end("Accelerator");
    return stream;
}

inline
void Accelerator::destroyCUstream(CUstream stream)
{
    gmac::trace::Function::start("Accelerator","createCUstream");
    pushContext();
    CUresult ret = cuStreamDestroy(stream);
    popContext();
    CFatal(ret == CUDA_SUCCESS, "Unable to destroy CUDA stream");
    gmac::trace::Function::end("Accelerator");
}

inline
CUresult Accelerator::queryCUstream(CUstream stream)
{
    gmac::trace::Function::start("Accelerator","queryCUstream");
    pushContext();
    CUresult ret = cuStreamQuery(stream);
    popContext();
    gmac::trace::Function::end("Accelerator");
    return ret;
}

inline
gmacError_t Accelerator::syncCUstream(CUstream stream)
{
    gmac::trace::Function::start("Accelerator","syncCUstream");
    pushContext();
    CUresult ret = cuStreamSynchronize(stream);
    popContext();
    gmac::trace::Function::end("Accelerator");
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
    CFatal(ret == CUDA_SUCCESS, "Error pushing CUcontext");
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
    CFatal(ret == CUDA_SUCCESS, "Error pushing CUcontext");
}

#ifdef USE_MULTI_CONTEXT
inline void
Accelerator::setCUcontext(CUcontext * ctx)
{
    _Ctx.set(ctx);
}
#endif

}}

#endif

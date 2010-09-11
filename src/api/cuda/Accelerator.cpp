#include "Accelerator.h"
#include "Mode.h"

#include <kernel/Process.h>

namespace gmac { namespace cuda {

Accelerator::Accelerator(int n, CUdevice device) :
	gmac::Accelerator(n), _device(device)
{
    unsigned int size = 0;
    CUresult ret = cuDeviceTotalMem(&size, _device);
    cfatal(ret == CUDA_SUCCESS, "Unable to initialize CUDA %d", ret);
    ret = cuDeviceComputeCapability(&_major, &_minor, _device);
    cfatal(ret == CUDA_SUCCESS, "Unable to initialize CUDA %d", ret);
    _memory = size;

#ifndef USE_MULTI_CONTEXT
    CUcontext tmp;
    unsigned int flags = 0;
#if CUDART_VERSION >= 2020
    if(_major >= 2 || (_major == 1 && _minor >= 1)) flags |= CU_CTX_MAP_HOST;
#else
    trace("Host mapped memory not supported by the HW");
#endif
    ret = cuCtxCreate(&_ctx, flags, _device);
    cfatal(ret == CUDA_SUCCESS, "Unable to create CUDA context %d", ret);
    ret = cuCtxPopCurrent(&tmp);
    cfatal(ret == CUDA_SUCCESS, "Error setting up a new context %d", ret);
#endif
}

Accelerator::~Accelerator()
{
#ifndef USE_MULTI_CONTEXT
    switchIn();
    ModuleVector::iterator i;
    for(i = _modules.begin(); i != _modules.end(); i++)
        delete *i;
    _modules.clear();
    switchOut();
    assertion(cuCtxDestroy(_ctx) == CUDA_SUCCESS);
#endif
}

gmac::Mode *Accelerator::createMode()
{
	cuda::Mode *mode = new cuda::Mode(this);
	_queue.insert(mode);
	trace("Attaching Execution Mode %p to Accelerator", mode);
    _load++;
	return mode;
}

void Accelerator::destroyMode(gmac::Mode *mode)
{
	trace("Destroying Execution Mode %p", mode);
	if(mode == NULL) return;
	std::set<Mode *>::iterator c = _queue.find((Mode *)mode);
	assertion(c != _queue.end());
	_queue.erase(c);
    _load--;
}


#ifdef USE_MULTI_CONTEXT
CUcontext
Accelerator::createCUDAContext()
{
    CUcontext ctx, tmp;
    unsigned int flags = 0;
#if CUDART_VERSION >= 2020
    if(_major >= 2 || (_major == 1 && _minor >= 1)) flags |= CU_CTX_MAP_HOST;
#else
    trace("Host mapped memory not supported by the HW");
#endif
    CUresult ret = cuCtxCreate(&ctx, flags, _device);
    if(ret != CUDA_SUCCESS)
        FATAL("Unable to create CUDA context %d", ret);
    ret = cuCtxPopCurrent(&tmp);
    assertion(ret == CUDA_SUCCESS);
    return ctx;
}

void
Accelerator::destroyCUDAContext(CUcontext ctx)
{
    cfatal(cuCtxDestroy(ctx) == CUDA_SUCCESS, "Error destroying CUDA context");
}
#endif

#ifndef USE_MULTI_CONTEXT
ModuleVector &Accelerator::createModules()
{
    if(_modules.empty()) _modules = ModuleDescriptor::createModules();
    return _modules;
}
#endif

gmacError_t Accelerator::malloc(void **addr, size_t size, unsigned align) 
{
    assertion(addr != NULL);
    *addr = NULL;
    if(align > 1) {
        size += align;
    }
    CUdeviceptr ptr = 0;
    CUresult ret = cuMemAlloc(&ptr, size);
    if(ret != CUDA_SUCCESS) return error(ret);
    CUdeviceptr gpuPtr = ptr;
    if(gpuPtr % align) {
        gpuPtr += align - (gpuPtr % align);
    }
    *addr = (void *)gpuPtr;
    _alignMap.insert(AlignmentMap::value_type(gpuPtr, ptr));
    trace("Allocating device memory: %p - %zd bytes (alignment %u)", *addr, size, align);
    return error(ret);
}

gmacError_t Accelerator::free(void *addr)
{
    assertion(addr != NULL);
    AlignmentMap::const_iterator i;
    CUdeviceptr gpuPtr = gpuAddr(addr);
    i = _alignMap.find(gpuPtr);
    if (i == _alignMap.end()) return gmacErrorInvalidValue;
    CUresult ret = cuMemFree(i->second);
    return error(ret);
}

gmacError_t Accelerator::memset(void *addr, int c, size_t size)
{
    CUresult ret = CUDA_SUCCESS;
    if(size % 32 == 0) {
        int seed = c | (c << 8) | (c << 16) | (c << 24);
        ret = cuMemsetD32(gpuAddr(addr), seed, size);
    }
    else if(size % 16) {
        short seed = c | (c << 8);
        ret = cuMemsetD16(gpuAddr(addr), seed, size);
    }
    else ret = cuMemsetD8(gpuAddr(addr), c, size);
    return error(ret);
}

gmacError_t Accelerator::sync()
{
    CUresult ret = cuCtxSynchronize();
    return error(ret);
}

gmacError_t Accelerator::hostAlloc(void **addr, size_t size)
{
#if CUDART_VERSION >= 2020
    CUresult ret = cuMemHostAlloc(addr, size, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP);
#else
    CUresult ret = cuMemAllocHost(addr, size);
#endif
    return error(ret);
}

gmacError_t Accelerator::hostFree(void *addr)
{
    CUresult r = cuMemFreeHost(addr);
    return error(r);
}

void *Accelerator::hostMap(void *addr)
{
    CUdeviceptr device;
    CUresult ret = cuMemHostGetDevicePointer(&device, addr, 0);
    if(ret != CUDA_SUCCESS) device = NULL;
    return (void *)device;
}


}}

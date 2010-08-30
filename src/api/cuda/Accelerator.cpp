#include "Accelerator.h"
#include "Mode.h"

#include <kernel/Process.h>

namespace gmac {
namespace gpu {

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
    ret = cuCtxCreate(&__ctx, flags, _device);
    cfatal(ret == CUDA_SUCCESS, "Unable to create CUDA context %d", ret);
    ret = cuCtxPopCurrent(&tmp);
    cfatal(ret == CUDA_SUCCESS, "Error setting up a new context %d", ret);
#endif
}

Accelerator::~Accelerator()
{}

gmac::Mode *Accelerator::createMode()
{
	trace("Attaching Execution Mode to Accelerator");
	gpu::Mode *mode = new gpu::Mode(this);
	queue.insert(mode);
	return mode;
}

void Accelerator::destroyMode(gmac::Mode *mode)
{
	trace("Destroying Execution Mode");
	if(mode == NULL) return;
	Mode *m = dynamic_cast<gpu::Mode*>(mode);
	std::set<Mode *>::iterator c = queue.find(m);
	assertion(c != queue.end());
	//delete ctx;
	queue.erase(c);
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

gmacError_t Accelerator::malloc(void **addr, size_t size, unsigned align) 
{
    assertion(addr != NULL);
    *addr = NULL;
    if(align > 1) {
        size += align;
    }
    CUdeviceptr ptr = 0;
    CUresult ret = cuMemAlloc(&ptr, size);
    CUdeviceptr gpuPtr = ptr;
    if(gpuPtr % align) {
        gpuPtr += align - (gpuPtr % align);
    }
    *addr = (void *)gpuPtr;
    __alignMap.insert(AlignmentMap::value_type(gpuPtr, ptr));
    return error(ret);
}

gmacError_t Accelerator::free(void *addr)
{
    assertion(addr != NULL);
    AlignmentMap::const_iterator i;
    CUdeviceptr gpuPtr = gpuAddr(addr);
    i = __alignMap.find(gpuPtr);
    if (i == __alignMap.end()) return gmacErrorInvalidValue;
    CUresult ret = cuMemFree(i->second);
    return error(ret);
}

gmacError_t Accelerator::sync()
{
    CUresult ret = cuCtxSynchronize();
    return error(ret);
}

}}

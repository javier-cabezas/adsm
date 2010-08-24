#include "Accelerator.h"
#include "Context.h"

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
    ret = cuCtxCreate(&_ctx, flags, _device);
    cfatal(ret == CUDA_SUCCESS, "Unable to create CUDA context %d", ret);
    ret = cuCtxPopCurrent(&tmp);
    cfatal(ret == CUDA_SUCCESS, "Error setting up a new context %d", ret);
#endif
}

Accelerator::~Accelerator()
{}

gmac::Context *Accelerator::create()
{
	trace("Attaching context to Accelerator");
	gpu::Context *ctx = new gpu::Context(this);
	queue.insert(ctx);
	return ctx;
}

void Accelerator::destroy(gmac::Context *context)
{
	trace("Destroying Context");
	if(context == NULL) return;
	gpu::Context *ctx = dynamic_cast<gpu::Context *>(context);
	std::set<gpu::Context *>::iterator c = queue.find(ctx);
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

}}

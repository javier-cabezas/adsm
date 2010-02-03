#include "Accelerator.h"
#include "Context.h"

#include "kernel/Process.h"

#include <debug.h>


namespace gmac {
namespace gpu {

Accelerator::Accelerator(int n, CUdevice device) :
	gmac::Accelerator(n), _device(device)
{
    unsigned int size = 0;
	assert(cuDeviceTotalMem(&size, _device) == CUDA_SUCCESS);
    assert(cuDeviceComputeCapability(&major, &minor, _device) == CUDA_SUCCESS);
    _memory = size;
    int async = 0;
	assert(cuDeviceGetAttribute(&async, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, _device) == CUDA_SUCCESS);
    _async = bool(async);
}

Accelerator::~Accelerator()
{}

gmac::Context *Accelerator::create()
{
	TRACE("Attaching context to Accelerator");
	gpu::Context *ctx = new gpu::Context(*this);
	queue.insert(ctx);
	return ctx;
}

#if 0
gmac::Context *Accelerator::clone(const gmac::Context &root)
{
	TRACE("Accelerator %p: new cloned context");
	const gpu::Context &_root = dynamic_cast<const Context &>(root);
	gpu::Context *ctx = new gpu::Context(_root, *this);
	queue.insert(ctx);
	return ctx;
}
#endif

void Accelerator::destroy(gmac::Context *context)
{
	TRACE("Destroying Context");
	if(context == NULL) return;
	gpu::Context *ctx = dynamic_cast<gpu::Context *>(context);
	std::set<gpu::Context *>::iterator c = queue.find(ctx);
	assert(c != queue.end());
	delete ctx;
	queue.erase(c);
}

CUcontext
Accelerator::createCUDAContext()
{
    CUcontext ctx, tmp;
    unsigned int flags = 0;
    if(major > 0 && minor > 0) flags |= CU_CTX_MAP_HOST;
    CUresult ret = cuCtxCreate(&ctx, flags, _device);
    if(ret != CUDA_SUCCESS)
        FATAL("Unable to create CUDA context %d", ret);
    assert(cuCtxPopCurrent(&tmp) == CUDA_SUCCESS);
    return ctx;
}

}}

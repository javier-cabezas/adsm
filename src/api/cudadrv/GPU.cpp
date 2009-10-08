#include "GPU.h"
#include "Context.h"

#include <kernel/Process.h>

#include <debug.h>


namespace gmac {

GPU::GPU(int n, CUdevice device) :
	id(n), _device(device)
{
	unsigned int size = 0;
	assert(cuDeviceTotalMem(&size, _device) == CUDA_SUCCESS);
	_memory = size;
}

Context *GPU::create()
{
	TRACE("Attaching context to GPU");
	gpu::Context *ctx = new gpu::Context(*this);
	queue.insert(ctx);
	return ctx;
}

Context *GPU::clone(const gmac::Context &root)
{
	TRACE("GPU %p: new cloned context");
	const gpu::Context &_root = dynamic_cast<const gpu::Context &>(root);
	gpu::Context *ctx = new gpu::Context(_root, *this);
	queue.insert(ctx);
	return ctx;
}

void GPU::destroy(Context *context)
{
	TRACE("Destroying Context");
	if(context == NULL) return;
	gpu::Context *ctx = dynamic_cast<gpu::Context *>(context);
	std::set<gpu::Context *>::iterator c = queue.find(ctx);
	assert(c != queue.end());
	delete ctx;
	queue.erase(c);
}

};

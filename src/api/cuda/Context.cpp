#include "Context.h"

namespace gmac { namespace gpu {

#ifdef USE_VM
const char *Context::pageTableSymbol = "__pageTable";
#endif

Context::Context(GPU &gpu) :
    gmac::Context(gpu),
    gpu(gpu)
{
    enable();
    cudaSetDevice(gpu.device());
    if (gpu.async()) {
        assert(cudaHostAlloc(&_bufferPageLocked, paramBufferPageLockedSize, cudaHostAllocPortable) == cudaSuccess);
        _bufferPageLockedSize = paramBufferPageLockedSize;
    } else {
        _bufferPageLocked     = NULL;
        _bufferPageLockedSize = 0;
    }
    TRACE("New GPU context [%p]", this);
}

Context::~Context()
{
    TRACE("Remove GPU context [%p]", this);
}

gmacError_t
Context::hostLockAlloc(void **addr, size_t size)
{
	cudaError_t ret = cudaSuccess;
	lock();
	ret = cudaHostAlloc(addr, size, cudaHostAllocPortable);
	unlock();
	return error(ret);
}


}}

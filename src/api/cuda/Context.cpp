#include "Context.h"

namespace gmac { namespace gpu {

#ifdef USE_VM
const char *Context::pageTableSymbol = "__pageTable";
#endif

gmacError_t Context::hostLockAlloc(void **addr, size_t size)
{
	cudaError_t ret = cudaSuccess;
	lock();
	ret = cudaHostAlloc(addr, size, cudaHostAllocPortable);
	unlock();
	return error(ret);
}


}};

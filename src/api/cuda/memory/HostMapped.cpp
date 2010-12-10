#include "api/cuda/Accelerator.h"
#include "api/cuda/Mode.h"

namespace __impl { namespace memory {

hostptr_t HostMappedAlloc(size_t size)
{
    cuda::Mode &mode = cuda::Mode::current();
    hostptr_t ret = NULL;
    if(mode.hostAlloc(&ret, size) != gmacSuccess) return NULL;
    return ret;
}

void HostMappedFree(hostptr_t addr)
{
    cuda::Mode &mode = cuda::Mode::current();
    mode.hostFree(addr);
}

accptr_t HostMappedPtr(const hostptr_t addr)
{
    cuda::Mode &mode = cuda::Mode::current();
    return mode.hostMap(addr);
}

} }


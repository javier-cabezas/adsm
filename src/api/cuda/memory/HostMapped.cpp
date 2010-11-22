#include "api/cuda/Accelerator.h"
#include "api/cuda/Mode.h"

namespace __impl { namespace memory {

void *HostMappedAlloc(size_t size)
{
    cuda::Mode &mode = cuda::Mode::current();
    void *ret = NULL;
    if(mode.hostAlloc(&ret, size) != gmacSuccess) return NULL;
    return ret;
}

void HostMappedFree(void *addr)
{
    cuda::Mode &mode = cuda::Mode::current();
    mode.hostFree(addr);
}

void *HostMappedPtr(const void *addr)
{
    cuda::Mode &mode = cuda::Mode::current();
    return mode.hostMap(addr);
}

} }


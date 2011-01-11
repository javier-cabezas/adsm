#include "memory/HostMappedObject.h"
#if 0
#include "api/opencl/Accelerator.h"
#include "api/opencl/Mode.h"
#endif
namespace __impl { namespace memory {

hostptr_t HostMappedAlloc(size_t size)
{
#if 0
    opencl::Mode &mode = cuda::Mode::current();
    hostptr_t ret = NULL;
    if(mode.hostAlloc(&ret, size) != gmacSuccess) return NULL;
    return ret;
#endif
return NULL;
}

void HostMappedFree(hostptr_t addr)
{
#if 0
    opencl::Mode &mode = opencl::Mode::current();
    mode.hostFree(addr);
#endif
}

accptr_t HostMappedPtr(const hostptr_t addr)
{
#if 0
    opencl::Mode &mode = opencl::Mode::current();
    return mode.hostMap(addr);
#endif
    return accptr_t(NULL);
}

} }


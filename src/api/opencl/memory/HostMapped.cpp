#include "memory/HostMappedObject.h"
#include "api/opencl/Accelerator.h"
#include "api/opencl/Mode.h"

namespace __impl { namespace memory {

hostptr_t HostMappedAlloc(size_t size)
{
    opencl::Mode &mode = opencl::Mode::current();
    hostptr_t ret = NULL;
    if(mode.hostAlloc(&ret, size) != gmacSuccess) return NULL;
    return ret;
}

void HostMappedFree(hostptr_t addr)
{
    opencl::Mode &mode = opencl::Mode::current();
    mode.hostFree(addr);
}

accptr_t HostMappedPtr(const hostptr_t addr)
{
    opencl::Mode &mode = opencl::Mode::current();
    return mode.hostMap(addr);
}

} }


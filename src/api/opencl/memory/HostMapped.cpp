#include "memory/HostMappedObject.h"
#include "api/opencl/Mode.h"

namespace __impl { namespace memory {

hostptr_t
HostMappedObject::alloc(core::Mode &current)
{
    opencl::Mode &mode = dynamic_cast<opencl::Mode &>(current);
    hostptr_t ret = NULL;
    if(mode.hostAlloc(ret, size_) != gmacSuccess) return NULL;
    return ret;
}

void
HostMappedObject::free(core::Mode &current)
{
    opencl::Mode &mode = dynamic_cast<opencl::Mode &>(current);
    mode.hostFree(addr_);
}

accptr_t
HostMappedObject::getAccPtr(core::Mode &current) const
{
    opencl::Mode &mode = dynamic_cast<opencl::Mode &>(current);
    return mode.hostMapAddr(addr_);
}

} }


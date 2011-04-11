#include "memory/HostMappedObject.h"
#include "api/cuda/Mode.h"

namespace __impl { namespace memory {

hostptr_t
HostMappedObject::alloc(core::Mode &current)
{
    cuda::Mode &mode = dynamic_cast<cuda::Mode &>(current);
    hostptr_t ret = NULL;
    if(mode.hostAlloc(&ret, size_) != gmacSuccess) return NULL;
    return ret;
}

void
HostMappedObject::free(core::Mode &current)
{
    cuda::Mode &mode = dynamic_cast<cuda::Mode &>(current);
    mode.hostFree(addr_);
}

accptr_t
HostMappedObject::getAccPtr(core::Mode &current) const
{
    cuda::Mode &mode = dynamic_cast<cuda::Mode &>(current);
    return mode.hostMap(addr_);
}

} }


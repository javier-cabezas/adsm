#include "memory/HostMappedObject.h"
#include "api/cuda/Accelerator.h"
#include "api/cuda/Mode.h"

namespace __impl { namespace memory {

hostptr_t
HostMappedObject::alloc()
{
    cuda::Mode &mode = cuda::Mode::getCurrent();
    hostptr_t ret = NULL;
    if(mode.hostAlloc(&ret, size_) != gmacSuccess) return NULL;
    return ret;
}

void
HostMappedObject::free()
{
    cuda::Mode &mode = cuda::Mode::getCurrent();
    mode.hostFree(addr_);
}

accptr_t
HostMappedObject::getAccPtr() const
{
    cuda::Mode &mode = cuda::Mode::getCurrent();
    return mode.hostMap(addr_);
}

} }


#include "memory/HostMappedObject.h"
#include "api/opencl/Accelerator.h"
#include "api/opencl/Mode.h"

namespace __impl { namespace memory {

hostptr_t
HostMappedObject::alloc()
{
    opencl::Mode &mode = opencl::Mode::getCurrent();
    hostptr_t ret = NULL;
    if(mode.hostAlloc(ret, size_) != gmacSuccess) return NULL;
    return ret;
}

void
HostMappedObject::free()
{
    opencl::Mode &mode = opencl::Mode::getCurrent();
    mode.hostFree(addr_);
}

accptr_t
HostMappedObject::getAccPtr() const
{
    opencl::Mode &mode = opencl::Mode::getCurrent();
    return mode.hostMapAddr(addr_);
}

} }


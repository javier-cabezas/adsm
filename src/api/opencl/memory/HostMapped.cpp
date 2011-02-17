#include "memory/HostMappedObject.h"
#include "api/opencl/Accelerator.h"
#include "api/opencl/Mode.h"

namespace __impl { namespace memory {

hostptr_t
HostMappedObject::alloc()
{
    opencl::Mode &mode = opencl::Mode::getCurrent();
    hostptr_t ret = Memory::map(NULL, size_, GMAC_PROT_READWRITE);
    if(mode.hostMap(ret, size_) != gmacSuccess) return NULL;
    return ret;
}

void
HostMappedObject::free()
{
    opencl::Mode &mode = opencl::Mode::getCurrent();
	// TODO: Use a different method for this purpose
    //mode.hostFree(addr_);
    Memory::unmap(addr_, size_);
}

accptr_t
HostMappedObject::getAccPtr() const
{
    opencl::Mode &mode = opencl::Mode::getCurrent();
    return mode.hostMapAddr(addr_);
}

} }


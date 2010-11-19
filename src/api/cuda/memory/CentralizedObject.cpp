#include <memory/CentralizedObject.h>

#include "../Mode.h"

namespace __impl { namespace memory {

#ifndef USE_MMAP
CentralizedObject::CentralizedObject(size_t size) :
    Object(NULL, size)
{
    TRACE(LOCAL,"Creating Centralized Object ("FMT_SIZE" bytes)", size_);
}


CentralizedObject::~CentralizedObject() {}

gmacError_t CentralizedObject::init()
{
    cuda::Mode &mode = cuda::Mode::current();
    if(mode.hostAlloc(&addr_, size_) != gmacSuccess) {
        addr_ = NULL;
        return gmacErrorMemoryAllocation;
    }
    TRACE(LOCAL,"Centralized Object @ %p initialized", getAcceleratorAddr(addr_));

    return gmacSuccess;
}

void CentralizedObject::fini()
{
    if(addr_ == NULL) return;
    TRACE(LOCAL,"Centralized Object @ %p finalized", getAcceleratorAddr(addr_));
    cuda::Mode &mode = cuda::Mode::current();
    mode.hostFree(addr_);
}

void *CentralizedObject::getAcceleratorAddr(void *addr) const
{
    off_t offset = (unsigned long)addr - (unsigned long)addr_;
    cuda::Mode &mode = cuda::Mode::current();
    return (uint8_t *)mode.hostMap(addr_) + offset;
}
#endif

}}

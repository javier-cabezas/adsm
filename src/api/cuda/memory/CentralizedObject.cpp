#include <memory/CentralizedObject.h>

#include "../Mode.h"

namespace gmac { namespace memory {

#ifndef USE_MMAP
CentralizedObject::CentralizedObject(size_t size) :
    Object(NULL, size)
{
    TRACE(LOCAL,"Creating Centralized Object (%zd bytes)", size_);
}


CentralizedObject::~CentralizedObject() {}

gmacError_t CentralizedObject::init()
{
    gmac::cuda::Mode &mode = gmac::cuda::Mode::current();
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
    gmac::cuda::Mode &mode = gmac::cuda::Mode::current();
    mode.hostFree(addr_);
}

void *CentralizedObject::getAcceleratorAddr(void *addr) const
{
    off_t offset = (unsigned long)addr - (unsigned long)addr_;
    gmac::cuda::Mode &mode = gmac::cuda::Mode::current();
    return (uint8_t *)mode.hostMap(addr_) + offset;
}
#endif

}}

#include <memory/CentralizedObject.h>

#include "../Mode.h"

namespace gmac { namespace memory {

#ifndef USE_MMAP
CentralizedObject::CentralizedObject(size_t size) :
    Object(NULL, size)
{
    trace("Creating Centralized Object (%zd bytes)", size_);
}


CentralizedObject::~CentralizedObject() {}

void CentralizedObject::init()
{
    gmac::cuda::Mode &mode = gmac::cuda::Mode::current();
    if(mode.hostAlloc(&addr_, size_) != gmacSuccess) {
        addr_ = NULL;
        return;
    }
    trace("Centralized Object @ %p initialized", device(addr_));
}

void CentralizedObject::fini()
{
    if(addr_ == NULL) return;
    trace("Centralized Object @ %p finalized", device(addr_));
    gmac::cuda::Mode &mode = gmac::cuda::Mode::current();
    mode.hostFree(addr_);
}

void *CentralizedObject::device(void *addr) const
{
    off_t offset = (unsigned long)addr - (unsigned long)addr_;
    gmac::cuda::Mode &mode = gmac::cuda::Mode::current();
    return (uint8_t *)mode.hostMap(addr_) + offset;
}
#endif

}}

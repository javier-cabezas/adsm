#include <memory/CentralizedObject.h>

#include "../Mode.h"

namespace gmac { namespace memory {

#ifndef USE_MMAP
CentralizedObject::CentralizedObject(size_t size) :
    Object(NULL, size)
{
    trace("Creating Centralized Object (%zd bytes)", _size);
    gmac::cuda::Mode &mode = gmac::cuda::Mode::current();
    if(mode.hostAlloc(&_addr, size) != gmacSuccess) {
        _addr = NULL;
        return;
    }
    trace("Centralized Object @ %p to mode %p", device(_addr), &mode);
}

CentralizedObject::~CentralizedObject()
{
    if(_addr == NULL) return;
    gmac::cuda::Mode &mode = gmac::cuda::Mode::current();
    mode.hostFree(_addr);
}

void *CentralizedObject::device(void *addr) const
{
    off_t offset = (unsigned long)addr - (unsigned long)_addr;
    gmac::cuda::Mode &mode = gmac::cuda::Mode::current();
    return (uint8_t *)mode.hostMap(_addr) + offset;
}
#endif

}}

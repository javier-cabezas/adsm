#include <memory/CentralizedObject.h>

#include "Mode.h"

namespace gmac { namespace memory {

#ifndef USE_MMAP
CentralizedObject::CentralizedObject(size_t size) :
    Object(NULL, size)
{
    trace("Creating Centralized Object (%zd bytes)", __size);
    gmac::gpu::Mode *mode = dynamic_cast<gmac::gpu::Mode *>(gmac::Mode::current());
    if(mode->hostAlloc(&__addr, size) != gmacSuccess) {
        __addr = NULL;
        return;
    }
    trace("Centralized Object @ %p to mode %p", device(__addr), mode);
}

CentralizedObject::~CentralizedObject()
{
    if(__addr == NULL) return;
    gmac::gpu::Mode *mode = dynamic_cast<gmac::gpu::Mode *>(gmac::Mode::current());
    mode->hostFree(__addr);
}

void *CentralizedObject::device(void *addr)
{
    off_t offset = (unsigned long)addr - (unsigned long)__addr;
    gmac::gpu::Mode *mode = dynamic_cast<gmac::gpu::Mode *>(gmac::Mode::current());
    return (uint8_t *)mode->hostAddress(__addr) + offset;
}
#endif

}}

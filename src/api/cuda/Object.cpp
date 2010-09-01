#include <memory/Object.h>

#include "Mode.h"

namespace gmac { namespace memory {

CentralizedObject::CentralizedObject(size_t size) :
    Object(NULL, size)
{
    gmac::gpu::Mode *mode = dynamic_cast<gmac::gpu::Mode *>(gmac::Mode::current());
    mode->hostAlloc(&__addr, size);
}

CentralizedObject::~CentralizedObject()
{
    gmac::gpu::Mode *mode = dynamic_cast<gmac::gpu::Mode *>(gmac::Mode::current());
    mode->hostFree(__addr);
}

void *CentralizedObject::device(void *addr) const
{
    off_t offset = (unsigned long)addr - (unsigned long)__addr;
    gmac::gpu::Mode *mode = dynamic_cast<gmac::gpu::Mode *>(gmac::Mode::current());
    return (uint8_t *)mode->hostAddress(addr) + offset;
}

}}

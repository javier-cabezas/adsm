#include "memory/Object.h"
#include "memory/os/Memory.h"

#include <sys/mman.h>

namespace gmac { namespace memory {

void *Object::map(void *addr, size_t count)
{
    return Memory::map(addr, count, PROT_READ);
}

void Object::unmap(void *addr, size_t count)
{
    Memory::unmap(addr, count);
}

}}

#include "config/config.h"

#include "Object.h"
#if defined(POSIX)
#include "memory/posix/Memory.h"
#elif defined(WINDOWS
#include "memory/windows/Memory.h"
#endif

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

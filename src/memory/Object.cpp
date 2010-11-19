#include "config/config.h"

#include "Object.h"
#include "memory/Memory.h"

namespace __impl { namespace memory {

void *Object::map(void *addr, size_t count)
{
	return Memory::map(addr, count, GMAC_PROT_READ);
}

void Object::unmap(void *addr, size_t count)
{
    Memory::unmap(addr, count);
}

}}

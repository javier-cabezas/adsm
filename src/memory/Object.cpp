#include "config/config.h"

#include "Object.h"
#include "memory/Memory.h"

namespace gmac { namespace memory {

void *Object::map(void *addr, size_t count)
{
	return Memory::map(addr, count, Memory::Read);
}

void Object::unmap(void *addr, size_t count)
{
    Memory::unmap(addr, count);
}

}}

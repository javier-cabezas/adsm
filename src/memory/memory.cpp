#include "Manager.h"
#include "Allocator.h"

#include "allocator/Slab.h"

namespace gmac { namespace core {

void memoryInit(const char *, const char *)
{
	TRACE(GLOBAL, "Initializing Memory Subsystem");
    memory::Manager::create<memory::Manager>();
    memory::Allocator::create<memory::allocator::Slab>();
}

void memoryFini(void)
{
	TRACE(GLOBAL, "Cleaning Memory Subsystem");
    memory::Allocator::destroy();
    memory::Manager::destroy();
}


}}

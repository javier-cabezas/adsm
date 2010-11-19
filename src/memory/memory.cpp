#include "Manager.h"
#include "Allocator.h"

#include "allocator/Slab.h"

namespace __impl { namespace core {

void memoryInit(const char *, const char *)
{
	TRACE(GLOBAL, "Initializing Memory Subsystem");
    memory::Manager::create<gmac::memory::Manager>();
    memory::Allocator::create<__impl::memory::allocator::Slab>();
}

void memoryFini(void)
{
	TRACE(GLOBAL, "Cleaning Memory Subsystem");
    memory::Allocator::destroy();
    memory::Manager::destroy();
}


}}

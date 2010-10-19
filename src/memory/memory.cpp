#include "Manager.h"
#include "Allocator.h"

#include "allocator/Slab.h"

namespace gmac {

void memoryInit(const char *managerName, const char *allocatorName)
{
	util::Logger::TRACE("Initializing Memory Subsystem");
    memory::Manager::create<memory::Manager>();
    memory::Allocator::create<memory::allocator::Slab>();
}

void memoryFini(void)
{
	util::Logger::TRACE("Cleaning Memory Subsystem");
    memory::Allocator::destroy();
    memory::Manager::destroy();
}


}

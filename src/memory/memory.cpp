#include <memory/Manager.h>
#include <memory/Allocator.h>


namespace gmac {

memory::Manager *manager= NULL;
memory::Allocator *allocator = NULL;

void memoryInit(const char *managerName, const char *allocatorName)
{
	util::Logger::TRACE("Initializing Memory Subsystem");
    manager = memory::Manager::create();
    allocator = memory::Allocator::create();
}

void memoryFini(void)
{
	util::Logger::TRACE("Cleaning Memory Subsystem");
    memory::Allocator::destroy(); allocator = NULL;
    memory::Manager::destroy(); manager = NULL;
    
}


}

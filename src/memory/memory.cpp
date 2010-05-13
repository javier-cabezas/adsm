#include "Manager.h"
#include "Allocator.h"

#include <util/Lock.h>
#include <debug.h>

gmac::memory::Manager *manager= NULL;
gmac::memory::Allocator *allocator = NULL;

namespace gmac {

static util::Lock *mutex = NULL;

static void destroyManager(void)
{
	mutex->lock();
	if(manager != NULL) delete manager;
	manager = NULL;
	mutex->unlock();
}

static void destroyAllocator(void)
{
    mutex->lock();
    if(allocator != NULL) delete allocator;
    allocator = NULL;
    mutex->unlock();
}

static void createManager(const char *name)
{
	mutex->lock();
	if(manager == NULL)
		manager = memory::getManager(name);
	mutex->unlock();
}

static void createAllocator(const char *name)
{
    mutex->lock();
    if(allocator == NULL)
        allocator = memory::getAllocator(manager, name);
    mutex->unlock();
}

void memoryInit(const char *manager, const char *allocator)
{
	TRACE("Initializing Memory Subsystem");
	mutex = new util::Lock(LockManager);
	memory::Map::init();
	createManager(manager);
    createAllocator(allocator);
}

void memoryFini(void)
{
	TRACE("Cleaning Memory Subsystem");
    destroyAllocator();
	destroyManager();
	delete mutex;
}


}

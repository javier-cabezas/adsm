#include "Manager.h"

#include <util/Lock.h>
#include <debug.h>

gmac::memory::Manager *manager= NULL;

namespace gmac {

static util::Lock *mutex = NULL;

static void destroyManager(void)
{
	mutex->lock();
	delete manager;
	manager = NULL;
	mutex->unlock();
}

static void createManager(const char *name)
{
	mutex->lock();
	if(manager == NULL)
		manager = memory::getManager(name);
	mutex->unlock();
}

void memoryInit(const char *manager)
{
	TRACE("Initializing Memory Subsystem");
	mutex = new util::Lock(LockManager);
	memory::Map::init();
	createManager(manager);
}

void memoryFini(void)
{
	TRACE("Cleaning Memory Subsystem");
	destroyManager();
	delete mutex;
}


}

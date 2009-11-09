#include "Manager.h"

#include <debug.h>

gmac::memory::Manager *manager= NULL;

namespace gmac {

static MUTEX(mutex);

static void destroyManager(void)
{
	MUTEX_LOCK(mutex);
	delete manager;
	manager = NULL;
	MUTEX_UNLOCK(mutex);
}

static void createManager(const char *name)
{
	MUTEX_LOCK(mutex);
	if(manager == NULL)
		manager = gmac::memory::getManager(name);
	MUTEX_UNLOCK(mutex);
}

void memoryInit(const char *manager)
{
	TRACE("Initializing Memory Subsystem");
	MUTEX_INIT(mutex);
	gmac::memory::Map::init();
	createManager(manager);
}

void memoryFini(void)
{
	TRACE("Cleaning Memory Subsystem");
	destroyManager();
}


}

#include "MemManager.h"

#include <debug.h>

extern gmac::MemManager *memManager;

namespace gmac {

static MUTEX(mutex);
static unsigned count;

void memoryInit(void)
{
	TRACE("Initializing Memory Subsystem");
	MUTEX_INIT(mutex);
	count = 0;
}

void destroyManager(void)
{
	MUTEX_LOCK(mutex);
	count--;
	if(count <= 0) {
		delete memManager;
		memManager = NULL;
	}
	MUTEX_UNLOCK(mutex);
}

void createManager(const char *manager)
{
	MUTEX_LOCK(mutex);
	if(memManager == NULL)
		memManager = gmac::getManager(manager);
	count++;
	MUTEX_UNLOCK(mutex);
}

}

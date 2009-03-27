#include "BatchManager.h"
#include "paraver.h"

#include <config/debug.h>
#include <acc/api.h>

namespace gmac {
BatchManager::~BatchManager()
{
	Map::const_iterator i;
	MUTEX_LOCK(memMutex);
	for(i = memMap.begin(); i != memMap.end(); i++)
		delete i->second;
	memMap.clear();
	MUTEX_UNLOCK(memMutex);
}
void BatchManager::flush(void)
{
	Map::const_iterator i;
	MUTEX_LOCK(memMutex);
	for(i = memMap.begin(); i != memMap.end(); i++) {
		if(i->second->isOwner() == false) continue;
		TRACE("DMA To Device");
		__gmacMemcpyToDevice(safe(i->first), i->first, i->second->getSize());
	}
	MUTEX_UNLOCK(memMutex);
}

void BatchManager::sync(void)
{
	Map::const_iterator i;
	MUTEX_LOCK(memMutex);
	for(i = memMap.begin(); i != memMap.end(); i++) {
		if(i->second->isOwner() == false) continue;
		TRACE("DMA From Device");
		__gmacMemcpyToHost(i->first, safe(i->first), i->second->getSize());
	}
	MUTEX_UNLOCK(memMutex);
}

};

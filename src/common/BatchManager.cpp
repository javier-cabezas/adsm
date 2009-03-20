#include "BatchManager.h"
#include "paraver.h"
#include "debug.h"

#include <cuda_runtime.h>

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
void BatchManager::execute(void)
{
	Map::const_iterator i;
	MUTEX_LOCK(memMutex);
	for(i = memMap.begin(); i != memMap.end(); i++) {
		if(i->second->isOwner() == false) continue;
		TRACE("DMA To Device");
		cudaMemcpy(safe(i->first), i->first, i->second->getSize(),
				cudaMemcpyHostToDevice);
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
		cudaMemcpy(i->first, safe(i->first), i->second->getSize(),
				cudaMemcpyDeviceToHost);
	}
	MUTEX_UNLOCK(memMutex);
}
};

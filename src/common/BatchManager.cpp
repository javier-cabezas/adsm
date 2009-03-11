#include "BatchManager.h"

#include "debug.h"

#include <cuda_runtime.h>

namespace gmac {
void BatchManager::execute(void)
{
	HASH_MAP<void *, MemRegion *>::const_iterator i;
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
	HASH_MAP<void *, MemRegion *>::const_iterator i;
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

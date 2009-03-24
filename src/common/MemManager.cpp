#include "MemManager.h"
#include "BatchManager.h"
#include "LazyManager.h"
#include "CacheManager.h"

#include "debug.h"

#include <strings.h>


namespace gmac {

MemManager *getManager(const char *managerName)
{
	if(managerName == NULL) return new CacheManager();
	TRACE("Using %s Manager", managerName);
	if(strcasecmp(managerName, "None") == 0)
		return NULL;
	else if(strcasecmp(managerName, "Lazy") == 0)
		return new LazyManager();
	else if(strcasecmp(managerName, "Batch") == 0)
		return new BatchManager();
	return new CacheManager();
}

void MemManager::insertVirtual(void *cpuPtr, void *devPtr, size_t count)
{
	uint8_t *cpuAddr = (uint8_t *)cpuPtr;
	uint8_t *devAddr = (uint8_t *)devPtr;
	count += ((unsigned long)cpuPtr & (pageSize -1));
	MUTEX_LOCK(virtMutex);
	for(size_t off = 0; off < count; off += pageSize)
		virtTable[cpuAddr + off] = devAddr + off;
	MUTEX_UNLOCK(virtMutex);
}

};

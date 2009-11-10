#include "Manager.h"
#include "BatchManager.h"
#include "LazyManager.h"
#include "RollingManager.h"

#include <debug.h>

#include <strings.h>


namespace gmac { namespace memory {

Manager *getManager(const char *managerName)
{
	if(managerName == NULL) return new RollingManager();
	TRACE("Using %s Manager", managerName);
	if(strcasecmp(managerName, "None") == 0)
		return NULL;
	else if(strcasecmp(managerName, "Lazy") == 0)
		return new LazyManager();
	else if(strcasecmp(managerName, "Batch") == 0)
		return new BatchManager();
	return new RollingManager();
}

Region *Manager::remove(void *addr)
{
	Region *ret = gmac::Context::current()->mm().remove(addr);
	if(ret->owner() == gmac::Context::current()) {
		if(ret->relatives().empty()) Map::shared().erase(ret);
		else {
			// Change ownership
			ret->transfer();
			ret->owner()->mm().insert(ret);
		}
	}
	else ret->unrelate(gmac::Context::current());
	return ret;
}

void Manager::insertVirtual(Context *ctx, void *cpuPtr, void *devPtr, size_t count)
{
	TRACE("Virtual Request %p -> %p", cpuPtr, devPtr);
	uint8_t *cpuAddr = (uint8_t *)cpuPtr;
	uint8_t *devAddr = (uint8_t *)devPtr;
	gmac::memory::PageTable &pageTable = ctx->mm().pageTable();
	count += ((unsigned long)cpuPtr & (pageTable.getPageSize() -1));
	for(size_t off = 0; off < count; off += pageTable.getPageSize())
		pageTable.insert(cpuAddr + off, devAddr + off);
}

void Manager::removeVirtual(Context *ctx, void *cpuPtr, size_t count)
{
	uint8_t *cpuAddr = (uint8_t *)cpuPtr;
	gmac::memory::PageTable &pageTable = ctx->mm().pageTable();
	count += ((unsigned long)cpuPtr & (pageTable.getPageSize() -1));
	for(size_t off = 0; off < count; off += pageTable.getPageSize())
		pageTable.remove(cpuAddr + off);
}

void Manager::map(Context *ctx, void *cpuPtr, void *devPtr, size_t count)
{
	Region *region = gmac::Context::current()->mm().find<Region>(cpuPtr);
	assert(region != NULL); assert(region->size() == count);
	insertVirtual(ctx, cpuPtr, devPtr, count);
	region->relate(ctx);
}

void Manager::unmap(Context *ctx, void *cpuPtr)
{
	Region *region = gmac::Context::current()->mm().find<Region>(cpuPtr);
	assert(region != NULL);
	region->unrelate(ctx);
	removeVirtual(ctx, cpuPtr, region->size());
}

} };

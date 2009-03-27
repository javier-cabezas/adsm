#include "LazyManager.h"

#include <config/debug.h>
#include <acc/api.h>

namespace gmac {

bool LazyManager::alloc(void *addr, size_t count)
{
	if(map(addr, count, PROT_NONE) == MAP_FAILED) return false;
	TRACE("Alloc %p (%d bytes)", addr, count);
	ProtRegion *region = new ProtRegion(addr, count);
	MUTEX_LOCK(memMutex);
	memMap[addr] = region;
	MUTEX_UNLOCK(memMutex);
	return true;
}

void *LazyManager::safeAlloc(void *addr, size_t count)
{
	void *cpuAddr = NULL;
	if((cpuAddr = safeMap(addr, count, PROT_NONE)) == MAP_FAILED) return NULL;
	TRACE("SafeAlloc %p (%d bytes)", cpuAddr, count);
	ProtRegion *region = new ProtRegion(cpuAddr, count);
	MUTEX_LOCK(memMutex);
	memMap[cpuAddr] = region;
	MUTEX_UNLOCK(memMutex);
	return cpuAddr;
}

void LazyManager::release(void *addr)
{
	Map::const_iterator i;
	MUTEX_LOCK(memMutex);
	i = memMap.find(addr);
	if(i != memMap.end()) {
		unmap(addr, i->second->getSize());
		delete i->second;
		memMap.erase(addr);
	}
	MUTEX_UNLOCK(memMutex);
}

void LazyManager::flush()
{
	Map::const_iterator i;
	MUTEX_LOCK(memMutex);
	for(i = memMap.begin(); i != memMap.end(); i++) {
		if(i->second->isOwner() == false) continue;
		if(i->second->isDirty()) {
			TRACE("DMA to Device from %p (%d bytes)", i->first,
				i->second->getSize());
			__gmacMemcpyToDevice(safe(i->first), i->first, i->second->getSize());
		}
		i->second->noAccess();
	}
	MUTEX_UNLOCK(memMutex);
}

void LazyManager::sync()
{
}

ProtRegion *LazyManager::find(const void *addr)
{
	HASH_MAP<void *, ProtRegion *>::const_iterator i;
	MUTEX_LOCK(memMutex);
	for(i = memMap.begin(); i != memMap.end(); i++) {
		if(*(i->second) == addr) {
			MUTEX_UNLOCK(memMutex);
			return i->second;
		}
	}
	MUTEX_UNLOCK(memMutex);
	return NULL;
}

void LazyManager::read(ProtRegion *region, void *addr)
{
	TRACE("DMA from Device from %p (%d bytes)", region->getAddress(),
			region->getSize());
	region->readWrite();
	__gmacMemcpyToHost(region->getAddress(), safe(region->getAddress()), region->getSize());
	region->readOnly();
}

void LazyManager::write(ProtRegion *region, void *addr)
{
	bool present = region->isPresent();
	region->readWrite();
	if(present == false) {
		TRACE("DMA from Device from %p (%d bytes)", region->getAddress(),
				region->getSize());
		__gmacMemcpyToHost(region->getAddress(), safe(region->getAddress()), region->getSize());
	}
}

}

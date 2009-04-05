#include "LazyManager.h"

#include <config/debug.h>
#include <acc/api.h>

#include <assert.h>

namespace gmac {

// MemManager Interface

bool LazyManager::alloc(void *addr, size_t count)
{
	if(map(addr, count, PROT_NONE) == MAP_FAILED) return false;
	TRACE("Alloc %p (%d bytes)", addr, count);
	memMap.insert(new ProtRegion(addr, count));
	return true;
}


void *LazyManager::safeAlloc(void *addr, size_t count)
{
	void *cpuAddr = NULL;
	if((cpuAddr = safeMap(addr, count, PROT_NONE)) == MAP_FAILED) return NULL;
	TRACE("SafeAlloc %p (%d bytes)", cpuAddr, count);
	memMap.insert(new ProtRegion(cpuAddr, count));
	return cpuAddr;
}

void LazyManager::release(void *addr)
{
	ProtRegion *reg = memMap.remove(addr);
	assert(reg != NULL);
	unmap(reg->getAddress(), reg->getSize());
	delete reg;
}

void LazyManager::flush()
{
	MemMap<ProtRegion>::const_iterator i;
	memMap.lock();
	for(i = memMap.begin(); i != memMap.end(); i++) {
		if(i->second->isOwner() == false) continue;
		if(i->second->isDirty()) {
			__gmacMemcpyToDevice(safe(i->second->getAddress()),
					i->second->getAddress(), i->second->getSize());
		}
		i->second->invalidate();
	}
	memMap.unlock();
}

void LazyManager::flush(MemRegion *region)
{
	ProtRegion *r = dynamic_cast<ProtRegion *>(region);
	if(r->isDirty()) {
		__gmacMemcpyToDevice(safe(r->getAddress()), r->getAddress(),
				r->getSize());
	}
	r->invalidate();
}

// MemHandler Interface

void LazyManager::read(ProtRegion *region, void *addr)
{
	TRACE("DMA from Device from %p (%d bytes)", region->getAddress(),
			region->getSize());
	region->readWrite();
	__gmacMemcpyToHost(region->getAddress(), safe(region->getAddress()),
			region->getSize());
	region->readOnly();
}

void LazyManager::write(ProtRegion *region, void *addr)
{
	bool present = region->isPresent();
	region->readWrite();
	if(present == false) {
		TRACE("DMA from Device from %p (%d bytes)", region->getAddress(),
				region->getSize());
		__gmacMemcpyToHost(region->getAddress(), safe(region->getAddress()),
				region->getSize());
	}
}

}

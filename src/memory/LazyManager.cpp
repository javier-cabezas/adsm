#include "LazyManager.h"

#include <debug.h>
#include <kernel/Context.h>

#include <assert.h>

namespace gmac {

// MemManager Interface

bool LazyManager::alloc(void *addr, size_t count)
{
	if(map(addr, count, PROT_NONE) == MAP_FAILED) return false;
	TRACE("Alloc %p (%d bytes)", addr, count);
	insert(new ProtRegion(addr, count));
	return true;
}


void *LazyManager::safeAlloc(void *addr, size_t count)
{
	void *cpuAddr = NULL;
	if((cpuAddr = safeMap(addr, count, PROT_NONE)) == MAP_FAILED) return NULL;
	TRACE("SafeAlloc %p (%d bytes)", cpuAddr, count);
	insert(new ProtRegion(cpuAddr, count));
	return cpuAddr;
}

void LazyManager::release(void *addr)
{
	ProtRegion *reg = dynamic_cast<ProtRegion *>(remove(addr));
	assert(reg != NULL);
	unmap(reg->getAddress(), reg->getSize());
	delete reg;
}

void LazyManager::flush()
{
	MemMap::const_iterator i;
	MemMap &mm = current->mm();
	mm.lock();
	for(i = mm.begin(); i != mm.end(); i++) {
		ProtRegion *reg = dynamic_cast<ProtRegion *>(i->second);
		if(reg->dirty()) {
			current->copyToDevice(safe(i->second->getAddress()),
					i->second->getAddress(), i->second->getSize());
		}
		reg->invalidate();
	}
	mm.unlock();
}

void LazyManager::flush(MemRegion *region)
{
	ProtRegion *r = dynamic_cast<ProtRegion *>(region);
	if(r->dirty()) {
		current->copyToDevice(safe(r->getAddress()), r->getAddress(),
				r->getSize());
	}
	r->invalidate();
}

void LazyManager::dirty(MemRegion *region) 
{
	ProtRegion *r = dynamic_cast<ProtRegion *>(region);
	r->readWrite();
}

bool LazyManager::present(MemRegion *region) const
{
	ProtRegion *r = dynamic_cast<ProtRegion *>(region);
	return r->present();
}

// MemHandler Interface

void LazyManager::read(ProtRegion *region, void *addr)
{
	TRACE("DMA from Device from %p (%d bytes)", region->getAddress(),
			region->getSize());
	region->readWrite();
	region->context()->copyToHost(region->getAddress(),
			safe(region->getAddress()), region->getSize());
	region->readOnly();
}

void LazyManager::write(ProtRegion *region, void *addr)
{
	bool present = region->present();
	region->readWrite();
	if(present == false) {
		TRACE("DMA from Device from %p (%d bytes)", region->getAddress(),
				region->getSize());
		region->context()->copyToHost(region->getAddress(),
				safe(region->getAddress()), region->getSize());
	}
}

}

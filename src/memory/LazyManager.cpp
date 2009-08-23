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
	unmap(reg->start(), reg->size());
	delete reg;
}

void LazyManager::flush()
{
	MemMap::const_iterator i;
	current().lock();
	for(i = current().begin(); i != current().end(); i++) {
		ProtRegion *r = dynamic_cast<ProtRegion *>(i->second);
		if(r->dirty()) {
			r->context()->copyToDevice(safe(r->start()),
					r->start(), r->size());
		}
		r->invalidate();
	}
	current().unlock();
}

Context *LazyManager::owner(const void *addr)
{
	ProtRegion *region = get(addr);
	if(addr == NULL) return NULL;
	return region->context();
}

void LazyManager::invalidate(const void *addr, size_t size)
{
	ProtRegion *region = get(addr);
	assert(region != NULL);
	assert(region->end() >= (void *)((addr_t)addr + size));
	if(region->dirty()) {
		if(region->start() < addr ||
				region->end() > (void *)((addr_t)addr + size))
			region->context()->copyToDevice(safe(region->start()),
					region->start(), region->size());
	}
	region->invalidate();
}

void LazyManager::flush(const void *addr, size_t size)
{
	ProtRegion *region = get(addr);
	assert(region != NULL);
	assert(region->end() >= (void *)((addr_t)addr + size));
	if(region->dirty()) {
		region->context()->copyToDevice(safe(region->start()),
				region->start(), region->size());
	}
	region->readOnly();
}

#if 0
void LazyManager::flush(MemRegion *region)
{
	ProtRegion *r = dynamic_cast<ProtRegion *>(region);
	if(r->dirty()) {
		r->context()->copyToDevice(safe(r->start()), r->start(),
				r->size());
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
#endif

// MemHandler Interface

bool LazyManager::read(void *addr)
{
	ProtRegion *region = current().find<ProtRegion>(addr);
	if(region == NULL) region = mem.find<ProtRegion>(addr);
	if(region == NULL) return false;

	region->readWrite();
	region->context()->copyToHost(region->start(),
			safe(region->start()), region->size());
	region->readOnly();

	return true;
}

bool LazyManager::write(void *addr)
{
	ProtRegion *region = current().find<ProtRegion>(addr);
	if(region == NULL) region = mem.find<ProtRegion>(addr);
	if(region == NULL) return false;

	bool present = region->present();
	region->readWrite();
	if(present == false) {
		TRACE("DMA from Device from %p (%d bytes)", region->start(),
				region->size());
		region->context()->copyToHost(region->start(),
				safe(region->start()), region->size());
	}
}

}

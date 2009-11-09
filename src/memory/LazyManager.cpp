#include "LazyManager.h"

#include <debug.h>
#include <kernel/Context.h>

#include <assert.h>

namespace gmac { namespace memory {

// Manager Interface

void *LazyManager::alloc(void *addr, size_t count)
{
	void *cpuAddr = NULL;
	if((cpuAddr = map(addr, count, PROT_NONE)) == MAP_FAILED) return NULL;
	TRACE("Alloc %p (%d bytes)", cpuAddr, count);
	insert(new ProtRegion(cpuAddr, count));
	return cpuAddr;
}

void LazyManager::release(void *addr)
{
	ProtRegion *reg = dynamic_cast<ProtRegion *>(remove(ptr(addr)));
	assert(reg != NULL);
	unmap(reg->start(), reg->size());
	delete reg;
}

void LazyManager::flush()
{
	memory::Map::const_iterator i;
	for(i = current()->begin(); i != current()->end(); i++) {
		ProtRegion *r = dynamic_cast<ProtRegion *>(i->second);
		if(r->dirty()) {
			r->copyToDevice();
		}
		r->invalidate();
	}
	gmac::Context::current()->flush();
	gmac::Context::current()->invalidate();
}

Context *LazyManager::owner(const void *addr)
{
	ProtRegion *region = get(addr);
	if(region == NULL) return NULL;
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
			region->copyToDevice();
	}
	region->invalidate();
}

void LazyManager::flush(const void *addr, size_t size)
{
	ProtRegion *region = get(addr);
	assert(region != NULL);
	assert(region->end() >= (void *)((addr_t)addr + size));
	if(region->dirty()) {
		region->copyToDevice();
	}
	region->readOnly();
}

#if 0
void LazyManager::flush(Region *region)
{
	ProtRegion *r = dynamic_cast<ProtRegion *>(region);
	if(r->dirty()) {
		r->copyToDevice();
	}
	r->invalidate();
}

void LazyManager::dirty(Region *region) 
{
	ProtRegion *r = dynamic_cast<ProtRegion *>(region);
	r->readWrite();
}

bool LazyManager::present(Region *region) const
{
	ProtRegion *r = dynamic_cast<ProtRegion *>(region);
	return r->present();
}
#endif

// Handler Interface

bool LazyManager::read(void *addr)
{
	ProtRegion *region = get(addr);
	if(region == NULL) return false;

	region->readWrite();
	region->copyToHost();
	region->readOnly();

	return true;
}

bool LazyManager::write(void *addr)
{
	ProtRegion *region = get(addr);
	if(region == NULL) return false;

	bool present = region->present();
	region->readWrite();
	if(present == false) {
		TRACE("DMA from Device from %p (%d bytes)", region->start(),
				region->size());
		region->copyToHost();
	}

	return true;
}

} }

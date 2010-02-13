#include "LazyManager.h"

#include <debug.h>
#include <kernel/Context.h>

#include <cassert>

namespace gmac { namespace memory { namespace manager {

// Manager Interface
LazyManager::LazyManager()
{}

void *LazyManager::alloc(void *addr, size_t count, int attr)
{
    void *cpuAddr;

    if (attr == GMAC_MALLOC_PINNED) {
        Context * ctx = Context::current();
        void *hAddr;
        if (ctx->halloc(&hAddr, count) != gmacSuccess) return NULL;
        cpuAddr = hostRemap(addr, hAddr, count);
    } else {
        cpuAddr = hostMap(addr, count, PROT_NONE);
    }
    if (cpuAddr == NULL) return NULL;
	insertVirtual(cpuAddr, addr, count);
	insert(new ProtRegion(cpuAddr, count));
	TRACE("Alloc %p (%d bytes)", cpuAddr, count);
	return cpuAddr;
}

void LazyManager::release(void *addr)
{
	Region *reg = remove(addr);
	assert(reg != NULL);

    if(reg->owner() == Context::current()) {
        hostUnmap(reg->start(), reg->size());
        delete reg;
    }
	removeVirtual(reg->start(), reg->size());
}

void LazyManager::invalidate()
{
    TRACE("LazyManager Invalidation Starts");
    Map::const_iterator i;
    Map * m = current();
	for(i = m->begin(); i != m->end(); i++) {
        ProtRegion *r = dynamic_cast<ProtRegion *>(i->second);
		r->invalidate();
	}
	//gmac::Context::current()->flush();
    /// \todo Change to invalidate(regions)
	Context::current()->invalidate();
    TRACE("LazyManager Invalidation Ends");
}

void LazyManager::invalidate(const RegionSet & regions)
{
    if (regions.size() == 0) {
        invalidate();
        return;
    }

    TRACE("LazyManager Invalidation Starts");
	RegionSet::const_iterator i;
	for(i = regions.begin(); i != regions.end(); i++) {
        ProtRegion *r = dynamic_cast<ProtRegion *>(*i);
		r->invalidate();
	}
	//gmac::Context::current()->flush();
    /// \todo Change to invalidate(regions)
	Context::current()->invalidate();
    TRACE("LazyManager Invalidation Ends");
}

void LazyManager::flush()
{
    Map::const_iterator i;
    Map * m = current();
    m->lock();
	for(i = m->begin(); i != m->end(); i++) {
		ProtRegion *r = dynamic_cast<ProtRegion *>(i->second);
		if(r->dirty()) {
			r->copyToDevice();
		}
        r->readOnly();
	}
    m->unlock();
	//gmac::Context::current()->flush();
    /// \todo Change to invalidate(regions)
	Context::current()->invalidate();
}

void LazyManager::flush(const RegionSet & regions)
{
    if (regions.size() == 0) {
        flush();
        return;
    }

	RegionSet::const_iterator i;
	for(i = regions.begin(); i != regions.end(); i++) {
		ProtRegion *r = dynamic_cast<ProtRegion *>(*i);
		if(r->dirty()) {
			r->copyToDevice();
		}
        r->readOnly();
	}
	//gmac::Context::current()->flush();
    /// \todo Change to invalidate(regions)
	Context::current()->invalidate();
}

void LazyManager::invalidate(const void *addr, size_t size)
{
	ProtRegion *region = current()->find<ProtRegion>(addr);
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
	ProtRegion *region = current()->find<ProtRegion>(addr);
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
	ProtRegion *region = current()->find<ProtRegion>(addr);
	if(region == NULL) return false;

    Context * owner = region->owner();
    if (owner->status() == Context::RUNNING) owner->sync();

	region->readWrite();
	region->copyToHost();
	region->readOnly();

	return true;
}

bool LazyManager::write(void *addr)
{
	ProtRegion *region = current()->find<ProtRegion>(addr);
	if(region == NULL) return false;

    Context * owner = region->owner();
    if (owner->status() == Context::RUNNING) owner->sync();

	bool present = region->present();
	region->readWrite();
	if(present == false) {
		TRACE("DMA from Device from %p (%d bytes)", region->start(),
				region->size());
		region->copyToHost();
	}

	return true;
}

}}}

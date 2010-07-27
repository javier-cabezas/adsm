#include "LazyManager.h"

#include <kernel/Context.h>

namespace gmac { namespace memory { namespace manager {

Region *    
LazyManager::newRegion(void * addr, size_t count, bool shared)
{
    return new ProtRegion(addr, count, shared);
}

// Manager Interface
LazyManager::LazyManager()
{}


void LazyManager::invalidate()
{
	Context * ctx = Context::current();
    trace("LazyManager Invalidation Starts");

    Map::const_iterator i;
    Map * m = current();
    m->lockRead();
	for(i = m->begin(); i != m->end(); i++) {
        ProtRegion *r = dynamic_cast<ProtRegion *>(i->second);
        r->lockWrite();
		r->invalidate();
        r->unlock();
	}
    m->unlock();

    RegionMap::iterator s;
    RegionMap &shared = Map::shared();
    shared.lockRead();
    for (s = shared.begin(); s != shared.end(); s++) {
        ProtRegion * r = dynamic_cast<ProtRegion *>(s->second);
        r->lockWrite();
        if(r->owner() == ctx) {
            r->invalidate();
        }
        r->unlock();
    }
    shared.unlock();
	//gmac::Context::current()->flush();
    /// \todo Change to invalidate(regions)
	ctx->invalidate();
    trace("LazyManager Invalidation Ends");
}

void LazyManager::invalidate(const RegionSet & regions)
{
    if (regions.size() == 0) {
        invalidate();
        return;
    }

    trace("LazyManager Invalidation Starts");
	RegionSet::const_iterator i;
	for(i = regions.begin(); i != regions.end(); i++) {
        ProtRegion *r = dynamic_cast<ProtRegion *>(*i);
        r->lockWrite();
		r->invalidate();
        r->unlock();
	}
	//gmac::Context::current()->flush();
    /// \todo Change to invalidate(regions)
	Context::current()->invalidate();
    trace("LazyManager Invalidation Ends");
}

void LazyManager::flush()
{
    Context * ctx = Context::current();

    Map::const_iterator i;
    Map * m = current();
    m->lockRead();
	for(i = m->begin(); i != m->end(); i++) {
		ProtRegion *r = dynamic_cast<ProtRegion *>(i->second);
        r->lockWrite();
		if(r->dirty()) {
			r->copyToDevice();
		}
        r->readOnly();
        r->unlock();
	}
    m->unlock();

    RegionMap::iterator s;
    RegionMap &shared = Map::shared();
    shared.lockRead();
    for (s = shared.begin(); s != shared.end(); s++) {
        ProtRegion * r = dynamic_cast<ProtRegion *>(s->second);
        r->lockWrite();
        if(r->dirty()) {
            r->copyToDevice();
        }
        r->readOnly();
        r->unlock();
    }
    shared.unlock();

	//ctx->flush();
    /// \todo Change to invalidate(regions)
	ctx->invalidate();
}

void LazyManager::flush(const RegionSet & regions)
{
    if (regions.size() == 0) {
        flush();
        return;
    }

    Context * ctx = Context::current();
	RegionSet::const_iterator i;

	for(i = regions.begin(); i != regions.end(); i++) {
		ProtRegion *r = dynamic_cast<ProtRegion *>(*i);
        r->lockWrite();
		if(r->dirty()) {
			r->copyToDevice();
		}
        r->readOnly();
        r->unlock();
	}

	//gmac::Context::current()->flush();
    /// \todo Change to invalidate(regions)
	ctx->invalidate();
}

void LazyManager::invalidate(const void *addr, size_t size)
{
	ProtRegion *region = current()->find<ProtRegion>(addr);
	assertion(region != NULL);
    region->lockWrite();
	assertion(region->end() >= (void *)((addr_t)addr + size));
	if(region->dirty()) {
		if(region->start() < addr ||
				region->end() > (void *)((addr_t)addr + size))
			region->copyToDevice();
	}
	region->invalidate();
    region->unlock();
}

void LazyManager::flush(const void *addr, size_t size)
{
	ProtRegion *region = current()->find<ProtRegion>(addr);
	assertion(region != NULL);
    region->lockWrite();
	assertion(region->end() >= (void *)((addr_t)addr + size));
	if(region->dirty()) {
		region->copyToDevice();
	}
	region->readOnly();
    region->unlock();
}

void
LazyManager::map(Context *ctx, Region *r, void *devPtr)
{
	ProtRegion *region = dynamic_cast<ProtRegion *>(r);
	assertion(region != NULL);
	insertVirtual(ctx, region->start(), devPtr, region->size());
	region->relate(ctx);
    if (region->dirty() == false && region->present()) {
        ctx->copyToDevice(ptr(ctx, region->start()),
                          region->start(),
                          region->size());
    }
}


// Handler Interface

bool LazyManager::read(void *addr)
{
	ProtRegion *region = current()->find<ProtRegion>(addr);
	if(region == NULL) return false;

    region->lockWrite();

	region->readWrite();
	region->copyToHost();
	region->readOnly();

    region->unlock();

	return true;
}

bool LazyManager::write(void *addr)
{
	ProtRegion *region = current()->find<ProtRegion>(addr);
	if (region == NULL) return false;

    region->lockWrite();
	bool present = region->present();
	region->readWrite();
	if(present == false) {
		trace("DMA from Device from %p (%zd bytes)", (void *) region->start(),
				region->size());
		region->copyToHost();
	}

    region->unlock();

	return true;
}

bool LazyManager::touch(Region * r)
{
    assertion(r != NULL);
    ProtRegion *root = dynamic_cast<ProtRegion *>(r);
    if(root->dirty() == false) {
        root->readWrite();

        if(!root->present()) {
            gmacError_t ret = root->copyToHost();
            assertion(ret == gmacSuccess);
        }
    }

    return true;
}

}}}

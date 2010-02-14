#include "LazyManager.h"

#include <debug.h>
#include <kernel/Context.h>

#include <cassert>

namespace gmac { namespace memory { namespace manager {

// Manager Interface
LazyManager::LazyManager()
{}

void *LazyManager::alloc(void *addr, size_t count)
{
	void *cpuAddr = NULL;
	if((cpuAddr = hostMap(addr, count, PROT_NONE)) == NULL)
        return NULL;

	insertVirtual(cpuAddr, addr, count);
	insert(new ProtRegion(cpuAddr, count));
	TRACE("Alloc %p (%d bytes)", cpuAddr, count);
	return cpuAddr;
}

void LazyManager::release(void *addr)
{
	Region *reg = remove(addr);
	assert(reg != NULL);
	removeVirtual(reg->start(), reg->size());
    if(reg->owner() == Context::current()) {
    	hostUnmap(reg->start(), reg->size());
        delete reg;
    }
}

void LazyManager::flush()
{
    Context * ctx = Context::current();
    Process::SharedMap::iterator i;
	Map::const_iterator j;
	Process::SharedMap &sharedMem = proc->sharedMem();
	for(i = sharedMem.begin(); i != sharedMem.end(); i++) {
		ProtRegion * r = current()->find<ProtRegion>(i->second.start());
        if(r->dirty()) {
			r->copyToDevice();
		}
		r->invalidate();
	}

	current()->lock();
	for(j = current()->begin(); j != current()->end(); j++) {
		ProtRegion *r = dynamic_cast<ProtRegion *>(j->second);
		if(r->dirty()) {
			r->copyToDevice();
		}
		r->invalidate();
	}
	current()->unlock();
	ctx->flush();
	ctx->invalidate();
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

void
LazyManager::remap(Context *ctx, void *cpuPtr, void *devPtr, size_t count)
{
	ProtRegion *region = current()->find<ProtRegion>(cpuPtr);
	assert(region != NULL); assert(region->size() == count);
	insertVirtual(ctx, cpuPtr, devPtr, count);
	region->relate(ctx);
    if (!region->dirty()) {
        ctx->copyToDevice(ptr(region->start()),
                          region->start(),
                          count);
    }
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

	region->readWrite();
	region->copyToHost();
	region->readOnly();

	return true;
}

bool LazyManager::write(void *addr)
{
	ProtRegion *region = current()->find<ProtRegion>(addr);
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

}}}

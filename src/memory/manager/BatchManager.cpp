#include "BatchManager.h"

#include <debug.h>

#include <kernel/Context.h>

namespace gmac { namespace memory { namespace manager {

void BatchManager::release(void *addr)
{
	Region *reg = remove(addr);
	removeVirtual(reg->start(), reg->size());
    if(reg->owner() == Context::current()) {
	    hostUnmap(reg->start(), reg->size());
        delete reg;
    }
}
void BatchManager::flush(void)
{
    Process::SharedMap::iterator i;
	Map::const_iterator j;

    Context * ctx = Context::current();
	Process::SharedMap &sharedMem = proc->sharedMem();
	for(i = sharedMem.begin(); i != sharedMem.end(); i++) {
		ctx->copyToDevice(ptr(i->second.start()), i->second.start(), i->second.size());
	}

	current()->lock();
	for(j = current()->begin(); j != current()->end(); j++) {
		TRACE("Memory Copy to Device");
		ctx->copyToDevice(ptr(j->second->start()),
				j->second->start(), j->second->size());
	}
	current()->unlock();
	ctx->flush();
	ctx->sync();
}

void BatchManager::sync(void)
{
    Context * ctx = Context::current();
    Process::SharedMap::iterator i;
	Map::const_iterator j;
	Process::SharedMap &sharedMem = proc->sharedMem();
	for(i = sharedMem.begin(); i != sharedMem.end(); i++) {
		ctx->copyToHost(i->second.start(),
                        ptr(i->second.start()), i->second.size());
	}
	current()->lock();
	for(j = current()->begin(); j != current()->end(); j++) {
		TRACE("Memory Copy from Device");
		ctx->copyToHost(j->second->start(),
			        	  ptr(j->second->start()), j->second->size());
	}
	current()->unlock();
}

void
BatchManager::invalidate(const void *, size_t)
{
    // Do nothing
}

void
BatchManager::flush(const void *, size_t)
{
    // Do nothing
}

void
BatchManager::remap(Context *ctx, void *cpuPtr, void *devPtr, size_t count)
{
	Region *region = current()->find<Region>(cpuPtr);
	assert(region != NULL); assert(region->size() == count);
	insertVirtual(ctx, cpuPtr, devPtr, count);
	region->relate(ctx);
}



}}}

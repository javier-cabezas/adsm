#include "BatchManager.h"

#include <debug.h>

#include "kernel/Context.h"

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

void BatchManager::flush()
{
    Process::SharedMap::iterator i;
    Map::const_iterator j;

    Context * ctx = Context::current();
	Process::SharedMap &sharedMem = proc->sharedMem();
    for(i = sharedMem.begin(); i != sharedMem.end(); i++) {
		ctx->copyToDevice(ptr(i->second.start()), i->second.start(), i->second.size());
	}

    Map * m = current();
    m->lockRead();
    for(j = m->begin(); j != m->end(); j++) {
        TRACE("Memory Copy to Device");
        ctx->copyToDevice(ptr(j->second->start()),
                          j->second->start(),
                          j->second->size());
    }
    m->unlock();
    /*!
      \todo Fix vm
    */
    //ctx->flush();
    ctx->sync();
}

void BatchManager::flush(const RegionSet & regions)
{
    if (regions.size() == 0) {
        flush();
        return;
    }

    Process::SharedMap::iterator i;
    RegionSet::const_iterator j;
    Context * ctx = Context::current();
    Process::SharedMap &sharedMem = proc->sharedMem();
	for(i = sharedMem.begin(); i != sharedMem.end(); i++) {
		ctx->copyToDevice(ptr(i->second.start()), i->second.start(), i->second.size());
	}
    for(j = regions.begin(); j != regions.end(); j++) {
        TRACE("Memory Copy to Device");
        ctx->copyToDevice(ptr((*j)->start()),
                          (*j)->start(),
                          (*j)->size());
    }
    /*!
      \todo Fix vm
    */
    ctx->sync();
}

void
BatchManager::invalidate()
{
    // Do nothing
}

void
BatchManager::invalidate(const RegionSet & regions)
{
    // Do nothing
}

void BatchManager::sync()
{
    Map::const_iterator i;
    current()->lockRead();
    for(i = current()->begin(); i != current()->end(); i++) {
        TRACE("Memory Copy from Device");
        Context::current()->copyToHost(i->second->start(),
                                       ptr(i->second->start()), i->second->size());
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
    ASSERT(region != NULL); ASSERT(region->size() == count);
    insertVirtual(ctx, cpuPtr, devPtr, count);
    region->relate(ctx);
}

}}}

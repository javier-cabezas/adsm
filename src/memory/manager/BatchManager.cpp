#include "BatchManager.h"

#include <debug.h>

#include "kernel/Context.h"

namespace gmac { namespace memory { namespace manager {

void BatchManager::free(void *addr)
{
    Region *reg = remove(addr);
	ASSERT(reg != NULL);
	removeVirtual(reg->start(), reg->size());
    if(reg->owner() == Context::current()) {
	    hostUnmap(reg->start(), reg->size());
        delete reg;
    }
}

void BatchManager::flush()
{
    Map::const_iterator i;

    Context * ctx = Context::current();
    Map * m = current();
    m->lockRead();
    for(i = m->begin(); i != m->end(); i++) {
        TRACE("Memory Copy to Device");
        Region * r = i->second;
        r->lockRead();
        r->copyToDevice();
        r->unlock();
    }
    m->unlock();

    RegionMap::iterator s;
    RegionMap &shared = Map::shared();
    shared.lockRead();
    for (s = shared.begin(); s != shared.end(); s++) {
        TRACE("Shared Memory Copy to Device");
        Region * r = s->second;
        r->lockRead();
        r->copyToDevice();
        r->unlock();
    }
    shared.unlock();

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

    Context * ctx = Context::current();
    RegionSet::iterator i;
    for(i = regions.begin(); i != regions.end(); i++) {
        Region * r = *i;
        r->lockRead();
        r->copyToDevice();
        r->unlock();
    }
    /*!
      \todo Fix vm
    */
    ctx->sync();
}

void
BatchManager::invalidate()
{
    Map::const_iterator i;
    current()->lockRead();
    for(i = current()->begin(); i != current()->end(); i++) {
        Region * r = i->second;
        r->lockRead();
        r->copyToHost();
        r->unlock();
    }
    current()->unlock();

    Context * ctx = Context::current();
    RegionMap::iterator s;
    RegionMap &shared = Map::shared();
    shared.lockRead();
    for (s = shared.begin(); s != shared.end(); s++) {
        Region * r = s->second;
        r->lockRead();
        if(r->owner() == ctx) {
            r->copyToHost();
        }
        r->unlock();
    }
    shared.unlock();
}

void
BatchManager::invalidate(const RegionSet & regions)
{
    if (regions.size() == 0) {
        invalidate();
        return;
    }

    Context * ctx = Context::current();
    RegionSet::const_iterator i;
    for(i = regions.begin(); i != regions.end(); i++) {
        Region * r = *i;
        r->lockRead();
        if (r->shared() == false || r->owner() == ctx) {
            r->copyToHost();
        }
        r->unlock();
    }
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
BatchManager::remap(Context *ctx, Region *r, void *devPtr)
{
    ASSERT(r != NULL);
    r->lockWrite();
    insertVirtual(ctx, r->start(), devPtr, r->size());
    if (ctx != r->owner()) {
        r->relate(ctx);
    }
    r->unlock();
}

}}}

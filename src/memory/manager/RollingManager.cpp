#include "RollingManager.h"
#include "os/Memory.h"

#include <kernel/Context.h>

#include <unistd.h>

#include <typeinfo>

namespace gmac { namespace memory { namespace manager {

RollingBuffer::RollingBuffer() :
    RWLock(LockRollingBuffer),
    _max(paramLruDelta)
    {}

RollingMap::~RollingMap()
{
    RollingMap::iterator i;
    for(i = this->begin(); i != this->end(); i++)
        delete i->second;
}

void RollingManager::writeBack()
{
    // Get the buffer for the current thread
    RollingBlock *r = rollingMap.currentBuffer()->pop();
    r->lockWrite();
    flush(r);
    r->unlock();
}

Region *RollingManager::newRegion(void * addr, size_t count, bool shared)
{
    rollingMap.currentBuffer()->inc(lruDelta);
    return new RollingRegion(*this, addr, count, shared, pageTable().getPageSize());
}

RollingManager::RollingManager() :
    Handler(),
    lruDelta(0),
    writeMutex(LockWriteMutex),
    writeBuffer(NULL),
    writeBufferSize(0)
{
    lruDelta = paramLruDelta;
    trace("Using %zd as LRU Delta Size", lruDelta);
}

RollingManager::~RollingManager()
{
}



void RollingManager::flush()
{
    trace("RollingManager Flush Starts");
    Context * ctx = Context::current();

    // We need to go through all regions from the context because
    // other threads might have regions owned by this context in
    // their flush buffer
    Map::iterator i;
    Map * m = current();
    m->lockRead();
    for(i = m->begin(); i != m->end(); i++) {
        RollingRegion *r = dynamic_cast<RollingRegion *>(i->second);
        r->lockWrite();
        r->flush();
        r->unlock();
    }
    m->unlock();

    RegionMap::iterator s;
    RegionMap &shared = Map::shared();
    shared.lockRead();
    for (s = shared.begin(); s != shared.end(); s++) {
        RollingRegion * r = dynamic_cast<RollingRegion *>(s->second);
        r->lockWrite();
        r->transferDirty();
        r->unlock();
    }
    shared.unlock();


    /** \todo Fix vm */
    ctx->flush();
    trace("RollingManager Flush Ends");
}

void RollingManager::flush(const RegionSet & regions)
{
    // If no dependencies, a global flush is assumed
    if (regions.size() == 0) {
        flush();
        return;
    }

    trace("RollingManager Flush Starts");
    Context * ctx = Context::current();
    size_t blocks = rollingMap.currentBuffer()->size();

    for(unsigned j = 0; j < blocks; j++) {
        RollingBlock *r = rollingMap.currentBuffer()->pop();
        r->lockWrite();
        // Check if we have to flush
        if(std::find(regions.begin(), regions.end(), &r->getParent()) == regions.end()) {
            rollingMap.currentBuffer()->push(r);
            r->unlock();
            continue;
        }
        flush(r);
        r->unlock();
        trace("Flush to Device %p", r->start());
    }

    /** \todo Fix vm */
    ctx->flush();
    trace("RollingManager Flush Ends");
}

void RollingManager::invalidate()
{
    Context * ctx = Context::current();
#ifdef USE_VM
    ctx->mm().dirtyBitmap().synced(false);
#endif

    trace("RollingManager Invalidation Starts");
    Map::iterator i;
    Map * m = current();
    m->lockRead();
    for(i = m->begin(); i != m->end(); i++) {
        RollingRegion *r = dynamic_cast<RollingRegion *>(i->second);
        r->lockWrite();
        r->invalidate();
        r->unlock();
    }
    m->unlock();

    RegionMap::iterator s;
    RegionMap &shared = Map::shared();
    shared.lockRead();
    for (s = shared.begin(); s != shared.end(); s++) {
        RollingRegion * r = dynamic_cast<RollingRegion *>(s->second);
        r->lockWrite();
        if(r->owner() == ctx) {
            r->invalidate();
        }
        r->unlock();
    }
    shared.unlock();

#ifdef USE_VM
#ifdef DEBUG
    gmac::memory::vm::Bitmap & bitmap = ctx->mm().dirtyBitmap(); 
    bitmap.dump();
#endif
#endif
    trace("RollingManager Invalidation Ends");
}

void RollingManager::invalidate(const RegionSet & regions)
{
    // If no dependencies, a global invalidation is assumed
    if (regions.size() == 0) {
        invalidate();
        return;
    }

    trace("RollingManager Invalidation Starts");
    RegionSet::const_iterator i;
    for(i = regions.begin(); i != regions.end(); i++) {
        RollingRegion *r = dynamic_cast<RollingRegion *>(*i);
        r->lockWrite();
        r->invalidate();
        r->unlock();
    }

    trace("RollingManager Invalidation Ends");
}

void RollingManager::invalidate(const void *addr, size_t size)
{
    RollingRegion *reg = current()->find<RollingRegion>(addr);
    assertion(reg != NULL);
    reg->lockWrite();
    assertion(reg->end() >= (void *)((addr_t)addr + size));
    reg->invalidate(addr, size);
    reg->unlock();
}

void RollingManager::flush(const void *addr, size_t size)
{
    RollingRegion *reg = current()->find<RollingRegion>(addr);
    assertion(reg != NULL);
    reg->lockWrite();
    assertion(reg->end() >= (void *)((addr_t)addr + size));
    reg->flush(addr, size);
    reg->unlock();
}

//
// Handler Interface
//

bool RollingManager::read(void *addr)
{
#ifdef USE_VM
    Context * ctx = Context::current();
    gmac::memory::vm::Bitmap & bitmap = ctx->mm().dirtyBitmap(); 
    if (!bitmap.synced()) {
        bitmap.sync();

        Map::iterator i;
        Map * m = current();
        m->lockRead();
        for(i = m->begin(); i != m->end(); i++) {
            RollingRegion *r = dynamic_cast<RollingRegion *>(i->second);
            r->lockWrite();
            r->invalidateWithBitmap(PROT_READ);
            r->unlock();
        }
        m->unlock();

        RegionMap::iterator s;
        RegionMap &shared = Map::shared();
        shared.lockRead();
        for (s = shared.begin(); s != shared.end(); s++) {
            RollingRegion * r = dynamic_cast<RollingRegion *>(s->second);
            r->lockWrite();
            if(r->owner() == ctx) {
                r->invalidateWithBitmap(PROT_READ);
            }
            r->unlock();
        }
        shared.unlock();
    }
#endif
    RollingRegion *root = current()->find<RollingRegion>(addr);
    if(root == NULL) return false;
    root->lockRead();
    RollingBlock *region = root->find(addr);
    assertion(region != NULL);
    region->lockWrite();
    if (region->present() == true) {
        region->unlock();
        root->unlock();
        return true;
    }
    int ret = Memory::protect(region->start(), region->size(), PROT_READ | PROT_WRITE);
    assertion(ret == 0);
#ifdef USE_VM
    if(current()->dirtyBitmap().check(ptr(Context::current(), addr))) {
#endif
        gmacError_t ret2 = region->copyToHost();
        assertion(ret2 == gmacSuccess);
#ifdef USE_VM
    }
#endif
    region->readOnly();
    region->unlock();
    root->unlock();
    return true;
}

bool RollingManager::touch(Region * r)
{
    assertion(r != NULL);
    RollingRegion *root = dynamic_cast<RollingRegion *>(r);
    const std::vector<RollingBlock *> & regions = root->subRegions();
    assertion(regions.size() > 0);
    std::vector<RollingBlock *>::const_iterator it;

    for (it = regions.begin(); it != regions.end(); it++) {
        RollingBlock * block  = *it;
        block->lockWrite();
        if(block->dirty() == false) {
            block->readWrite();
            if(block->present() == false) {
#ifdef USE_VM
                /// \todo: Fix this! Should set ownership to the device
                if(current()->dirtyBitmap().check(ptr(Context::current(), block->start()))) {
#endif
                    gmacError_t ret = block->copyToHost();
                    assertion(ret == gmacSuccess);
#ifdef USE_VM
                }
#endif
            }
        }
        block->unlock();

        rollingMap.currentBuffer()->push(dynamic_cast<RollingBlock *>(block));
    }

    while(rollingMap.currentBuffer()->overflows()) writeBack();
    return true;
}

#ifdef USE_VM
#include "memory/Bitmap.h"
#endif

bool RollingManager::write(void *addr)
{
#ifdef USE_VM
    Context * ctx = Context::current();
    gmac::memory::vm::Bitmap & bitmap = ctx->mm().dirtyBitmap(); 
    if (!bitmap.synced()) {
        bitmap.sync();

        Map::iterator i;
        Map * m = current();
        m->lockRead();
        for(i = m->begin(); i != m->end(); i++) {
            RollingRegion *r = dynamic_cast<RollingRegion *>(i->second);
            r->lockWrite();
            r->invalidateWithBitmap(PROT_READ | PROT_WRITE);
            r->unlock();
        }
        m->unlock();

        RegionMap::iterator s;
        RegionMap &shared = Map::shared();
        shared.lockRead();
        for (s = shared.begin(); s != shared.end(); s++) {
            RollingRegion * r = dynamic_cast<RollingRegion *>(s->second);
            r->lockWrite();
            if(r->owner() == ctx) {
                r->invalidateWithBitmap(PROT_READ | PROT_WRITE);
            }
            r->unlock();
        }
        shared.unlock();

    }
#endif
    RollingRegion *root = current()->find<RollingRegion>(addr);
    if (root == NULL) return false;
    root->lockRead();
    RollingBlock *region = root->find(addr);
    assertion(region != NULL);

    while(rollingMap.currentBuffer()->overflows()) writeBack();
    // Other thread fixed the fault?
    region->lockWrite();
    if(region->dirty() == true) {
#ifdef USE_VM
        unsigned chunk = region->chunk(addr);
        if (region->isSeq && chunk == region->lastChunk + 1) {
            region->lastChunk = chunk;
            region->seqChunks++;
        } else {
            region->isSeq = false;
        }
        if (region->isSeq && region->seqChunks >= 3) {
            region->readWrite();
        } else {
            region->readWriteChunk(chunk);
        }
#endif
        region->unlock();
        root->unlock();
        return true;
    }

#ifdef USE_VM
    region->readWriteChunk(region->chunk(addr));
#else
    region->readWrite();
#endif
    if(region->present() == false) {
#ifdef USE_VM
        if(current()->dirtyBitmap().check(ptr(Context::current(), addr))) {
#endif
            gmacError_t ret = region->copyToHost();
            assertion(ret == gmacSuccess);
#ifdef USE_VM
        }
#endif
    }
#ifdef USE_VM
    region->lastChunk = region->chunk(addr);
    region->isSeq = true;
    region->seqChunks = 1;
#endif

    region->unlock();
    root->unlock();
    rollingMap.currentBuffer()->push(dynamic_cast<RollingBlock *>(region));
    return true;
}

void
RollingManager::map(Context *ctx, Region *r, void *devPtr)
{
    RollingRegion *region = dynamic_cast<RollingRegion *>(r);
    assertion(region != NULL);
    insertVirtual(ctx, r->start(), devPtr, r->size());
    region->relate(ctx);
    region->transferNonDirty();
}

}}}

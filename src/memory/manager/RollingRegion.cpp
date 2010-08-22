#include "RollingRegion.h"
#include "RollingManager.h"

#include "os/Memory.h"

#include <algorithm>

namespace gmac { namespace memory { namespace manager {

BlockList::BlockList()
    : RWLock(LockBlockList)
{}

RollingRegion::RollingRegion(RollingManager &manager, void *addr, size_t size, bool shared,
                      size_t cacheLine) :
   Region(addr, size, shared),
   manager(manager),
   _subRegions(),
   cacheLine(cacheLine),
   offset((unsigned long)addr & (cacheLine -1))
{
   trace("RollingRegion Starts");
   for(size_t s = 0; s < size; s += cacheLine) {
      void *p = (void *)((uint8_t *)addr + s);
      size_t regionSize = ((size -s) > cacheLine) ? cacheLine : (size - s);
      RollingBlock *region = new RollingBlock(*this, p, regionSize, shared);
      void *key = (void *)((uint8_t *)p + cacheLine);
      _map.insert(Map::value_type(key, region));
      _subRegions.push_back(region);
      _memory.lockWrite();
      _memory.insert(region);
      _memory.unlock();
   }
   trace("RollingRegion Ends");
}

RollingRegion::~RollingRegion()
{
   Map::const_iterator i;
   for(i = _map.begin(); i != _map.end(); i++) {
      delete i->second;
   }
   _map.clear();
}

void RollingRegion::syncToHost()
{
    Map::const_iterator i;
    for(i = _map.begin(); i != _map.end(); i++) {
        RollingBlock * block = i->second;
        block->lockWrite();
        if(block->present() != true)  {
            block->readWrite();
            block->copyToHost();
            block->readOnly();
        }
        block->unlock();
    }
}

void RollingRegion::relate(Context *ctx)
{
    RollingBuffer *buffer = manager.rollingMap.contextBuffer(ctx);
    Map::const_iterator i;
    // Push dirty regions in the rolling buffer
    // and copy to device clean regions
    for(i = _map.begin(); i != _map.end(); i++) {
        RollingBlock * block = i->second;
        block->lockWrite();
        if(block->dirty()) {
            buffer->push(block);
        }
        else {
            gmacError_t ret = ctx->copyToDevice(Manager::ptr(ctx, start()), start(), size());
            assertion(ret == gmacSuccess);
        }
        block->relate(ctx);
        block->unlock();
    }
    _relatives.push_back(ctx);
    buffer->inc(manager.lruDelta);
}

void RollingRegion::unrelate(Context *ctx)
{
   Map::iterator i;
   for(i = _map.begin(); i != _map.end(); i++) i->second->unrelate(ctx);
   _relatives.remove(ctx);
}

void RollingRegion::transfer()
{
   Map::iterator i;
   for(i = _map.begin(); i != _map.end(); i++) i->second->transfer();
   Region::transfer();
}

RollingBlock *RollingRegion::find(const void *addr)
{
   Map::const_iterator i = _map.upper_bound(addr);
   if(i == _map.end()) return NULL;
   if((addr_t)addr < (addr_t)i->second->start()) return NULL;
   return i->second;
}

void RollingRegion::flush()
{
    assertion(tryWrite() == false);
    trace("RollingRegion Invalidate %p (%zd bytes)", (void *) _addr, _size);

    // Flush those sub-regions that are present in _memory and dirty
    Map::const_iterator i;
    for(i = _map.begin(); i != _map.end(); i++) {
        RollingBlock * block = i->second;
        block->lockWrite();
        if(block->dirty() == true)  {
            trace("Flush RollingBlock %p (%zd bytes)", (void *) block->start(), block->size());
            manager.flush(block);
        }
        block->flush(); // Reset internal counters
        block->unlock();
    }
}


void RollingRegion::invalidate()
{
    //printf("HOLA NOBITMAP\n");
    assertion(tryWrite() == false);
    trace("RollingRegion Invalidate %p (%zd bytes)", (void *) _addr, _size);
    // Check if the region is already invalid

    _memory.lockWrite();
    if(_memory.empty()) { _memory.unlock(); return; }

    Context * ctx = Context::current();

#ifndef USE_VM
    BlockList::iterator i;
    for(i = _memory.begin(); i != _memory.end(); i++) {
        RollingBlock * block = *i;
        block->lockWrite();
        trace("Protected RollingBlock %p:%p)", block->start(), (uint8_t *) block->start() + block->size() - 1);
    }
#endif

    // Protect the region
    Memory::protect(__void(_addr), _size, PROT_NONE);

#ifndef USE_VM
    // Invalidate those sub-regions that are present in _memory
    for(i = _memory.begin(); i != _memory.end(); i++) {
        RollingBlock * block = *i;
        trace("Invalidate RollingBlock %p (%zd bytes)", (void *) block->start(), block->size());
        block->preInvalidate();
        block->unlock();
    }
    _memory.clear();
#endif
    _memory.unlock();
}

void RollingRegion::invalidateWithBitmap(int prot)
{
    //printf("HOLA BITMAP\n");
#ifdef USE_VM
    assertion(tryWrite() == false);
    trace("RollingRegion Invalidate with Bitmap %p (%zd bytes)", (void *) _addr, _size);
    // Check if the region is already invalid

    _memory.lockWrite();
    if(_memory.empty()) { _memory.unlock(); return; }

    Context * ctx = Context::current();

    BlockList::iterator i;
    for(i = _memory.begin(); i != _memory.end();) {
        RollingBlock * block = *i;
        block->lockWrite();
        trace("Protected RollingBlock %p:%p)", block->start(), (uint8_t *) block->start() + block->size() - 1);

        bool remove = false;
        for (unsigned c = 0; c < paramBitmapChunksPerPage; c++) {
            //printf("Chunk: %u\n", c);
            // Protect the region
            if (ctx->mm().dirtyBitmap().check(manager.ptr(ctx, block->startChunk(c)))) {
                remove = true;
                Memory::protect(block->startChunk(c), block->sizeChunk(), PROT_NONE);
            } else {
                Memory::protect(block->startChunk(c), block->sizeChunk(), prot);
            }
        }
        if (remove) {
            BlockList::iterator j = i;
            i++;
            block->preInvalidate();
            _memory.erase(j);
        } else {
            i++;
        }
        block->unlock();
    }
    _memory.unlock();
#endif
}


void RollingRegion::invalidate(const void *addr, size_t size)
{
    void *end = (void *)((addr_t)addr + size);
    Map::iterator i = _map.lower_bound(addr);
    assertion(i != _map.end());
    for(; i != _map.end() && i->second->start() < end; i++) {
        RollingBlock * block = i->second;
        block->lockWrite();
        // If the region is not present, just ignore it
        if(block->present() == false) {
            block->unlock();
            continue;
        }

        if(block->dirty()) { // We might need to update the device
            // Check if there is _memory that will not be invalidated
            if(block->start() <= addr ||
               block->end()   >= __void(__addr(addr) + size))
                manager.flush(block);
        }
        _memory.lockWrite();
        _memory.erase(block);
        _memory.unlock();
        block->invalidate();
        block->unlock();
    }
}

void RollingRegion::flush(const void *addr, size_t size)
{
   assert(tryWrite() == false);
   trace("RollingRegion Invalidate %p (%zd bytes)", (void *) addr, size);

   Map::iterator i = _map.lower_bound(addr);
   assertion(i != _map.end());
   for(; i != _map.end() && i->second->start() < addr; i++) {
       RollingBlock * block = i->second;
       block->lockWrite();
      // If the region is not present, just ignore it
      if(block->dirty() == true) {
          trace("Flush RollingBlock %p (%zd bytes)", (void *) block->start(), block->size());
          manager.flush(block);
      }
      block->unlock();
   }
}

void
RollingRegion::transferNonDirty()
{
    BlockList::iterator i;
    _memory.lockRead();
    for(i = _memory.begin(); i != _memory.end(); i++) {
        RollingBlock * block = *i;
        block->lockWrite();
        if (!block->dirty()) {
            manager.forceFlush(block);
        }
        block->unlock();
    }
    _memory.unlock();
}

void
RollingRegion::transferDirty()
{
    BlockList::iterator i;
    _memory.lockRead();
    for(i = _memory.begin(); i != _memory.end(); i++) {
        RollingBlock * block = *i;
        block->lockWrite();
        if (block->dirty()) {
            manager.flush(block);
        }
        block->unlock();
    }
    _memory.unlock();
}

const std::vector<RollingBlock *> &
RollingRegion::subRegions()
{
    return _subRegions;
}

RollingBlock::RollingBlock(RollingRegion &parent, void *addr, size_t size, bool shared) :
    ProtRegion(addr, size, shared),
    _parent(parent)
#ifdef USE_VM
    , transfers(0)
#endif
{}


RollingBlock::~RollingBlock()
{
   trace("RollingBlock %p released", (void *) _addr);
}

#ifdef USE_VM

enum ModelDirection {
    MODEL_TOHOST = 0,
    MODEL_TODEVICE = 1
};

template <ModelDirection M>
static inline
float costTransferCache(const RollingBlock & block, size_t blocks)
{
    if (M == MODEL_TOHOST) {
        if (blocks * block.sizeChunk() <= paramModelL1/2) {
            return paramModelToHostTransferL1;
        } else if (blocks * block.sizeChunk() <= paramModelL2/2) {
            return paramModelToHostTransferL2;
        } else {
            return paramModelToHostTransferMem;
        }
    } else { 
        if (blocks * block.sizeChunk() <= paramModelL1/2) {
            return paramModelToDeviceTransferL1;
        } else if (blocks * block.sizeChunk() <= paramModelL2/2) {
            return paramModelToDeviceTransferL2;
        } else {
            return paramModelToDeviceTransferMem;
        }
    }
}

template <ModelDirection M>
static inline
float costGaps(const RollingBlock & block, size_t gaps, size_t totalSize)
{
    return costTransferCache<M>(block, totalSize) * gaps * block.sizeChunk();
}

template <ModelDirection M>
static inline
float costTransfer(const RollingBlock & block, size_t blocks)
{
    return costTransferCache<M>(block, blocks) * blocks * block.sizeChunk();
}

template <ModelDirection M>
static inline
float costConfig()
{
    if (M == MODEL_TOHOST) {
        return paramModelToHostConfig;
    } else {
        return paramModelToDeviceConfig;
    }
}

template <ModelDirection M>
static inline
float cost(const RollingBlock & block, size_t blocks)
{
    return costConfig<M>() + costTransfer<M>(block, blocks);
}

gmacError_t
RollingBlock::toDevice(Context * ctx, void * addr, size_t size)
{
    gmacError_t ret = gmacSuccess;
    trace("Sending %zd bytes to device", size);
    if ((ret = ctx->copyToDevice(Manager::ptr(ctx, addr), addr, size)) != gmacSuccess)
        return ret;
    std::list<Context *>::iterator i;
    trace("I have %zd relatives", _relatives.size());
    for (i = _relatives.begin(); i != _relatives.end(); i++) {
        Context * ctx = *i;
        if ((ret = ctx->copyToDevice(Manager::ptr(ctx, addr), addr, size)) != gmacSuccess) {
            break;
        }
    }
    return ret;

}

gmacError_t
RollingBlock::toHost(Context * ctx, void * addr, size_t size)
{
    trace("Sending %zd bytes to host", size);
    return ctx->copyToHost(addr, Manager::ptr(ctx, addr), size);
}


gmacError_t
RollingBlock::copyToDevice()
{
    gmacError_t ret = gmacSuccess;
    Context * ctx = Context::current();
    bool in_subgroup = false;
    size_t i = 0, g_start = 0, g_end = 0;
    int gaps = 0;
    gmac::memory::vm::Bitmap & bitmap = ctx->mm().dirtyBitmap();
    transfers++;
    while (i < chunks()) {
        if (in_subgroup) {
            if (bitmap.checkAndClear(_parent.manager.ptr(ctx, startChunk(i)))) {
                //printf("CHECKED\n");
                g_end = i;
            } else {
                if (costGaps<MODEL_TODEVICE>(*this, gaps + 1, i - g_start + 1) < cost<MODEL_TODEVICE>(*this, 1)) {
                    gaps++;
                } else {
                    in_subgroup = false;
                    //printf("START %zd STOP %zd\n", g_start, g_end);
                    if ((ret = toDevice(ctx, startChunk(g_start), (g_end - g_start + 1) * sizeChunk())) != gmacSuccess) {
                        break;
                    }
                }
            }
        } else {
            if (bitmap.checkAndClear(_parent.manager.ptr(ctx, ((uint8_t *) start()) + i * sizeChunk()))) {
                g_start = i;
                gaps = 0;
                in_subgroup = true;
            }
        }
        i++;
    }
    if (in_subgroup) {
        //printf("AT END START %zd STOP %zd\n", g_start, g_end);
        ret = toDevice(ctx, startChunk(g_start), (g_end - g_start + 1) * sizeChunk());
    }
    return ret;
}

gmacError_t
RollingBlock::copyToHost()
{
    gmacError_t ret = gmacSuccess;
    Context * ctx = Context::current();
    bool in_subgroup = false;
    size_t i = 0, g_start = 0, g_end = 0;
    int gaps = 0;
    uint8_t * b_start = (uint8_t *) start();
    gmac::memory::vm::Bitmap & bitmap = ctx->mm().dirtyBitmap();
    while (i < chunks()) {
        if (in_subgroup) {
            if (bitmap.checkAndClear(_parent.manager.ptr(ctx, b_start + i * sizeChunk()))) {
                //printf("CHECKED\n");
                g_end = i;
            } else {
                if (costGaps<MODEL_TOHOST>(*this, gaps + 1, i - g_start + 1) < cost<MODEL_TOHOST>(*this, 1)) {
                    gaps++;
                } else {
                    in_subgroup = false;
                    //printf("START %zd STOP %zd\n", g_start, g_end);
                    if ((ret = toHost(ctx, startChunk(g_start), (g_end - g_start + 1) * sizeChunk())) != gmacSuccess) {
                        break;
                    }
                }
            }
        } else {
            if (bitmap.checkAndClear(_parent.manager.ptr(ctx, ((uint8_t *) start()) + i * sizeChunk()))) {
                g_start = i;
                gaps = 0;
                in_subgroup = true;
            }
        }
        i++;
    }
    if (in_subgroup) {
        //printf("AT END START %zd STOP %zd\n", g_start, g_end);
        ret = toHost(ctx, startChunk(g_start), (g_end - g_start + 1) * sizeChunk());
    }
    return ret;
}

void
RollingBlock::readWriteChunk(unsigned chunk)
{
    if (present() == false) _parent.push(this);

    assertion(tryWrite() == false);
    _present = _dirty = true;
    int ret = Memory::protect(startChunk(chunk), sizeChunk(), PROT_READ | PROT_WRITE);
    gmac::memory::vm::Bitmap & bitmap = Context::current()->mm().dirtyBitmap();
    bitmap.set(_parent.manager.ptr(Context::current(), startChunk(chunk)));
    assertion(ret == 0);
}

#endif

inline void
RollingBlock::readWrite()
{
    if (present() == false) _parent.push(this);

    assertion(tryWrite() == false);
    _present = _dirty = true;
    int ret = Memory::protect(__void(_addr), _size, PROT_READ | PROT_WRITE);
    assertion(ret == 0);

#ifdef USE_VM
    gmac::memory::vm::Bitmap & bitmap = Context::current()->mm().dirtyBitmap();
    for (unsigned c = 0; c < chunks(); c++) {
        bitmap.set(_parent.manager.ptr(Context::current(), startChunk(c)));
    }
#endif
}


}}}

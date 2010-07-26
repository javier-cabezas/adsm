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
      block->unlock();
   }
}


void RollingRegion::invalidate()
{
    assertion(tryWrite() == false);
    trace("RollingRegion Invalidate %p (%zd bytes)", (void *) _addr, _size);
    // Check if the region is already invalid

    _memory.lockWrite();
    if(_memory.empty()) { _memory.unlock(); return; }

    Context * ctx = Context::current();

    BlockList::iterator i;
    for(i = _memory.begin(); i != _memory.end(); i++) {
        RollingBlock * block = *i;
        block->lockWrite();
        trace("Protected RollingBlock %p:%p)", block->start(), (uint8_t *) block->start() + block->size() - 1);
    }

    // Protect the region
    Memory::protect(__void(_addr), _size, PROT_NONE);

    // Invalidate those sub-regions that are present in _memory
    for(i = _memory.begin(); i != _memory.end(); i++) {
        RollingBlock * block = *i;
#ifndef USE_VM
        trace("Invalidate RollingBlock %p (%zd bytes)", (void *) block->start(), block->size());
        block->preInvalidate();
#endif
        block->unlock();
    }
#ifndef USE_VM
    _memory.clear();
#endif
    _memory.unlock();
}

void RollingRegion::invalidateWithBitmap(int prot)
{
#ifdef USE_VM
    assertion(tryWrite() == false);
    trace("RollingRegion Invalidate with Bitmap %p (%zd bytes)", (void *) _addr, _size);
    // Check if the region is already invalid

    _memory.lockWrite();
    if(_memory.empty()) { _memory.unlock(); return; }

    Context * ctx = Context::current();

    BlockList::iterator i;
    for(i = _memory.begin(); i != _memory.end(); i++) {
        RollingBlock * block = *i;
        block->lockWrite();
        trace("Protected RollingBlock %p:%p)", block->start(), (uint8_t *) block->start() + block->size() - 1);

        // Protect the region
        if (ctx->mm().dirtyBitmap().check(manager.ptr(ctx, block->start()))) {
            Memory::protect(block->start(), block->size(), PROT_NONE);
        } else {
            Memory::protect(block->start(), block->size(), prot);
        }
    }

    // Invalidate those sub-regions that are present in _memory
    for(i = _memory.begin(); i != _memory.end();) {
        RollingBlock * block = *i;
        if (ctx->mm().dirtyBitmap().check(manager.ptr(ctx, block->start()))) {
            BlockList::iterator j = i;
            i++;
            trace("Invalidate RollingBlock %p (%zd bytes)", (void *) block->start(), block->size());
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
{ }


RollingBlock::~RollingBlock()
{
   trace("RollingBlock %p released", (void *) _addr);
}

void
RollingBlock::readOnly()
{
    if(present() == false) _parent.push(this);
    ProtRegion::readOnly();
}

void
RollingBlock::readWrite()
{
    if(present() == false) _parent.push(this);
    ProtRegion::readWrite();
}

}}}

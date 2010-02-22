#include "RollingRegion.h"
#include "RollingManager.h"

#include "os/Memory.h"

#include <algorithm>

namespace gmac { namespace memory { namespace manager {

BlockList::BlockList()
    : RWLock(paraver::blockList)
{}

RollingRegion::RollingRegion(RollingManager &manager, void *addr, size_t size, bool shared,
                      size_t cacheLine) :
   Region(addr, size, shared),
   manager(manager),
   cacheLine(cacheLine),
   offset((unsigned long)addr & (cacheLine -1))
{
   TRACE("RollingRegion Starts");
   for(size_t s = 0; s < size; s += cacheLine) {
      void *p = (void *)((uint8_t *)addr + s);
      size_t regionSize = ((size -s) > cacheLine) ? cacheLine : (size - s);
      RollingBlock *region = new RollingBlock(*this, p, regionSize, shared);
      void *key = (void *)((uint8_t *)p + cacheLine);
      map.insert(Map::value_type(key, region));
      memory.lockWrite();
      memory.insert(region);
      memory.unlock();
   }
   TRACE("RollingRegion Ends");
}

RollingRegion::~RollingRegion()
{
   Map::const_iterator i;
   for(i = map.begin(); i != map.end(); i++) {
      delete i->second;
   }
   map.clear();
}

void RollingRegion::relate(Context *ctx)
{
    RollingBuffer *buffer = manager.rollingMap.contextBuffer(ctx);
    Map::const_iterator i;
    // Push dirty regions in the rolling buffer
    // and copy to device clean regions
    for(i = map.begin(); i != map.end(); i++) {
        RollingBlock * block = i->second;
        block->lockWrite();
        if(block->dirty()) {
            buffer->push(block);
        }
        else {
            gmacError_t ret = ctx->copyToDevice(Manager::ptr(start()), start(), size());
            ASSERT(ret == gmacSuccess);
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
   for(i = map.begin(); i != map.end(); i++) i->second->unrelate(ctx);
   _relatives.remove(ctx);
}

void RollingRegion::transfer()
{
   Map::iterator i;
   for(i = map.begin(); i != map.end(); i++) i->second->transfer();
   Region::transfer();
}

RollingBlock *RollingRegion::find(const void *addr)
{
   Map::const_iterator i = map.upper_bound(addr);
   if(i == map.end()) return NULL;
   if((addr_t)addr < (addr_t)i->second->start()) return NULL;
   return i->second;
}

void RollingRegion::flush()
{
   assert(tryWrite() == false);
   TRACE("RollingRegion Invalidate %p (%zd bytes)", (void *) _addr, _size);

   // Flush those sub-regions that are present in memory and dirty
   Map::const_iterator i;
   for(i = map.begin(); i != map.end(); i++) {
      RollingBlock * block = i->second;
      block->lockWrite();
      if(block->dirty() == true)  {
          TRACE("Flush RollingBlock %p (%zd bytes)", (void *) block->start(), block->size());
          manager.flush(block);
      }
      block->unlock();
   }
}


void RollingRegion::invalidate()
{
    ASSERT(tryWrite() == false);
    TRACE("RollingRegion Invalidate %p (%zd bytes)", (void *) _addr, _size);
    // Check if the region is already invalid

    memory.lockWrite();
    if(memory.empty()) { memory.unlock(); return; }

    BlockList::iterator i;
    for(i = memory.begin(); i != memory.end(); i++) {
        RollingBlock * block = *i;
        block->lockWrite();
        TRACE("Protected RollingBlock %p:%p)", block->start(), (uint8_t *) block->start() + block->size() - 1);
    }

    // Protect the region
    Memory::protect(__void(_addr), _size, PROT_NONE);
    // Invalidate those sub-regions that are present in memory
    for(i = memory.begin(); i != memory.end(); i++) {
        RollingBlock * block = *i;
        TRACE("Invalidate RollingBlock %p (%zd bytes)", (void *) block->start(), block->size());
        block->preInvalidate();
        block->unlock();
    }
    memory.clear();
    memory.unlock();
}

void RollingRegion::invalidate(const void *addr, size_t size)
{
    void *end = (void *)((addr_t)addr + size);
    Map::iterator i = map.lower_bound(addr);
    ASSERT(i != map.end());
    for(; i != map.end() && i->second->start() < end; i++) {
        RollingBlock * block = i->second;
        block->lockWrite();
        // If the region is not present, just ignore it
        if(block->present() == false) {
            block->unlock();
            continue;
        }

        if(block->dirty()) { // We might need to update the device
            // Check if there is memory that will not be invalidated
            if(block->start() <= addr ||
               block->end()   >= __void(__addr(addr) + size))
                manager.flush(block);
        }
        memory.lockWrite();
        memory.erase(block);
        memory.unlock();
        block->invalidate();
        block->unlock();
    }
}

void RollingRegion::flush(const void *addr, size_t size)
{
   Map::iterator i = map.lower_bound(addr);
   ASSERT(i != map.end());
   for(; i != map.end() && i->second->start() < addr; i++) {
       RollingBlock * block = i->second;
       block->lockWrite();
      // If the region is not present, just ignore it
      if(block->dirty() == true) {
          manager.flush(block);
      }
      block->unlock();
   }
}

void
RollingRegion::transferNonDirty()
{
    BlockList::iterator i;
    memory.lockRead();
    for(i = memory.begin(); i != memory.end(); i++) {
        RollingBlock * block = *i;
        block->lockWrite();
        if (!block->dirty()) {
            manager.forceFlush(block);
        }
        block->unlock();
    }
    memory.unlock();
}

void
RollingRegion::transferDirty()
{
    BlockList::iterator i;
    memory.lockRead();
    for(i = memory.begin(); i != memory.end(); i++) {
        RollingBlock * block = *i;
        block->lockWrite();
        if (block->dirty()) {
            manager.flush(block);
        }
        block->unlock();
    }
    memory.unlock();
}


RollingBlock::RollingBlock(RollingRegion &parent, void *addr, size_t size, bool shared) :
   ProtRegion(addr, size, shared),
   _parent(parent)
{ }


RollingBlock::~RollingBlock()
{
   TRACE("RollingBlock %p released", (void *) _addr);
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

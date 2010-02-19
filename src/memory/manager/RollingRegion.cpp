#include "RollingRegion.h"
#include "RollingManager.h"

#include "os/Memory.h"

#include <algorithm>

namespace gmac { namespace memory { namespace manager {

RollingRegion::RollingRegion(RollingManager &manager, void *addr, size_t size,
                      size_t cacheLine) :
   Region(addr, size),
   manager(manager),
   cacheLine(cacheLine),
   offset((unsigned long)addr & (cacheLine -1))
{
   TRACE("RollingRegion Starts");
   for(size_t s = 0; s < size; s += cacheLine) {
      void *p = (void *)((uint8_t *)addr + s);
      size_t regionSize = ((size -s) > cacheLine) ? cacheLine : (size - s);
      RollingBlock *region = new RollingBlock(*this, p, regionSize);
      void *key = (void *)((uint8_t *)p + cacheLine);
      map.insert(Map::value_type(key, region));
      memory.insert(region);
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
     i->second->lockWrite();
     if(i->second->dirty()) {
       buffer->push(i->second);
     }
     else {
       gmacError_t ret = ctx->copyToDevice(Manager::ptr(start()), start(), size());
       ASSERT(ret == gmacSuccess);
     }
     i->second->relate(ctx);
     i->second->unlock();
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
   TRACE("RollingRegion Invalidate %p (%d bytes)", _addr, _size);
   // Check if the region is already invalid
   if(memory.empty()) return;

   List::iterator i;

   // Flush those sub-regions that are present in memory and dirty
   for(i = memory.begin(); i != memory.end(); i++) {
      (*i)->lockRead();
      bool dirty = (*i)->dirty();
      (*i)->unlock();
      if(dirty == false) continue;
      TRACE("Flush SubRegion %p (%d bytes)", (*i)->start(),
           (*i)->size());
      manager.flush(*i);
   }
   memory.clear();
}


void RollingRegion::invalidate()
{
   ASSERT(tryWrite() == false);
   TRACE("RollingRegion Invalidate %p (%d bytes)", _addr, _size);
   // Check if the region is already invalid
   if(memory.empty()) return;

   List::iterator i;
   for(i = memory.begin(); i != memory.end(); i++) {
      (*i)->lockWrite();
   }
   // Protect the region
   Memory::protect(__void(_addr), _size, PROT_NONE);
   // Invalidate those sub-regions that are present in memory
   for(i = memory.begin(); i != memory.end(); i++) {
      TRACE("Invalidate SubRegion %p (%d bytes)", (*i)->start(),
           (*i)->size());
      (*i)->silentInvalidate();
      (*i)->unlock();
   }
   memory.clear();
}

void RollingRegion::invalidate(const void *addr, size_t size)
{
   void *end = (void *)((addr_t)addr + size);
   Map::iterator i = map.lower_bound(addr);
   ASSERT(i != map.end());
   for(; i != map.end() && i->second->start() < end; i++) {
      i->second->lockWrite();
      // If the region is not present, just ignore it
      if(i->second->present() == false) {
         i->second->unlock();
         continue;
      }

      if(i->second->dirty()) { // We might need to update the device
         // Check if there is memory that will not be invalidated
         if(i->second->start() < addr ||
            i->second->end() > __void(__addr(addr) + size))
            manager.flush(i->second);
      }
      memory.erase(i->second);
      i->second->invalidate();
      i->second->unlock();
   }
}

void RollingRegion::flush(const void *addr, size_t size)
{
   Map::iterator i = map.lower_bound(addr);
   ASSERT(i != map.end());
   for(; i != map.end() && i->second->start() < addr; i++) {
      // If the region is not present, just ignore it
      if(i->second->dirty() == false) {
         i->second->unlock();
         continue;
      }
      manager.flush(i->second);
      i->second->unlock();
   }
}

void
RollingRegion::transferNonDirty()
{
   List::iterator i;
	for(i = memory.begin(); i != memory.end(); i++) {
		if (!(*i)->dirty()) {
         manager.flush(*i);
      }
	}
}

void
RollingRegion::transferDirty()
{
   List::iterator i;
	for(i = memory.begin(); i != memory.end(); i++) {
		(*i)->lockWrite();
		if ((*i)->dirty()) {
         manager.flush(*i);
         (*i)->invalidate();
      }
		(*i)->unlock();
	}
}


RollingBlock::RollingBlock(RollingRegion &parent, void *addr, size_t size) :
   ProtRegion(addr, size),
   _parent(parent)
{ }


RollingBlock::~RollingBlock()
{
   TRACE("SubRegion %p released", _addr);
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

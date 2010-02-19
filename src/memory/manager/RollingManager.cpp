#include "RollingManager.h"
#include "os/Memory.h"

#include <kernel/Context.h>

#include <unistd.h>

#include <typeinfo>

namespace gmac { namespace memory { namespace manager {

RollingBuffer::RollingBuffer() :
   RWLock(paraver::rollingBuffer),
   _max(0)
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
   flush(r);
}


RollingManager::RollingManager() :
   Handler(),
   lineSize(0),
   lruDelta(0),
   writeMutex(paraver::writeMutex),
   writeBuffer(NULL),
   writeBufferSize(0)
{
   lineSize = paramLineSize;
   lruDelta = paramLruDelta;
   TRACE("Using %d as Line Size", lineSize);
   TRACE("Using %d as LRU Delta Size", lruDelta);
}

RollingManager::~RollingManager()
{
}

void *RollingManager::alloc(void *addr, size_t count, int attr)
{
   void *cpuAddr;

   Context * ctx = Context::current();
   if (attr == GMAC_MALLOC_PINNED) {
      void *hAddr;
      if (ctx->halloc(&hAddr, count) != gmacSuccess) return NULL;
      cpuAddr = hostRemap(addr, hAddr, count);
   } else {
      cpuAddr = hostMap(addr, count, PROT_NONE);
   }
   insertVirtual(cpuAddr, addr, count);
   rollingMap.currentBuffer()->inc(lruDelta);
   insert(new RollingRegion(*this, cpuAddr, count, pageTable().getPageSize()));
	TRACE("Alloc %p (%d bytes)", cpuAddr, count);
   return cpuAddr;
}


void RollingManager::release(void *addr)
{
   Region *reg = remove(addr);
   removeVirtual(reg->start(), reg->size());
   Context * ctx = Context::current();
   if(reg->owner() == ctx) {
      hostUnmap(addr, reg->size()); // Global mappings do not have a shadow copy in system memory
      TRACE("Deleting Region %p\n", addr);
      delete reg;
   }
#ifdef USE_GLOBAL_HOST
	// When using host-mapped memory, global regions do not
	// increase the rolling size
	if(proc->isShared(addr) == false)
      rollingMap.currentBuffer()->dec(lruDelta);
#else
   rollingMap.currentBuffer()->dec(lruDelta);
#endif
   TRACE("Released %p", addr);
}


void RollingManager::flush()
{
   TRACE("RollingManager Flush Starts");
   Context * ctx = Context::current();
   Process::SharedMap::iterator s;
	Process::SharedMap &sharedMem = proc->sharedMem();
   for(s = sharedMem.begin(); s != sharedMem.end(); s++) {
		RollingRegion *r = current()->find<RollingRegion>(s->second.start());
      r->transferDirty();
	}

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

   /** \todo Fix vm */
	// ctx->flush();
	ctx->invalidate();
   TRACE("RollingManager Flush Ends");
}

void RollingManager::flush(const RegionSet & regions)
{
   // If no dependencies, a global flush is assumed
   if (regions.size() == 0) {
      flush();
      return;
   }

   TRACE("RollingManager Flush Starts");
   Context * ctx = Context::current();
   Process::SharedMap::iterator i;
	Process::SharedMap &sharedMem = proc->sharedMem();
   for(i = sharedMem.begin(); i != sharedMem.end(); i++) {
		RollingRegion * r = current()->find<RollingRegion>(i->second.start());
      r->lockWrite();
      r->transferDirty();
      r->unlock();
	}
   size_t blocks = rollingMap.currentBuffer()->size();

   for(int j = 0; j < blocks; j++) {
      RollingBlock *r = rollingMap.currentBuffer()->pop();
      r->lockWrite();
      // Check if we have to flush
      if(std::find(regions.begin(), regions.end(), &r->getParent()) == regions.end()) {
         rollingMap.currentBuffer()->push(r);
         r->unlock();
         continue;
      }
      r->unlock();
      flush(r);
      TRACE("Flush to Device %p", r->start());
   }

   /** \todo Fix vm */
	// ctx->flush();
	ctx->invalidate();
   TRACE("RollingManager Flush Ends");
}

void RollingManager::invalidate()
{
   TRACE("RollingManager Invalidation Starts");
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
   //gmac::Context::current()->flush();
   gmac::Context::current()->invalidate();
   TRACE("RollingManager Invalidation Ends");
}

void RollingManager::invalidate(const RegionSet & regions)
{
   // If no dependencies, a global invalidation is assumed
   if (regions.size() == 0) {
      invalidate();
      return;
   }

   TRACE("RollingManager Invalidation Starts");
   RegionSet::const_iterator i;
   for(i = regions.begin(); i != regions.end(); i++) {
      RollingRegion *r = dynamic_cast<RollingRegion *>(*i);
      r->lockWrite();
      r->invalidate();
      r->unlock();
   }
   //gmac::Context::current()->flush();
   gmac::Context::current()->invalidate();
   TRACE("RollingManager Invalidation Ends");
}

void RollingManager::invalidate(const void *addr, size_t size)
{
	RollingRegion *reg = current()->find<RollingRegion>(addr);
   ASSERT(reg != NULL);
   reg->lockWrite();
   ASSERT(reg->end() >= (void *)((addr_t)addr + size));
   reg->invalidate(addr, size);
   reg->unlock();
}

void RollingManager::flush(const void *addr, size_t size)
{
	RollingRegion *reg = current()->find<RollingRegion>(addr);
   ASSERT(reg != NULL);
   reg->lockWrite();
   ASSERT(reg->end() >= (void *)((addr_t)addr + size));
   reg->flush(addr, size);
   reg->unlock();
}

//
// Handler Interface
//

bool RollingManager::read(void *addr)
{
	RollingRegion *root = current()->find<RollingRegion>(addr);
   if(root == NULL) return false;
   Context * owner = root->owner();
   if (owner->status() == Context::RUNNING) owner->sync();
   ProtRegion *region = root->find(addr);
   ASSERT(region != NULL);
   region->lockWrite();
	if (region->present() == true) {
      region->unlock();
      return true;
   }
   region->readWrite();
   if(current()->pageTable().dirty(addr)) {
       gmacError_t ret = region->copyToHost();
      ASSERT(ret == gmacSuccess);
      current()->pageTable().clear(addr);
   }
   region->readOnly();
   region->unlock();
   return true;
}


bool RollingManager::write(void *addr)
{
	RollingRegion *root = current()->find<RollingRegion>(addr);
   if (root == NULL) return false;
   root->lockWrite();
   ProtRegion *region = root->find(addr);
   ASSERT(region != NULL);
   // Other thread fixed the fault?
   region->lockWrite();
	if(region->dirty() == true) {
     region->unlock();
     root->unlock();
     return true;
   }
   Context *owner = root->owner();
   if(owner->status() == Context::RUNNING) owner->sync();

   Context *ctx = Context::current();
   
   while(rollingMap.currentBuffer()->overflows()) writeBack();
   region->readWrite();
   if(region->present() == false && current()->pageTable().dirty(addr)) {
       gmacError_t ret = region->copyToHost();
     ASSERT(ret == gmacSuccess);
     current()->pageTable().clear(addr);
   }
   region->unlock();
   root->unlock();
   rollingMap.currentBuffer()->push(dynamic_cast<RollingBlock *>(region));
   return true;
}

void
RollingManager::remap(Context *ctx, void *cpuPtr, void *devPtr, size_t count)
{
	RollingRegion *region = current()->find<RollingRegion>(cpuPtr);
	ASSERT(region != NULL); ASSERT(region->size() == count);
	insertVirtual(ctx, cpuPtr, devPtr, count);
	region->relate(ctx);
   region->transferNonDirty();
}

}}}

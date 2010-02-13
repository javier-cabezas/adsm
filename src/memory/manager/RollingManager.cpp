#include "RollingManager.h"
#include "os/Memory.h"

#include "config/params.h"

#include <kernel/Context.h>

#include <unistd.h>

#include <typeinfo>

namespace gmac { namespace memory { namespace manager {

RollingBuffer::RollingBuffer() :
    lock(paraver::rollingBuffer),
    _max(0)
{}

void RollingManager::waitForWrite(void *addr, size_t size)
{
	writeMutex.lock();
	if(writeBuffer) {
		ProtSubRegion *r = regionRolling[Context::current()]->front();
		r->sync();
		munlock(writeBuffer, writeBufferSize);
	}
	writeBuffer = addr;
	writeBufferSize = size;
	writeMutex.unlock();
}


void RollingManager::writeBack()
{
	ProtSubRegion *r = regionRolling[Context::current()]->pop();
	waitForWrite(r->start(), r->size());
	mlock(writeBuffer, writeBufferSize);
	assert(r->copyToDevice() == gmacSuccess);
	r->readOnly();
}


void RollingManager::flushToDevice() 
{
    Context * ctx = Context::current();
	waitForWrite();
	while(regionRolling[ctx]->empty() == false) {
		ProtSubRegion *r = regionRolling[ctx]->pop();
		assert(r->copyToDevice() == gmacSuccess);
		r->readOnly();
		TRACE("Flush to Device %p", r->start()); 
	}
}

PARAM_REGISTER(paramLineSize,
               size_t,
               1024,
               "GMAC_LINESIZE",
               PARAM_NONZERO);

PARAM_REGISTER(paramLruDelta,
               size_t,
               2,
               "GMAC_LRUDELTA",
               PARAM_NONZERO);

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
    std::map<Context *, RollingBuffer *>::iterator r;
    for (r = regionRolling.begin(); r != regionRolling.end(); r++) {
        delete r->second;
    }
}

void *RollingManager::alloc(void *addr, size_t count)
{
	void *cpuAddr = NULL;
	if((cpuAddr = hostMap(addr, count, PROT_NONE)) == NULL)
		return NULL;
	
	insertVirtual(cpuAddr, addr, count);
    Context * ctx = Context::current();
    if (!regionRolling[ctx]) {
        regionRolling[ctx] = new RollingBuffer();
    }
	regionRolling[ctx]->inc(lruDelta);
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
		hostUnmap(addr, reg->size());	// Global mappings do not have a shadow copy in system memory
		TRACE("Deleting Region %p\n", addr);
		delete reg;
	}
#ifdef USE_GLOBAL_HOST
	// When using host-mapped memory, global regions do not
	// increase the rolling size
	if(proc->isShared(addr) == false)
		regionRolling[ctx]->dec(lruDelta);
#else
	regionRolling[ctx]->dec(lruDelta);
#endif
	TRACE("Released %p", addr);
}


void RollingManager::flush()
{
	TRACE("RollingManager Flush Starts");
	flushToDevice();
	memory::Map::iterator i;
	current()->lock();
	for(i = current()->begin(); i != current()->end(); i++) {
		Region *r = i->second;
		if(typeid(*r) != typeid(RollingRegion)) continue;
		dynamic_cast<RollingRegion *>(r)->invalidate();
	}

    Context * ctx = Context::current();
	current()->unlock();
	ctx->flush();
	ctx->invalidate();
	TRACE("RollingManager Flush Ends");
}

void RollingManager::invalidate(const void *addr, size_t size)
{
	RollingRegion *reg = current()->find<RollingRegion>(addr);
	assert(reg != NULL);
	assert(reg->end() >= (void *)((addr_t)addr + size));
	reg->invalidate(addr, size);
}

void RollingManager::flush(const void *addr, size_t size)
{
	RollingRegion *reg = current()->find<RollingRegion>(addr);
	assert(reg != NULL);
	assert(reg->end() >= (void *)((addr_t)addr + size));
	reg->flush(addr, size);
}

// Handler Interface

bool RollingManager::read(void *addr)
{
	RollingRegion *root= current()->find<RollingRegion>(addr);
	if(root == NULL) return false;
	ProtRegion *region = root->find(addr);
	assert(region != NULL);
	assert(region->present() == false);
	region->readWrite();
	if(current()->pageTable().dirty(addr)) {
		assert(region->copyToHost() == gmacSuccess);
		current()->pageTable().clear(addr);
	}
	region->readOnly();
	return true;
}


bool RollingManager::write(void *addr)
{
	RollingRegion *root = current()->find<RollingRegion>(addr);
	if(root == NULL) return false;
	ProtRegion *region = root->find(addr);
	assert(region != NULL);
	assert(region->dirty() == false);
    Context * ctx = Context::current();
    if (!regionRolling[ctx]) {
        regionRolling[ctx] = new RollingBuffer();
    }
	while(regionRolling[ctx]->overflows()) writeBack();
	region->readWrite();
	if(region->present() == false && current()->pageTable().dirty(addr)) {
		assert(region->copyToHost() == gmacSuccess);
		current()->pageTable().clear(addr);
	}
	regionRolling[ctx]->push(
			dynamic_cast<ProtSubRegion *>(region));
	return true;
}


}}}

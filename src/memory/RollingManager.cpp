#include "RollingManager.h"
#include "os/Memory.h"

#include "config/params.h"

#include <kernel/Context.h>

#include <unistd.h>
#include <malloc.h>

#include <typeinfo>

namespace gmac { namespace memory {

void RollingManager::waitForWrite(void *addr, size_t size)
{
	MUTEX_LOCK(writeMutex);
	if(writeBuffer) {
		ProtSubRegion *r = regionRolling[Context::current()].front();
		r->sync();
		munlock(writeBuffer, writeBufferSize);
	}
	writeBuffer = addr;
	writeBufferSize = size;
	MUTEX_UNLOCK(writeMutex);
}


void RollingManager::writeBack()
{
	ProtSubRegion *r = regionRolling[Context::current()].pop();
	waitForWrite(r->start(), r->size());
	mlock(writeBuffer, writeBufferSize);
	assert(r->copyToDevice() == gmacSuccess);
	r->readOnly();
}


void RollingManager::flushToDevice() 
{
	waitForWrite();
	while(regionRolling[Context::current()].empty() == false) {
		ProtSubRegion *r = regionRolling[Context::current()].pop();
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
	writeBuffer(NULL),
	writeBufferSize(0)
{
	MUTEX_INIT(writeMutex);
	lineSize = paramLineSize;
	lruDelta = paramLruDelta;
	TRACE("Using %d as Line Size", lineSize);
	TRACE("Using %d as LRU Delta Size", lruDelta);
}

void *RollingManager::alloc(void *addr, size_t size)
{
	void *cpuAddr = NULL;
	if((cpuAddr = hostMap(addr, size, PROT_NONE)) == NULL)
		return NULL;
	
	insertVirtual(cpuAddr, addr, size);
	regionRolling[Context::current()].inc(lruDelta);
	insert(new RollingRegion(*this, cpuAddr, size, pageTable().getPageSize()));
	return cpuAddr;
}


void RollingManager::release(void *addr)
{
	Region *reg = remove(addr);
	removeVirtual(reg->start(), reg->size());
	if(reg->owner() == Context::current()) {
		hostUnmap(addr, reg->size());	// Global mappings do not have a shadow copy in system memory
		TRACE("Deleting Region %p\n", addr);
		delete reg;
	}
	regionRolling[Context::current()].dec(lruDelta);
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
	current()->unlock();
	gmac::Context::current()->flush();
	gmac::Context::current()->invalidate();
	TRACE("RollingManager Flush Ends");
}

Context *RollingManager::owner(const void *addr)
{
	RollingRegion *reg= get(addr);
	if(reg == NULL) return NULL;
	return reg->owner();
}

void RollingManager::invalidate(const void *addr, size_t size)
{
	RollingRegion *reg = get(addr);
	assert(reg != NULL);
	assert(reg->end() >= (void *)((addr_t)addr + size));
	reg->invalidate(addr, size);
}

void RollingManager::flush(const void *addr, size_t size)
{
	RollingRegion *reg = get(addr);
	assert(reg != NULL);
	assert(reg->end() >= (void *)((addr_t)addr + size));
	reg->flush(addr, size);
}

// Handler Interface

bool RollingManager::read(void *addr)
{
	RollingRegion *root = get(addr);
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
	RollingRegion *root = get(addr);
	if(root == NULL) return false;
	ProtRegion *region = root->find(addr);
	assert(region != NULL);
	assert(region->dirty() == false);
	while(regionRolling[Context::current()].overflows()) writeBack();
	region->readWrite();
	if(region->present() == false && current()->pageTable().dirty(addr)) {
		assert(region->copyToHost() == gmacSuccess);
		current()->pageTable().clear(addr);
	}
	regionRolling[Context::current()].push(
			dynamic_cast<ProtSubRegion *>(region));
	return true;
}


} };

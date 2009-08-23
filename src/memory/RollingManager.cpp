#include "RollingManager.h"
#include "os/Util.h"

#include <kernel/Context.h>

#include <unistd.h>
#include <algorithm>

namespace gmac {

const char *RollingManager::lineSizeVar = "GMAC_LINESIZE";
const char *RollingManager::lruDeltaVar = "GMAC_LRUDELTA";

void RollingManager::waitForWrite(void *addr, size_t size)
{
	MUTEX_LOCK(writeMutex);
	if(writeBuffer) {
		ProtSubRegion *r = regionRolling[Context::current()].front();
		r->context()->sync();
		munlock(writeBuffer, writeBufferSize);
	}
	writeBuffer = addr;
	writeBufferSize = size;
	MUTEX_UNLOCK(writeMutex);
}


void RollingManager::writeBack()
{
	ProtSubRegion *r = regionRolling[Context::current()].front();
	regionRolling[Context::current()].pop_front();
	waitForWrite(r->start(), r->size());
	mlock(writeBuffer, writeBufferSize);
	assert(r->context()->copyToDeviceAsync(safe(r->start()), r->start(),
		r->size()) == gmacSuccess);
	r->readOnly();
}


void RollingManager::flushToDevice() 
{
	waitForWrite();
	Rolling::iterator i;
	for(i = regionRolling[Context::current()].begin();
			i != regionRolling[Context::current()].end(); i++) {
		assert((*i)->context()->copyToDevice(safe((*i)->start()),
				(*i)->start(), (*i)->size()) == gmacSuccess);
		(*i)->readOnly();
		TRACE("Flush to Device %p", (*i)->start()); 
	}
	regionRolling[Context::current()].clear();
}

#ifdef DEBUG
void RollingManager::dumpRolling()
{
	std::map<Context *, Rolling>::const_iterator c;
	for(c = regionRolling.begin(); c != regionRolling.end(); c++) {
		Rolling::const_iterator i;
		for(i = c->second.begin(); i != c->second.end(); i++)
			TRACE("Context %p: Region %p (%p - %d bytes)", c->first, *i,
					(*i)->start(), (*i)->size());
	}
}
#endif

RollingManager::RollingManager() :
	MemHandler(),
	lineSize(0),
	lruDelta(0),
	lruSize(0),
	pageSize(getpagesize()),
	writeBuffer(NULL),
	writeBufferSize(0)
{
	MUTEX_INIT(writeMutex);
	const char *var = Util::getenv(lineSizeVar);
	if(var != NULL) lineSize = atoi(var);
	if(lineSize == 0) lineSize = 1024;
	var = Util::getenv(lruDeltaVar);
	if(var != NULL) lruDelta = atoi(var);
	if(lruDelta == 0) lruDelta = 2;
	TRACE("Using %d as Memory Block Size", lineSize * pageSize);
	TRACE("Using %d as LRU Delta Size", lruDelta);
#ifdef DEBUG
	dumpRolling();
#endif
}


// MemManager Interface

bool RollingManager::alloc(void *addr, size_t size)
{
	if(map(addr, size, PROT_NONE) == MAP_FAILED) return false;
	TRACE("Alloc %p (%d bytes)", addr, size);
	lruSize += lruDelta;
	insert(new RollingRegion(*this, addr, size, lineSize * pageSize));
	return true;
}


void *RollingManager::safeAlloc(void *addr, size_t size)
{
	void *cpuAddr = NULL;
	if((cpuAddr = safeMap(addr, size, PROT_NONE)) == MAP_FAILED) return NULL;
	TRACE("SafeAlloc %p (%d bytes)", cpuAddr, size);
	lruSize += lruDelta;
	insert(new RollingRegion(*this, cpuAddr, size, lineSize * pageSize));
	return cpuAddr;
}


void RollingManager::release(void *addr)
{
	RollingRegion *reg = dynamic_cast<RollingRegion *>(remove(addr));
	unmap(reg->start(), reg->size());
	delete reg;
	lruSize -= lruDelta;
	TRACE("Released %p", addr);
#ifdef DEBUG
	dumpRolling();
#endif
}


void RollingManager::flush()
{
	TRACE("RollingManager Flush Starts");
	flushToDevice();
	MemMap::iterator i;
	current().lock();
	for(i = current().begin(); i != current().end(); i++) {
		RollingRegion *r = dynamic_cast<RollingRegion *>(i->second);
		r->invalidate();
	}
	current().unlock();
	TRACE("RollingManager Flush Ends");
}

Context *RollingManager::owner(const void *addr)
{
	RollingRegion *reg= get(addr);
	if(reg == NULL) return NULL;
	return reg->context();
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

// MemHandler Interface

bool RollingManager::read(void *addr)
{
	RollingRegion *root = get(addr);
	if(root == NULL) return false;
	ProtRegion *region = root->find(addr);
	assert(region != NULL);
	assert(region->present() == false);
	region->readWrite();
	assert(region->context()->copyToHost(region->start(),
			safe(region->start()), region->size()) == gmacSuccess);
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
	while(regionRolling[Context::current()].size() >= lruSize) writeBack();
	region->readWrite();
	regionRolling[Context::current()].push_back(dynamic_cast<ProtSubRegion *>(region));
#ifdef DEBUG
	dumpRolling();
#endif
	return true;
}


};

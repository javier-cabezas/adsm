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
		current->sync();
		munlock(writeBuffer, writeBufferSize);
	}
	writeBuffer = addr;
	writeBufferSize = size;
	MUTEX_UNLOCK(writeMutex);
}


void RollingManager::writeBack()
{
	ProtSubRegion *r = regionRolling[current].front();
	regionRolling[current].pop_front();
	waitForWrite(r->getAddress(), r->getSize());
	mlock(writeBuffer, writeBufferSize);
	r->context()->copyToDeviceAsync(safe(r->getAddress()), r->getAddress(),
		r->getSize());
	r->readOnly();
}


void RollingManager::flushToDevice() 
{
	waitForWrite();
	Rolling::iterator i;
	for(i = regionRolling[current].begin(); i != regionRolling[current].end();
			i++) {
		(*i)->context()->copyToDevice(safe((*i)->getAddress()),
				(*i)->getAddress(), (*i)->getSize());
		(*i)->readOnly();
		TRACE("Flush to Device %p", (*i)->getAddress()); 
	}
	regionRolling[current].clear();
}

#ifdef DEBUG
void RollingManager::dumpRolling()
{
	std::map<Context *, Rolling>::const_iterator c;
	for(c = regionRolling.begin(); c != regionRolling.end(); c++) {
		Rolling::const_iterator i;
		for(i = c->second.begin(); i != c->second.end(); i++)
			TRACE("Context %p: Region %p (%p - %d bytes)", c->first, *i,
					(*i)->getAddress(), (*i)->getSize());
	}
}
#endif

RollingManager::RollingManager() :
	MemManager(),
	MemHandler(mem),
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
	unmap(reg->getAddress(), reg->getSize());
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
	MemMap &mm = current->mm();
	mm.lock();
	for(i = mm.begin(); i != mm.end(); i++) {
		RollingRegion *r = dynamic_cast<RollingRegion *>(i->second);
		r->invalidate();
	}
	mm.unlock();
	TRACE("RollingManager Flush Ends");
}

void RollingManager::flush(MemRegion *region)
{
	RollingRegion *r = dynamic_cast<RollingRegion *>(region);
	Rolling::iterator i;
	for(i = regionRolling[current].begin();
			i != regionRolling[current].end();) {
		(*i)->context()->copyToDevice(safe((*i)->getAddress()),
				(*i)->getAddress(), (*i)->getSize());
		(*i)->readOnly();
		i = regionRolling[current].erase(i);
	}
	r->invalidate();
}

void RollingManager::dirty(MemRegion *region)
{
	RollingRegion *r = dynamic_cast<RollingRegion *>(region);
	r->dirty();
}

bool RollingManager::present(MemRegion *region) const
{
	RollingRegion *r = dynamic_cast<RollingRegion *>(region);
	return r->present();
}

// MemHandler Interface

void RollingManager::read(ProtRegion *region, void *addr)
{
	assert(region->present() == false);
	region->readWrite();
	region->context()->copyToHost(region->getAddress(),
			safe(region->getAddress()), region->getSize());
	region->readOnly();
}


void RollingManager::write(ProtRegion *region, void *addr)
{
	assert(region->dirty() == false);
	while(regionRolling[current].size() >= lruSize) writeBack();
	region->readWrite();
	regionRolling[current].push_back(dynamic_cast<ProtSubRegion *>(region));
#ifdef DEBUG
	dumpRolling();
#endif

}


};

#include "RollingManager.h"
#include "os/Util.h"
#include <api/api.h>

#include <unistd.h>
#include <algorithm>

namespace gmac {

const char *RollingManager::lineSizeVar = "GMAC_LINESIZE";
const char *RollingManager::lruDeltaVar = "GMAC_LRUDELTA";

void RollingManager::waitForWrite(void *addr, size_t size)
{
	MUTEX_LOCK(writeMutex);
	if(writeBuffer) {
		__gmacThreadSynchronize();
		munlock(writeBuffer, writeBufferSize);
	}
	writeBuffer = addr;
	writeBufferSize = size;
	MUTEX_UNLOCK(writeMutex);
}


void RollingManager::writeBack(thread_t tid)
{
	ProtSubRegion *r = regionRolling[tid].front();
	regionRolling[tid].pop_front();
	waitForWrite(r->getAddress(), r->getSize());
	mlock(writeBuffer, writeBufferSize);
	__gmacMemcpyToDeviceAsync(safe(r->getAddress()), r->getAddress(),
		r->getSize());
	r->readOnly();
}


void RollingManager::flushToDevice(thread_t tid) 
{
	waitForWrite();
	Rolling::iterator i;
	for(i = regionRolling[tid].begin(); i != regionRolling[tid].end(); i++) {
		__gmacMemcpyToDevice(safe((*i)->getAddress()), (*i)->getAddress(),
				(*i)->getSize());
		(*i)->readOnly();
		TRACE("Flush to Device %p", (*i)->getAddress()); 
	}
	regionRolling[tid].clear();
}

#ifdef DEBUG
void RollingManager::dumpRolling()
{
	std::map<thread_t, Rolling>::const_iterator c;
	for(c = regionRolling.begin(); c != regionRolling.end(); c++) {
		Rolling::const_iterator i;
		for(i = c->second.begin(); i != c->second.end(); i++)
			TRACE("Thread %d: Region %p (%p - %d bytes)", c->first, *i,
					(*i)->getAddress(), (*i)->getSize());
	}
}
#endif

RollingManager::RollingManager() :
	MemManager(),
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
	memMap.insert(new RollingRegion(*this, addr, size, lineSize * pageSize));
	return true;
}


void *RollingManager::safeAlloc(void *addr, size_t size)
{
	void *cpuAddr = NULL;
	if((cpuAddr = safeMap(addr, size, PROT_NONE)) == MAP_FAILED) return NULL;
	TRACE("SafeAlloc %p (%d bytes)", cpuAddr, size);
	lruSize += lruDelta;
	memMap.insert(new RollingRegion(*this, cpuAddr, size, lineSize * pageSize));
	return cpuAddr;
}


void RollingManager::release(void *addr)
{
	RollingRegion *reg = memMap.remove(addr);
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
	flushToDevice(Process::gettid());
	MemMap<RollingRegion>::iterator i;
	memMap.lock();
	for(i = memMap.begin(); i != memMap.end(); i++) {
		if(i->second->isOwner()) 
			i->second->invalidate();
	}
	memMap.unlock();
	TRACE("RollingManager Flush Ends");
}

void RollingManager::flush(MemRegion *region)
{
	RollingRegion *r = dynamic_cast<RollingRegion *>(region);
	thread_t tid = Process::gettid();
	Rolling::iterator i;
	for(i = regionRolling[tid].begin(); i != regionRolling[tid].end();) {
		if((*i)->belongs(r)) {
			__gmacMemcpyToDevice(safe((*i)->getAddress()), (*i)->getAddress(),
					(*i)->getSize());
			(*i)->readOnly();
			i = regionRolling[tid].erase(i);
		}
		else i++;
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
	return r->isPresent();
}

// MemHandler Interface

ProtRegion *RollingManager::find(void *addr)
{
	RollingRegion *r = memMap.find(addr);
	if(r) return r->find(addr);
	return NULL;
}


void RollingManager::read(ProtRegion *region, void *addr)
{
	assert(region->isPresent() == false);
	region->readWrite();
	__gmacMemcpyToHost(region->getAddress(), safe(region->getAddress()),
			region->getSize());
	region->readOnly();
}


void RollingManager::write(ProtRegion *region, void *addr)
{
	assert(region->isDirty() == false);
	thread_t tid = Process::gettid();
	while(regionRolling[tid].size() >= lruSize) writeBack(tid);
	region->readWrite();
	regionRolling[tid].push_back(dynamic_cast<ProtSubRegion *>(region));
#ifdef DEBUG
	dumpRolling();
#endif

}


};

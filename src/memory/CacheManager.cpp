#include "CacheManager.h"
#include "os/Util.h"
#include <acc/api.h>

#include <unistd.h>
#include <algorithm>

namespace gmac {

const char *CacheManager::lineSizeVar = "GMAC_LINESIZE";
const char *CacheManager::lruDeltaVar = "GMAC_LRUDELTA";

void CacheManager::waitForWrite(void *addr, size_t size)
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


void CacheManager::writeBack(thread_t tid)
{
	ProtSubRegion *r = regionCache[tid].front();
	regionCache[tid].pop_front();
	waitForWrite(r->getAddress(), r->getSize());
	mlock(writeBuffer, writeBufferSize);
	__gmacMemcpyToDeviceAsync(safe(r->getAddress()), r->getAddress(),
		r->getSize());
	r->readOnly();
}


void CacheManager::flushToDevice(thread_t tid) 
{
	waitForWrite();
	Cache::iterator i;
	for(i = regionCache[tid].begin(); i != regionCache[tid].end(); i++) {
		__gmacMemcpyToDevice(safe((*i)->getAddress()), (*i)->getAddress(),
				(*i)->getSize());
		(*i)->readOnly();
		TRACE("Flush to Device %p", (*i)->getAddress()); 
	}
	regionCache[tid].clear();
}

#ifdef DEBUG
void CacheManager::dumpCache()
{
	std::map<thread_t, Cache>::const_iterator c;
	for(c = regionCache.begin(); c != regionCache.end(); c++) {
		Cache::const_iterator i;
		for(i = c->second.begin(); i != c->second.end(); i++)
			TRACE("Thread %d: Region %p (%p - %d bytes)", c->first, *i,
					(*i)->getAddress(), (*i)->getSize());
	}
}
#endif

CacheManager::CacheManager() :
	MemManager(),
	lineSize(0),
	lruDelta(0),
	lruSize(0),
	pageSize(getpagesize()),
	writeBuffer(NULL),
	writeBufferSize(0)
{
	const char *var = Util::getenv(lineSizeVar);
	if(var != NULL) lineSize = atoi(var);
	if(lineSize == 0) lineSize = 1024;
	var = Util::getenv(lruDeltaVar);
	if(var != NULL) lruDelta = atoi(var);
	if(lruDelta == 0) lruDelta = 2;
	TRACE("Using %d as Memory Block Size", lineSize * pageSize);
	TRACE("Using %d as LRU Delta Size", lruDelta);
#ifdef DEBUG
	dumpCache();
#endif
}


// MemManager Interface

bool CacheManager::alloc(void *addr, size_t size)
{
	if(map(addr, size, PROT_NONE) == MAP_FAILED) return false;
	TRACE("Alloc %p (%d bytes)", addr, size);
	lruSize += lruDelta;
	memMap.insert(new CacheRegion(*this, addr, size, lineSize * pageSize));
	return true;
}


void *CacheManager::safeAlloc(void *addr, size_t size)
{
	void *cpuAddr = NULL;
	if((cpuAddr = safeMap(addr, size, PROT_NONE)) == MAP_FAILED) return NULL;
	TRACE("SafeAlloc %p (%d bytes)", cpuAddr, size);
	lruSize += lruDelta;
	memMap.insert(new CacheRegion(*this, cpuAddr, size, lineSize * pageSize));
	return cpuAddr;
}


void CacheManager::release(void *addr)
{
	CacheRegion *reg = memMap.remove(addr);
	unmap(reg->getAddress(), reg->getSize());
	delete reg;
	lruSize -= lruDelta;
	TRACE("Released %p", addr);
#ifdef DEBUG
	dumpCache();
#endif
}


void CacheManager::flush()
{
	TRACE("CacheManager Flush Starts");
	flushToDevice(Process::gettid());
	MemMap<CacheRegion>::iterator i;
	memMap.lock();
	for(i = memMap.begin(); i != memMap.end(); i++) {
		if(i->second->isOwner()) 
			i->second->invalidate();
	}
	memMap.unlock();
	TRACE("CacheManager Flush Ends");
}

void CacheManager::flush(MemRegion *region)
{
	CacheRegion *r = dynamic_cast<CacheRegion *>(region);
	thread_t tid = Process::gettid();
	Cache::iterator i;
	for(i = regionCache[tid].begin(); i != regionCache[tid].end();) {
		if((*i)->belongs(r)) {
			__gmacMemcpyToDevice(safe((*i)->getAddress()), (*i)->getAddress(),
					(*i)->getSize());
			(*i)->readOnly();
			i = regionCache[tid].erase(i);
		}
		else i++;
	}
	r->invalidate();
}

void CacheManager::dirty(MemRegion *region)
{
	CacheRegion *r = dynamic_cast<CacheRegion *>(region);
	r->dirty();
}

bool CacheManager::present(MemRegion *region) const
{
	CacheRegion *r = dynamic_cast<CacheRegion *>(region);
	return r->isPresent();
}

// MemHandler Interface

ProtRegion *CacheManager::find(void *addr)
{
	CacheRegion *r = memMap.find(addr);
	if(r) return r->find(addr);
	return NULL;
}


void CacheManager::read(ProtRegion *region, void *addr)
{
	assert(region->isPresent() == false);
	region->readWrite();
	__gmacMemcpyToHost(region->getAddress(), safe(region->getAddress()),
			region->getSize());
	region->readOnly();
}


void CacheManager::write(ProtRegion *region, void *addr)
{
	assert(region->isDirty() == false);
	thread_t tid = Process::gettid();
	while(regionCache[tid].size() >= lruSize) writeBack(tid);
	region->readWrite();
	regionCache[tid].push_back(dynamic_cast<ProtSubRegion *>(region));
#ifdef DEBUG
	dumpCache();
#endif

}


};

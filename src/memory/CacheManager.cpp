#include "CacheManager.h"
#include <acc/api.h>

#include <unistd.h>
#include <algorithm>

namespace gmac {

void CacheManager::writeBack(thread_t tid)
{
	TRACE("Write Back");
	ProtRegion *r = regionCache[tid].front();
	regionCache[tid].pop_front();
	// If the write buffer was in use, make sure that the previous
	// transfer has already finished
	if(writeBuffer) {
		__gmacThreadSynchronize();
		munlock(writeBuffer, writeBufferSize);
	}
	// Set-up the new write buffer and send an asynchronous copy
	writeBuffer = r->getAddress();
	writeBufferSize = r->getSize();
	mlock(writeBuffer, writeBufferSize);
	__gmacMemcpyToDeviceAsync(safe(r->getAddress()), r->getAddress(), r->getSize());
	r->readOnly();
}

void CacheManager::flushToDevice(thread_t tid) 
{
	Cache::iterator i;
	// Wait for the write buffer
	if(writeBuffer) {
		__gmacThreadSynchronize();
		munlock(writeBuffer, writeBufferSize);
		writeBuffer = NULL;
	}
	for(i = regionCache[tid].begin(); i != regionCache[tid].end(); i++) {
		TRACE("DMA to Device from %p (%d bytes)", (*i)->getAddress(),
				(*i)->getSize());
		__gmacMemcpyToDevice(safe((*i)->getAddress()), (*i)->getAddress(), (*i)->getSize());
		(*i)->readOnly();
	}
	regionCache[tid].clear();
}

CacheManager::CacheManager() :
	MemManager(),
	lruSize(0),
	pageSize(getpagesize()),
	writeBuffer(NULL),
	writeBufferSize(0)
{
	MUTEX_INIT(memMutex);
}

bool CacheManager::alloc(void *addr, size_t size)
{
	if(map(addr, size, PROT_NONE) == MAP_FAILED) return false;
	lruSize++;
	MUTEX_LOCK(memMutex);
	memMap[addr] = new CacheRegion(addr, size, lineSize * pageSize);
	MUTEX_UNLOCK(memMutex);
	return true;
}

void *CacheManager::safeAlloc(void *addr, size_t size)
{
	void *cpuAddr = NULL;
	if((cpuAddr = safeMap(addr, size, PROT_NONE)) == MAP_FAILED) return NULL;
	lruSize++;
	MUTEX_LOCK(memMutex);
	memMap[cpuAddr] = new CacheRegion(cpuAddr, size, lineSize * pageSize);
	MUTEX_UNLOCK(memMutex);
	return cpuAddr;
}

void CacheManager::release(void *addr)
{
	Map::iterator i;
	MUTEX_LOCK(memMutex);
	if((i = memMap.find(addr)) != memMap.end()) {
		if(i->second == NULL) FATAL("Double-free for %p\n", addr);
		delete i->second;
		memMap.erase(addr);
	}
	MUTEX_UNLOCK(memMutex);
	lruSize--;
	TRACE("Released %p", addr);
}

void CacheManager::flush()
{
	Map::const_iterator i;

	flushToDevice(Process::gettid());

	MUTEX_LOCK(memMutex);
	for(i = memMap.begin(); i != memMap.end(); i++) {
		if(i->second->isOwner() == false) continue;
		i->second->invalidate();
	}
	MUTEX_UNLOCK(memMutex);
}

void CacheManager::sync()
{
}

ProtRegion *CacheManager::find(const void *addr)
{
	HASH_MAP<void *, CacheRegion *>::const_iterator i;
	MUTEX_LOCK(memMutex);
	for(i = memMap.begin(); i != memMap.end(); i++) {
		if(*(i->second) == addr) {
			MUTEX_UNLOCK(memMutex);
			return i->second->find(addr);
		}
	}
	MUTEX_UNLOCK(memMutex);
	return NULL;
}

void CacheManager::read(ProtRegion *region, void *addr)
{
	TRACE("DMA from Device %p (%d bytes)", region->getAddress(), region->getSize());
	region->readWrite();
	__gmacMemcpyToHost(region->getAddress(), safe(region->getAddress()), region->getSize());
	region->readOnly();
}


void CacheManager::write(ProtRegion *region, void *addr)
{
	thread_t tid = Process::gettid();
	if(regionCache[tid].size() == lruSize) writeBack(tid);
	region->readWrite();
	regionCache[tid].push_back(dynamic_cast<ProtSubRegion *>(region));
}


};

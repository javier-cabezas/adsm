#include "CacheManager.h"
#include "debug.h"

#include <unistd.h>
#include <cuda_runtime.h>

#include <algorithm>

namespace gmac {

void CacheManager::writeBack(pthread_t tid)
{
	TRACE("Write Back");
	ProtRegion *r = regionCache[tid].front();
	regionCache[tid].pop_front();
	cudaMemcpy(safe(r->getAddress()), r->getAddress(), r->getSize(),
			cudaMemcpyHostToDevice);
	r->readOnly();
}

void CacheManager::flushToDevice(pthread_t tid) 
{
	Cache::iterator i;
	for(i = regionCache[tid].begin(); i != regionCache[tid].end(); i++) {
		TRACE("DMA to Device from %p (%d bytes)", (*i)->getAddress(),
				(*i)->getSize());
		cudaMemcpy(safe((*i)->getAddress()), (*i)->getAddress(),
			(*i)->getSize(), cudaMemcpyHostToDevice);
		(*i)->readOnly();
	}
	regionCache[tid].clear();
}

CacheManager::CacheManager() :
	MemManager(),
	pageSize(getpagesize())
{
	MUTEX_INIT(memMutex);
}

bool CacheManager::alloc(void *addr, size_t size)
{
	if(map(addr, size, PROT_NONE) == MAP_FAILED) return false;
	MUTEX_LOCK(memMutex);
	memMap[addr] = new CacheRegion(addr, size, lineSize * pageSize);
	MUTEX_UNLOCK(memMutex);
	return true;
}

void *CacheManager::safeAlloc(void *addr, size_t size)
{
	void *cpuAddr = NULL;
	if((cpuAddr = safeMap(addr, size, PROT_NONE)) == MAP_FAILED) return NULL;
	MUTEX_LOCK(memMutex);
	memMap[cpuAddr] = new CacheRegion(cpuAddr, size, lineSize * pageSize);
	MUTEX_UNLOCK(memMutex);
	return cpuAddr;
}

void CacheManager::release(void *addr)
{
	void *cpuAddr = safe(addr);
	MUTEX_LOCK(memMutex);
	if(memMap.find(cpuAddr) != memMap.end())
		delete memMap[addr];
	MUTEX_UNLOCK(memMutex);
}

void CacheManager::execute()
{
	Map::const_iterator i;

	TRACE("Kernel execution scheduled");

	flushToDevice(gettid());

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
	region->readWrite();
	cudaMemcpy(region->getAddress(), safe(region->getAddress()), region->getSize(), cudaMemcpyDeviceToHost);
	region->readOnly();
}


void CacheManager::write(ProtRegion *region, void *addr)
{
	pthread_t tid = gettid();
	if(regionCache[tid].size() == lruSize) writeBack(tid);
	region->readWrite();
	regionCache[tid].push_back(dynamic_cast<ProtSubRegion *>(region));
}


};

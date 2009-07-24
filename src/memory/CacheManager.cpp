#include "CacheManager.h"
#include <acc/api.h>

#include <unistd.h>
#include <algorithm>

namespace gmac {

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
	ProtRegion *r = regionCache[tid].front();
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


CacheManager::CacheManager() :
	MemManager(),
	lruSize(0),
	pageSize(getpagesize()),
	writeBuffer(NULL),
	writeBufferSize(0)
{
}


// MemManager Interface

bool CacheManager::alloc(void *addr, size_t size)
{
	if(map(addr, size, PROT_NONE) == MAP_FAILED) return false;
	TRACE("Alloc %p (%d bytes)", addr, size);
	lruSize += 2;
	memMap.insert(new CacheRegion(*this, addr, size, lineSize * pageSize));
	return true;
}


void *CacheManager::safeAlloc(void *addr, size_t size)
{
	void *cpuAddr = NULL;
	if((cpuAddr = safeMap(addr, size, PROT_NONE)) == MAP_FAILED) return NULL;
	TRACE("SafeAlloc %p (%d bytes)", cpuAddr, size);
	lruSize += 2;
	memMap.insert(new CacheRegion(*this, addr, size, lineSize * pageSize));
	return cpuAddr;
}


void CacheManager::release(void *addr)
{
	CacheRegion *reg = memMap.remove(addr);
	unmap(reg->getAddress(), reg->getSize());
	delete reg;
	lruSize -= 2;
	TRACE("Released %p", addr);
}


void CacheManager::flush()
{
	flushToDevice(Process::gettid());
	MemMap<CacheRegion>::iterator i;
	memMap.lock();
	for(i = memMap.begin(); i != memMap.end(); i++) {
		if(i->second->isOwner()) 
			i->second->invalidate();
	}
	memMap.unlock();
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
	region->readWrite();
	__gmacMemcpyToHost(region->getAddress(), safe(region->getAddress()),
			region->getSize());
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

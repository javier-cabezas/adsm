#include "CacheManager.h"
#include "debug.h"

#include <unistd.h>
#include <cuda_runtime.h>

#include <algorithm>

namespace gmac {

CacheRegion::CacheRegion(MemHandler &memHandler, void *addr, size_t size,
		size_t cacheLine) :
	MemRegion(addr, size),
	memHandler(memHandler),
	cacheLine(cacheLine)
{
	for(size_t s = 0; s < size; s += cacheLine) {
		void *p = (void *)((uint8_t *)addr + s);
		cache.push_back(new ProtRegion(memHandler, p, cacheLine));
	}
}

CacheRegion::~CacheRegion()
{
	std::vector<ProtRegion *>::const_iterator i;
	for(i = cache.begin(); i != cache.end(); i++)
		delete (*i);
	cache.clear();
}

void CacheManager::writeBack()
{
	TRACE("Write Back");
	if(writeBuffer) {
		cudaThreadSynchronize();
		munlock(writeBuffer->getAddress(), writeBuffer->getSize());
	}
	ProtRegion *r = lru.front();
	mlock(r->getAddress(), r->getSize());
	cudaMemcpyAsync(r->getAddress(), r->getAddress(), r->getSize(),
			cudaMemcpyHostToDevice, 0);
	r->noAccess();
	lru.pop_front();
	writeBuffer = r;
}

void CacheManager::dmaToDevice(std::vector<ProtRegion *> &cache)
{
	std::vector<ProtRegion *>::iterator i;
	for(i = cache.begin(); i != cache.end(); i++) {
		if((*i)->isDirty()) {
			TRACE("DMA to Device from %p (%d bytes)", (*i)->getAddress(),
					(*i)->getSize());
			cudaMemcpy((*i)->getAddress(), (*i)->getAddress(), (*i)->getSize(),
					cudaMemcpyHostToDevice);
		}
		(*i)->clear();
		(*i)->noAccess();
	}
}

CacheManager::CacheManager() :
	pageSize(getpagesize()),
	writeBuffer(NULL)
{
}

bool CacheManager::alloc(void *addr, size_t size)
{
	if(map(addr, size, PROT_NONE) == MAP_FAILED) return false;
	memMap[addr] = new CacheRegion(*this, addr, size, lineSize * pageSize);
	return true;
}

void CacheManager::release(void *addr)
{
	if(memMap.find(addr) == memMap.end()) return;
	delete memMap[addr];
}

void CacheManager::execute()
{
	TRACE("Kernel execution scheduled");
	HASH_MAP<void *, CacheRegion *>::const_iterator i;
	for(i = memMap.begin(); i != memMap.end(); i++) {
		dmaToDevice(i->second->getCache());
	}
}

void CacheManager::sync()
{
}

void CacheManager::read(ProtRegion *region, void *addr)
{
	region->incAccess();
	TRACE("DMA from Device from %p (%d bytes)", region->getAddress(),
			region->getSize());
	region->readWrite();
	cudaMemcpy(region->getAddress(), region->getAddress(), region->getSize(),
			cudaMemcpyDeviceToHost);
	region->readOnly();
}

void CacheManager::write(ProtRegion *region, void *addr)
{
	if(lru.size() == lruSize) writeBack();
	region->incAccess();
	region->setDirty();
	region->readWrite();
	lru.push_back(region);
}

};

#include "LazyManager.h"

#include "debug.h"

#include <cuda_runtime.h>

namespace gmac {

bool LazyManager::alloc(void *addr, size_t count)
{
	if(map(addr, count, PROT_NONE) == MAP_FAILED) return false;
	TRACE("Alloc %p (%d bytes)", addr, count);
	ProtRegion *region = new ProtRegion(*this, addr, count);
	memMap[addr] = region;
	return true;
}

void *LazyManager::safeAlloc(void *addr, size_t count)
{
	void *cpuAddr = NULL;
	if((cpuAddr = safeMap(addr, count, PROT_NONE)) == MAP_FAILED) return NULL;
	TRACE("SafeAlloc %p (%d bytes)", cpuAddr, count);
	ProtRegion *region = new ProtRegion(*this, cpuAddr, count);
	memMap[cpuAddr] = region;
	return cpuAddr;
}

void LazyManager::release(void *addr)
{
	HASH_MAP<void *, ProtRegion *>::const_iterator i;
	i = memMap.find(addr);
	if(i != memMap.end()) {
		unmap(addr, i->second->getSize());
		delete i->second;
		memMap.erase(addr);
	}
}

void LazyManager::execute()
{
	TRACE("Kernel execution scheduled");
	HASH_MAP<void *, ProtRegion *>::const_iterator i;
	for(i = memMap.begin(); i != memMap.end(); i++) {
		if(i->second->isDirty()) {
			TRACE("DMA to Device from %p (%d bytes)", i->first,
				i->second->getSize());
			cudaMemcpy(safe(i->first), i->first, i->second->getSize(),
				cudaMemcpyHostToDevice);
		}
		i->second->clear();
		i->second->noAccess();
	}
}

void LazyManager::sync()
{
}

void LazyManager::read(ProtRegion *region, void *addr)
{
	region->incAccess();
	TRACE("DMA from Device from %p (%d bytes)", region->getAddress(),
			region->getSize());
	region->readWrite();
	cudaMemcpy(region->getAddress(), safe(region->getAddress()), region->getSize(),
			cudaMemcpyDeviceToHost);
	region->readOnly();
}

void LazyManager::write(ProtRegion *region, void *addr)
{
	region->incAccess();
	region->setDirty();
	region->readWrite();
}

}

#ifndef __MEMORY_BATCHMANAGER_IPP_
#define __MEMORY_BATCHMANAGER_IPP_

inline void *
BatchManager::alloc(void *addr, size_t count)
{
    void *cpuAddr = hostMap(addr, count);

	insertVirtual(cpuAddr, addr, count);
    insert(new Region(cpuAddr, count));
	TRACE("Alloc %p (%d bytes)", cpuAddr, count);
    return cpuAddr;
}

#endif

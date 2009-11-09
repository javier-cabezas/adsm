#include <memory/Manager.h>
#include <debug.h>

namespace gmac { namespace memory {

void *Manager::hostMap(void *addr, size_t count, int prot)
{
		void *cpuAddr = mmap(NULL, count, prot, MAP_32BIT | MAP_ANON | MAP_PRIVATE, 0, 0);
		if(cpuAddr == MAP_FAILED) return NULL;
		insertVirtual(cpuAddr, addr, count);
		return cpuAddr;
}


void Manager::hostUnmap(void *addr, size_t count)
{
	munmap(addr, count);
}

} };

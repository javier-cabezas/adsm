#include <memory/MemManager.h>
#include <debug.h>

namespace gmac {

void *MemManager::map(void *addr, size_t count, int prot)
{
	void *cpuAddr = mmap((void *)addr, count, prot,
		MAP_ANON | MAP_FIXED | MAP_PRIVATE, 0, 0);

	/* NOTICE: mmap() might fail because the CPU address range is already
	 * in use by the code/data for the program. In this case we might
    * want to try a new allocation. We can try to use getpid() to check
    * our own memory map and play with _cudaMalloc, so we avoid it
    * returning an address within one of the in-use memory ranges.
    * This very same problem arises also in cudaMallocPitch.
    * I am trying to keep the code simple and I am not running into
    * this problem in the tests I am doing. In case this ever happends
    * a nice message is printed in the screen requesting the programmer
    * to ask for a fix.
    */
	if(cpuAddr == MAP_FAILED) {
		TRACE("FIXME: the address (%p) is already in use by the CPU.\n\
			Please ask %s for a fix", addr, PACKAGE_BUGREPORT);
	}

	insertVirtual(cpuAddr, addr, count);
	return cpuAddr;
}


void *MemManager::safeMap(void *addr, size_t count, int prot)
{
		void *cpuAddr = mmap(NULL, count, prot, MAP_ANON | MAP_PRIVATE, 0, 0);
		if(cpuAddr == MAP_FAILED) return NULL;
		insertVirtual(cpuAddr, addr, count);
		return cpuAddr;
}


void MemManager::unmap(void *addr, size_t count)
{
	munmap(addr, count);
}

};

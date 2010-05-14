#include <memory/Manager.h>

namespace gmac { namespace memory {

#ifndef HAVE_POSIX_MEMALIGN
static int custom_memalign(void **addr, size_t align, size_t count)
{
	uint8_t *aligned_buffer = NULL;
	uint8_t *buffer = (uint8_t *)malloc(count + align - 1 + sizeof(void **) + sizeof(int));
	if(buffer == NULL) return ENOMEM;
	aligned_buffer = buffer + align - 1 + sizeof(void **) + sizeof(int);
	aligned_buffer -= (unsigned long)aligned_buffer & (align - 1);
	*((void **)(aligned_buffer - sizeof(void **))) = buffer;
	*((int *)(aligned_buffer - sizeof(void **) - sizeof(int))) = count;
	*addr = (void *)aligned_buffer;
	return 0;
}

static void custom_free(void *p)
{
	::free(*(((void **) p) - 1)); 
}
#endif

void *Manager::mapToHost(void *addr, size_t count, int prot)
{
	void *cpuAddr = NULL;
#ifndef USE_MMAP
#ifdef HAVE_POSIX_MEMALIGN
	if(posix_memalign(&cpuAddr, pageTable().getPageSize(), count) != 0)
		return NULL;
#else
	if(custom_memalign(&cpuAddr, pageTable().getPageSize(), count) != 0)
		return NULL;
#endif
	Memory::protect(cpuAddr, count, prot);
#else
	cpuAddr = (void *)((uint8_t *)addr + Context::current()->id() * mmSize);
	if(mmap(cpuAddr, count, prot, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0) != cpuAddr)
		return NULL;
#endif
	return cpuAddr;
}

void *Manager::hostRemap(void *addr, void *hAddr, size_t count)
{
#if 0
	void *cpuAddr = NULL;
#ifdef USE_MMAP
	cpuAddr = (void *)((uint8_t *)addr + Context::current()->id() * mmSize);
	if(mremap(hAddr, count, count, MREMAP_FIXED, cpuAddr) != cpuAddr)
		return NULL;
#endif
	return cpuAddr;
#endif
    return NULL;
}

void Manager::hostUnmap(void *addr, size_t count)
{
#ifdef USE_GLOBAL_HOST
	logger.assertion(Map::isShared(addr) == false);
#endif

#ifndef USE_MMAP
#ifdef HAVE_POSIX_MEMALIGN
	::free(addr);
#else
    custom_free(addr);
#endif
#else
	munmap(addr, count);
#endif
}

}}

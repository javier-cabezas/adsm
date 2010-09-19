#include "Memory.h"

#include <kernel/Mode.h>

namespace gmac { namespace memory {

int Memory::protect(void *addr, size_t count, int prot)
{
    util::Logger::TRACE("Setting memory permisions to %d @ %p - %p", prot, addr, (uint8_t *)addr + count);
    int ret = mprotect(addr, count, prot);
    util::Logger::ASSERTION(ret == 0);
    return 0;
}

void *Memory::map(void *addr, size_t count, int prot)
{
    void *cpuAddr = NULL;

    if (addr == NULL) {
        cpuAddr = mmap(NULL, count, prot, MAP_PRIVATE | MAP_ANON, -1, 0);
    } else {
        cpuAddr = addr;
        if(mmap(cpuAddr, count, prot, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0) != cpuAddr)
            return NULL;
    }

    return cpuAddr;
}

void *Memory::remap(void *addr, void *to, size_t count, int prot)
{
    void * ret = mremap(addr, count, count, MREMAP_FIXED | MREMAP_MAYMOVE, to);
    if (ret != to) {
        return NULL;
    }

    return to;
}

void Memory::unmap(void *addr, size_t count)
{
    munmap(addr, count);
}

}}


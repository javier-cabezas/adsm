
#ifndef __MEMORY_POSIX_MEMORY_IPP_
#define __MEMORY_POSIX_MEMORY_IPP_

namespace gmac { namespace memory {

inline int
Memory::protect(void *addr, size_t count, int prot)
{
    util::Logger::TRACE("Setting memory permisions to %d @ %p - %p", prot, addr, (uint8_t *)addr + count);
    int ret = mprotect(addr, count, prot);
    util::Logger::ASSERTION(ret == 0);
    return 0;
}

inline void *
Memory::map(void *addr, size_t count, int prot)
{
    void *cpuAddr = NULL;

    if (addr == NULL) {
        cpuAddr = mmap(NULL, count, prot, MAP_PRIVATE | MAP_ANON, -1, 0);
    } else {
#ifdef USE_MMAP
        cpuAddr = (void *)((uint8_t *)addr + Mode::current().id() * Memory::mmSize);
#else
        cpuAddr = addr;
#endif
        if(mmap(cpuAddr, count, prot, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0) != cpuAddr)
            return NULL;
    }

    return cpuAddr;
}

inline void *
Memory::remap(void *addr, void *to, size_t count, int prot)
{
    void * ret = mremap(addr, count, count, MREMAP_FIXED | MREMAP_MAYMOVE, to);
    if (ret != to) {
        return NULL;
    }

    return to;
}

inline void
Memory::unmap(void *addr, size_t count)
{
    munmap(addr, count);
}

}}

#endif
